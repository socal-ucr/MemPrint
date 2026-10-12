/* libmpr: allocations served according to MemPrint's per-site reservations (see mpr.h).

   Environment:
     MPR_RESERVATIONS=<file>  reservations: "site model a b" per line (analysis/memprint/sites.py)
     MPR_MIN=<bytes>          blocks smaller than this go to glibc unchanged (default 65536)
     MPR_SCALE=<x>            multiply every prediction (robustness tests)
     MPR_ALL=1                serve every large block, with a reservation or not (no reservation:
                              committed in full); measures mmap, zero pages and mremap alone
     MPR_REPORT=<file>        at exit, per site: blocks, requested, predicted (committed), resident
                              at free (mincore), blocks that used more than was predicted, bytes
                              of zeroing skipped
     MPR_THP=0|1|2            transparent huge pages for runtime blocks: 0 none (MADV_NOHUGEPAGE),
                              1 as the system sets them, 2 (default) none on the pages whose zeroing
                              was skipped (below). With THP, one touch backs 2 MB, so a block cleared
                              while untouched and then used sparsely (a KV cache written a few rows
                              per layer) would become resident in 2 MB pieces after all
     MPR_ZERO=0               do not skip zeroing (default 1: memset(p, 0, n) over a runtime block
                              leaves the pages the program has never touched alone; they are
                              zero already, and writing zeros would make them resident)

   A site is the id the MemPrintResize pass passes (key "id:0x..."), or in a preloaded run the
   caller's return address as "image+0xaddress", the address in the image file: the key MemPrint's
   Pin tool writes (-sites). A site reached through an allocation wrapper can have reservations per
   calling context, "site@frame1[@frame2]" (frames: the return addresses above the allocation call,
   as keys); its blocks are then matched by the frames glibc's backtrace() finds, and a context with
   no line falls back to the site's own line, if it has one.

   Models (bytes predicted for a request of n bytes, clamped to [0, n]): dead (0), ratio (a n),
   affine (a n + b), const (a). */
#define _GNU_SOURCE
#include "mpr.h"

#include <dlfcn.h>
#include <execinfo.h>
#include <elf.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void __libc_free(void *);
extern void *__libc_memalign(size_t, size_t);

#define PAGE 4096UL
#define ROUND(n) (((n) + PAGE - 1) & ~(PAGE - 1))

enum model { M_NONE, M_DEAD, M_RATIO, M_AFFINE, M_CONST, M_CAP, M_RUNS };
#define MAX_RUNS 32

struct site {
    char key[128];
    int model;
    double a, b;
    /* telemetry */
    uint64_t count, requested, predicted, resident, under;
    atomic_uint_least64_t elided; /* bytes of memset(0) skipped on untouched pages */
    double max_resident_ratio;
    int ctx_depth; /* >0: the site has per-context lines of up to this many frames */
    int nruns;     /* M_RUNS: the committed ranges, as fractions of the block */
    float runs[MAX_RUNS][2];
    int has_model; /* the site has a line of its own (else: a base for context lines only) */
};

/* sites: from the reservation file, plus (MPR_ALL) sites without one; read-only keys after init */
#define MAX_SITES 8192
static struct site *sites; /* the tables are mapped MAP_NORESERVE at start-up: no commit of their own */
static int nsites;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

/* return address -> site (or NULL) */
#define CACHE 8192
static struct cache_entry { uintptr_t ret; struct site *site; int known; } *cache;

/* live blocks served by the runtime */
#define LIVE (1 << 16)
static struct block { uintptr_t ptr; size_t len, commit, requested; struct site *site; int shaped; } *live;
static size_t nlive;
static uintptr_t *starts; /* the live blocks' addresses, sorted: the block holding an address (zeroing) */

static int ready;
static size_t min_size = 65536;
static double scale = 1;
static int serve_all;
static const char *report;
static atomic_size_t committed;
static int debug; /* MPR_DEBUG=1: every runtime block's events on stderr */
static int zero_skip = 1; /* MPR_ZERO */
static int thp = 2;       /* MPR_THP */
static int pagemap = -1;  /* /proc/self/pagemap, for zero_skip */
static void *(*libc_memset)(void *, int, size_t);
#define TRACE(...) do { if (debug) dprintf(2, "libmpr: " __VA_ARGS__); } while (0)

static uint64_t hash(uintptr_t x)
{
    x ^= x >> 33, x *= 0xff51afd7ed558ccdULL, x ^= x >> 33;
    return x;
}

static struct site *find_key(const char *key)
{
    for (int i = 0; i < nsites; i++)
        if (strcmp(sites[i].key, key) == 0) return &sites[i];
    return NULL;
}

static struct site *add_site(const char *key, int model, double a, double b)
{
    if (nsites == MAX_SITES) return NULL;
    struct site *s = &sites[nsites++];
    memset(s, 0, sizeof *s);
    snprintf(s->key, sizeof s->key, "%s", key);
    s->model = model, s->a = a, s->b = b;
    s->has_model = 1;
    return s;
}

/* Builds: "# image NAME build-id HEX" lines in the reservations give the build each image's
   address keys were profiled on. An image loaded with another build-id (or none) has its address
   keys ignored: an address in one build is unrelated code in another. Ids from the pass do not
   depend on the build. */
#define MAX_IMAGES 64
static struct image { char name[96], want[48]; int state; /* 0 unchecked, 1 same build, -1 another */ } images[MAX_IMAGES];
static int nimages;

static void parse_image(const char *line)
{
    char name[96], id[48];
    if (nimages < MAX_IMAGES && sscanf(line, "# image %95s build-id %47s", name, id) == 2) {
        snprintf(images[nimages].name, sizeof images[nimages].name, "%s", name);
        snprintf(images[nimages].want, sizeof images[nimages].want, "%s", id);
        nimages++;
    }
}

/* the GNU build-id of the loaded ELF image whose header is at `base` (hex), or "" */
static void loaded_build_id(uintptr_t base, char *out, size_t len)
{
    out[0] = 0;
    const Elf64_Ehdr *eh = (const Elf64_Ehdr *)base;
    if (memcmp(eh->e_ident, ELFMAG, SELFMAG) != 0) return;
    uintptr_t bias = eh->e_type == ET_EXEC ? 0 : base;
    const Elf64_Phdr *ph = (const Elf64_Phdr *)(base + eh->e_phoff);
    for (int i = 0; i < eh->e_phnum; i++) {
        if (ph[i].p_type != PT_NOTE) continue;
        const char *p = (const char *)(bias + ph[i].p_vaddr), *end = p + ph[i].p_memsz;
        while (p + sizeof(Elf64_Nhdr) <= end) {
            const Elf64_Nhdr *n = (const Elf64_Nhdr *)p;
            const char *name = p + sizeof *n, *desc = name + ((n->n_namesz + 3) & ~3u);
            if (n->n_type == NT_GNU_BUILD_ID && n->n_namesz == 4 && !memcmp(name, "GNU", 4)) {
                size_t k = 0;
                for (unsigned j = 0; j < n->n_descsz && k + 3 <= len; j++)
                    k += (size_t)snprintf(out + k, len - k, "%02x", (unsigned char)desc[j]);
                return;
            }
            p = desc + ((n->n_descsz + 3) & ~3u);
        }
    }
}

/* whether address keys of the image named `name` (loaded at `base`) may be used */
static int image_ok(const char *name, uintptr_t base)
{
    for (int i = 0; i < nimages; i++) {
        if (strcmp(images[i].name, name)) continue;
        if (!images[i].state) {
            char got[48];
            loaded_build_id(base, got, sizeof got);
            images[i].state = strcmp(got, images[i].want) ? -1 : 1;
            if (images[i].state < 0)
                fprintf(stderr, "libmpr: %s is not the build its reservations were profiled on (build-id %s, "
                        "profiled %s): its address sites are ignored\n", name, *got ? got : "none", images[i].want);
        }
        return images[i].state > 0;
    }
    return 1; /* no build recorded: as before */
}

/* "image+0xaddress" for a return address: the address in the image file (Pin's IMG_LoadOffset) */
static int key_of(uintptr_t ret, char *key, size_t len)
{
    Dl_info info;
    if (!dladdr((void *)ret, &info) || !info.dli_fbase) return 0;
    uintptr_t base = (uintptr_t)info.dli_fbase;
    uintptr_t bias = ((Elf64_Ehdr *)base)->e_type == ET_EXEC ? 0 : base;
    const char *name = info.dli_fname && *info.dli_fname ? info.dli_fname : "";
    const char *slash = strrchr(name, '/');
    if (!*name) { /* the main program: its name from /proc */
        static char exe[256];
        if (!exe[0]) {
            ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
            exe[n > 0 ? n : 0] = 0;
        }
        name = exe, slash = strrchr(exe, '/');
    }
    if (!image_ok(slash ? slash + 1 : name, base)) return 0;
    snprintf(key, len, "%s+0x%lx", slash ? slash + 1 : name, (unsigned long)(ret - bias));
    return 1;
}

/* the cache slot of a return address (known or the free slot to fill); lock held */
static int cache_slot(uintptr_t ret)
{
    uint64_t h = hash(ret);
    for (int i = 0; i < CACHE; i++) {
        size_t j = (h + i) % CACHE;
        if (!cache[j].known || cache[j].ret == ret) return (int)j;
    }
    return -1;
}

/* site of a call. dladdr takes the loader's lock, so it runs without ours: a thread in dlopen
   holds the loader's lock and may allocate. */
static struct site *site_of_return(uintptr_t ret)
{
    pthread_mutex_lock(&lock);
    int j = cache_slot(ret);
    if (j >= 0 && cache[j].known) {
        struct site *s = cache[j].site;
        pthread_mutex_unlock(&lock);
        return s;
    }
    pthread_mutex_unlock(&lock);
    char key[128];
    int ok = key_of(ret, key, sizeof key);
    pthread_mutex_lock(&lock);
    struct site *s = NULL;
    if (ok) {
        s = find_key(key);
        if (!s && serve_all) s = add_site(key, M_NONE, 0, 0);
    }
    j = cache_slot(ret);
    if (j >= 0) cache[j].ret = ret, cache[j].site = s, cache[j].known = 1;
    pthread_mutex_unlock(&lock);
    return s;
}

static struct site *site_of_id(uint64_t id)
{
    char key[64];
    snprintf(key, sizeof key, "id:0x%lx", (unsigned long)id);
    pthread_mutex_lock(&lock);
    struct site *s = find_key(key);
    if (!s && serve_all) s = add_site(key, M_NONE, 0, 0);
    pthread_mutex_unlock(&lock);
    return s;
}

static __thread int in_backtrace; /* backtrace() may allocate (it loads libgcc_s on first use) */

/* A site with per-context lines: the line for the frames above the call that returns to `ret`
   (deepest first), else the site's own line, else none. */
static struct site *with_context(struct site *s, uintptr_t ret)
{
    if (!s || !s->ctx_depth || !ret || in_backtrace) return s && s->has_model ? s : NULL;
    void *frames[24];
    in_backtrace = 1;
    int n = backtrace(frames, 24);
    in_backtrace = 0;
    int i = 0;
    while (i < n && (uintptr_t)frames[i] != ret) i++;
    for (int depth = s->ctx_depth; depth > 0; depth--) {
        if (i + depth >= n) continue;
        char key[sizeof s->key];
        size_t len = (size_t)snprintf(key, sizeof key, "%s", s->key);
        int ok = 1;
        for (int d = 1; d <= depth && ok && len < sizeof key; d++) {
            key[len++] = '@';
            ok = key_of((uintptr_t)frames[i + d], key + len, sizeof key - len);
            len += strlen(key + len);
        }
        if (!ok) continue;
        pthread_mutex_lock(&lock);
        struct site *c = find_key(key);
        pthread_mutex_unlock(&lock);
        if (c) {
            TRACE("context %s\n", key);
            return c;
        }
    }
    return s->has_model ? s : NULL;
}

/* the runtime's blocks start on a page; glibc's (16 bytes into a chunk) almost never do */
#define MAYBE_OURS(p) ((p) && nlive && ((uintptr_t)(p) & (PAGE - 1)) == 0)

/* the model's value for a request of n (not clamped) */
static double model_value(const struct site *s, size_t n)
{
    double p;
    switch (s->model) {
    case M_DEAD: p = 0; break;
    case M_RATIO: p = s->a * n; break;
    case M_AFFINE: p = s->a * n + s->b; break;
    case M_CONST: p = s->a; break;
    case M_RUNS: p = 0; for (int i = 0; i < s->nruns; i++) p += (s->runs[i][1] - s->runs[i][0]) * n; break;
    default: p = n; /* no reservation (MPR_ALL): all of it */
    }
    if (s->model != M_NONE) p *= scale;
    return p < 0 ? 0 : p;
}

/* bytes to commit for a request of n: the model's value within [0, n] */
static size_t predict(const struct site *s, size_t n)
{
    double p = model_value(s, n);
    return p > n ? n : (size_t)p;
}

/* live table; lock held */
static struct block *slot(uintptr_t ptr, int insert)
{
    uint64_t h = hash(ptr);
    for (size_t i = 0; i < LIVE; i++) {
        struct block *b = &live[(h + i) % LIVE];
        if (b->ptr == ptr) return b;
        if (b->ptr == 0) return insert ? b : NULL;
    }
    return NULL;
}

/* position of the first start >= p in the sorted starts; lock held */
static size_t start_index(uintptr_t p)
{
    size_t lo = 0, hi = nlive;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (starts[mid] < p) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* a block enters the table (after it is slotted, before nlive counts it); lock held */
static void add_start(uintptr_t p)
{
    size_t i = start_index(p);
    memmove(starts + i + 1, starts + i, (nlive - i) * sizeof *starts);
    starts[i] = p;
}

/* the runtime block holding [p, p + n), or NULL; lock held */
static struct block *holding(uintptr_t p, size_t n)
{
    size_t i = start_index(p + 1);
    if (i == 0) return NULL;
    struct block *b = slot(starts[i - 1], 0);
    return b && p + n <= b->ptr + b->len ? b : NULL;
}

static void unslot(struct block *b)
{
    /* the sorted starts first */
    size_t k = start_index(b->ptr);
    if (k < nlive && starts[k] == b->ptr) memmove(starts + k, starts + k + 1, (nlive - k - 1) * sizeof *starts);
    /* backward-shift deletion keeps probe chains whole */
    size_t i = b - live;
    b->ptr = 0;
    for (size_t j = (i + 1) % LIVE; live[j].ptr; j = (j + 1) % LIVE) {
        size_t home = hash(live[j].ptr) % LIVE;
        if ((j > i && (home <= i || home > j)) || (j < i && home <= i && home > j)) {
            live[i] = live[j], live[j].ptr = 0, i = j;
        }
    }
    nlive--;
}

/* a mapping of len bytes whose first `commit` bytes are committed and the rest MAP_NORESERVE */
static void *map_block(size_t len, size_t commit, size_t align)
{
    size_t extra = align > PAGE ? align : 0;
    char *p = mmap(NULL, len + extra, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return NULL;
    if (extra) {
        char *q = (char *)(((uintptr_t)p + align - 1) & ~(uintptr_t)(align - 1));
        if (q > p) munmap(p, q - p);
        if (q + len < p + len + extra) munmap(q + len, p + len + extra - (q + len));
        p = q;
    }
    if (commit && mmap(p, commit, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED) {
        munmap(p, len);
        return NULL;
    }
    if (thp == 0) madvise(p, len, MADV_NOHUGEPAGE);
    return p;
}

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

/* Resize a runtime block from len to newlen bytes (page multiples) without copying it. The block is
   two mappings (the committed prefix and the MAP_NORESERVE rest), which mremap cannot take as one,
   so: a shrink unmaps the tail; a growth maps the new tail right after the block if that address
   is free, and otherwise moves each mapping, page tables and all, into a new reserved range. */
static void *resize_block(char *p, size_t len, size_t commit, size_t newlen)
{
    if (newlen == len) return p;
    if (newlen < len) {
        munmap(p + newlen, len - newlen);
        return p;
    }
    char *tail = mmap(p + len, newlen - len, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
    if (tail == p + len) {
        if (thp == 0) madvise(tail, newlen - len, MADV_NOHUGEPAGE);
        return p;
    }
    if (tail != MAP_FAILED) munmap(tail, newlen - len); /* an old kernel took the flag as a hint */
    char *q = mmap(NULL, newlen, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (q == MAP_FAILED) return MAP_FAILED;
    if (thp == 0) madvise(q, newlen, MADV_NOHUGEPAGE); /* (mremap carries each moved piece's own flag) */
    /* A piece moves as one mapping, or, when the kernel kept it as several (tails added in place
       and written separately are not merged), is copied: the data is always kept */
    size_t pieces[2][2] = {{0, commit}, {commit, len - commit}};
    for (int i = 0; i < 2; i++) {
        size_t off = pieces[i][0], n = pieces[i][1];
        if (!n || mremap(p + off, n, n, MREMAP_MAYMOVE | MREMAP_FIXED, q + off) != MAP_FAILED) continue;
        TRACE("move piece %d of %p (+%zu, %zu bytes): errno %d, copied\n", i, (void *)p, off, n, errno);
        /* a failed mremap(MREMAP_FIXED) has already unmapped its destination: map it again */
        if (mmap(q + off, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0)
            == MAP_FAILED) {
            munmap(q, newlen);
            return MAP_FAILED;
        }
        memcpy(q + off, p + off, n);
        munmap(p + off, n);
    }
    return q;
}

static size_t resident_bytes(void *p, size_t len)
{
    size_t pages = len / PAGE, n = 0;
    unsigned char vec[4096];
    for (size_t off = 0; off < pages; off += sizeof vec) {
        size_t k = pages - off < sizeof vec ? pages - off : sizeof vec;
        if (mincore((char *)p + off * PAGE, k * PAGE, vec) != 0) return 0;
        for (size_t i = 0; i < k; i++) n += vec[i] & 1;
    }
    return n * PAGE;
}

/* telemetry for a block about to go; lock held */
static void retire(struct block *b)
{
    struct site *s = b->site;
    size_t res = resident_bytes((void *)b->ptr, b->len);
    s->resident += res;
    if (res > b->commit) s->under++;
    double r = b->requested ? (double)res / b->requested : 0;
    if (r > s->max_resident_ratio) s->max_resident_ratio = r;
    atomic_fetch_sub(&committed, b->commit);
}

/* serve a large block for site s; NULL: let glibc do it */
/* M_RUNS: commit the site's ranges of a block mapped MAP_NORESERVE; returns the bytes committed */
static size_t commit_runs(char *p, size_t len, const struct site *s)
{
    size_t total = 0, pages = len / PAGE;
    for (int i = 0; i < s->nruns; i++) {
        size_t lo = (size_t)(s->runs[i][0] * pages) * PAGE, hi = ROUND((size_t)(s->runs[i][1] * pages + 0.999999) * PAGE);
        if (hi > len) hi = len;
        if (hi <= lo) continue;
        if (mmap(p + lo, hi - lo, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED)
            continue; /* stays MAP_NORESERVE: backed on touch all the same */
        total += hi - lo;
    }
    return total;
}

static void *serve(size_t size, struct site *s, size_t align)
{
    size_t len = ROUND(size), commit = s->model == M_RUNS ? 0 : ROUND(predict(s, size));
    if (commit > len) commit = len;
    void *p = map_block(len, commit, align);
    if (!p) return NULL;
    if (s->model == M_RUNS) commit = commit_runs(p, len, s);
    pthread_mutex_lock(&lock);
    struct block *b = slot((uintptr_t)p, 1);
    if (!b) {
        pthread_mutex_unlock(&lock);
        munmap(p, len);
        return NULL;
    }
    *b = (struct block){(uintptr_t)p, len, commit, size, s, s->model == M_RUNS};
    add_start((uintptr_t)p), nlive++;
    s->count++, s->requested += size, s->predicted += commit;
    pthread_mutex_unlock(&lock);
    atomic_fetch_add(&committed, commit);
    TRACE("serve %p len %zu commit %zu site %s\n", p, len, commit, s->key);
    return p;
}

static void *alloc_at(size_t size, uintptr_t ret, uint64_t id, int by_id, size_t align, int zero)
{
    if (ready && size >= min_size) {
        struct site *s = with_context(by_id ? site_of_id(id) : site_of_return(ret), ret);
        if (s) {
            void *p = serve(size, s, align);
            if (p) return p; /* fresh pages are zero: calloc needs no memset */
        }
    }
    if (align > 16) {
        void *p = __libc_memalign(align, size);
        if (p && zero) memset(p, 0, size);
        return p;
    }
    return zero ? __libc_calloc(1, size) : __libc_malloc(size);
}

static int ours_free(void *ptr)
{
    if (!MAYBE_OURS(ptr)) return 0;
    pthread_mutex_lock(&lock);
    struct block *b = slot((uintptr_t)ptr, 0);
    if (!b) {
        pthread_mutex_unlock(&lock);
        return 0;
    }
    struct block copy = *b;
    retire(b);
    unslot(b);
    pthread_mutex_unlock(&lock);
    TRACE("free %p len %zu\n", ptr, copy.len);
    munmap(ptr, copy.len);
    return 1;
}

/* ---- zeroing ---- */

/* Before init has found glibc's memset (other libraries' constructors may run first). A plain loop
   that GCC must not turn back into a memset call. */
__attribute__((optimize("no-tree-loop-distribute-patterns"))) static void *bytewise_memset(void *d, int c, size_t n)
{
    volatile unsigned char *p = d;
    while (n--) *p++ = (unsigned char)c;
    return d;
}

static void *real_memset(void *d, int c, size_t n)
{
    return libc_memset ? libc_memset(d, c, n) : bytewise_memset(d, c, n);
}

/* memset(d, 0, n) inside a runtime block: whole pages that are neither resident nor swapped have
   never been touched since the block was mapped (private anonymous memory), so they read as zero
   and are skipped. A page that is present but not mapped exclusively (pagemap bit 56) is the
   kernel's shared zero page, which a read of an untouched page maps (a page shared with a forked
   process looks the same): it is skipped if it reads as zero. The rest is written. Returns 0 when
   [d, d + n) is not inside a runtime block (found by address: anywhere in the block). */
static int page_needs_zeroing(const char *page, uint64_t entry)
{
    if (!(entry >> 62)) return 0;                      /* neither present (63) nor swapped (62) */
    if ((entry >> 62 & 1) || (entry >> 56 & 1)) return 1; /* swapped, or the process's own page */
    const uint64_t *w = (const uint64_t *)page;        /* shared: the zero page if it reads as zero */
    for (size_t i = 0; i < PAGE / sizeof *w; i++)
        if (w[i]) return 1;
    return 0;
}

static int zero_block(char *d, size_t n)
{
    pthread_mutex_lock(&lock);
    struct block *b = holding((uintptr_t)d, n);
    struct site *s  = b ? b->site : NULL;
    pthread_mutex_unlock(&lock);
    if (!b) return 0;
    char *lo = (char *)ROUND((uintptr_t)d), *hi = (char *)(((uintptr_t)d + n) & ~(PAGE - 1));
    if (hi <= lo) return 0;
    real_memset(d, 0, lo - d);
    uint64_t entries[512];
    size_t skipped = 0;
    for (char *p = lo; p < hi;) {
        size_t k = (size_t)(hi - p) / PAGE;
        if (k > 512) k = 512;
        off_t at = (off_t)((uintptr_t)p / PAGE) * sizeof(uint64_t);
        if (pread(pagemap, entries, k * sizeof(uint64_t), at) != (ssize_t)(k * sizeof(uint64_t))) {
            real_memset(p, 0, hi - p); /* cannot tell: write it all */
            break;
        }
        for (size_t i = 0; i < k;) { /* runs of pages to write */
            size_t j = i;
            int touched = page_needs_zeroing(p + i * PAGE, entries[i]);
            while (j < k && page_needs_zeroing(p + j * PAGE, entries[j]) == touched) j++;
            if (touched)
                real_memset(p + i * PAGE, 0, (j - i) * PAGE);
            else {
                skipped += (j - i) * PAGE;
                if (thp == 2) madvise(p + i * PAGE, (j - i) * PAGE, MADV_NOHUGEPAGE);
            }
            i = j;
        }
        p += k * PAGE;
    }
    real_memset(hi, 0, d + n - hi);
    atomic_fetch_add(&s->elided, skipped);
    TRACE("memset %p %zu: %zu bytes on untouched pages skipped\n", (void *)d, n, skipped);
    return 1;
}

#define CLEARS(c, n) ((c) == 0 && zero_skip && pagemap >= 0 && ready && (n) >= 4 * PAGE && nlive)

void *memset(void *d, int c, size_t n)
{
    if (CLEARS(c, n) && zero_block(d, n)) return d;
    return real_memset(d, c, n);
}

/* _FORTIFY_SOURCE turns memset into __memset_chk where the destination's size is known */
void *__memset_chk(void *d, int c, size_t n, size_t dlen)
{
    if (n > dlen) {
        fprintf(stderr, "libmpr: __memset_chk: buffer overflow\n");
        abort();
    }
    return memset(d, c, n);
}

void bzero(void *d, size_t n) { memset(d, 0, n); }

/* the MemPrintResize pass routes zeroing of at least 4 KB (memset/bzero calls and llvm.memset with
   a zero value) here; site: the call's id (for MPR_DEBUG) */
void __mpr_zero(void *d, size_t n, uint64_t site)
{
    if (CLEARS(0, n) && zero_block(d, n)) {
        TRACE("zero site id:0x%lx\n", (unsigned long)site);
        return;
    }
    real_memset(d, 0, n);
}

static void *realloc_at(void *ptr, size_t size, uintptr_t ret, uint64_t id, int by_id)
{
    if (!ptr) return alloc_at(size, ret, id, by_id, 0, 0);
    if (MAYBE_OURS(ptr)) {
        pthread_mutex_lock(&lock);
        struct block *b = slot((uintptr_t)ptr, 0);
        if (b && b->shaped) {
            /* a block committed in ranges is many mappings: a new block and a copy */
            struct block old = *b;
            pthread_mutex_unlock(&lock);
            void *q = size >= min_size ? serve(size, old.site, 0) : NULL;
            if (!q) q = __libc_malloc(size ? size : 1);
            if (!q) {
                errno = ENOMEM;
                return NULL;
            }
            memcpy(q, ptr, old.requested < size ? old.requested : size);
            ours_free(ptr);
            return q;
        }
        if (b) {
            /* ours: the pages move without a copy */
            /* growth: at least half as much again, so a block grown a little at a time is
               resized (and its mappings split) a logarithmic number of times */
            size_t len = ROUND(size ? size : 1);
            if (len > b->len && len < b->len + b->len / 2) len = ROUND(b->len + b->len / 2);
            /* and a shrink keeps the pages unless it halves the block (no unmap-map churn) */
            if (len < b->len && len >= b->len / 2) len = b->len;
            void *q = resize_block(ptr, b->len, b->commit, len);
            if (q == MAP_FAILED) {
                pthread_mutex_unlock(&lock);
                errno = ENOMEM;
                return NULL;
            }
            struct block moved = *b;
            size_t old_len = b->len;
            (void)old_len;
            unslot(b);
            moved.ptr = (uintptr_t)q, moved.len = len, moved.requested = size;
            if (moved.commit > len) {
                atomic_fetch_sub(&committed, moved.commit - len);
                moved.commit = len;
            }
            TRACE("realloc %p len %zu -> %p len %zu (asked %zu)\n", ptr, old_len, q, len, size);
            if (debug) {
                unsigned char v[1024];
                for (size_t off = 0; off < len; off += 1024 * PAGE)
                    if (mincore((char *)q + off, len - off < 1024 * PAGE ? len - off : 1024 * PAGE, v) != 0)
                        TRACE("HOLE in %p len %zu at +%zu (errno %d)\n", q, len, off, errno);
            }
            *slot((uintptr_t)q, 1) = moved;
            add_start((uintptr_t)q), nlive++;
            pthread_mutex_unlock(&lock);
            return q;
        }
        pthread_mutex_unlock(&lock);
    }
    if (ready && size >= min_size) {
        /* a glibc block growing past the threshold at a site with a reservation */
        struct site *s = with_context(by_id ? site_of_id(id) : site_of_return(ret), ret);
        if (s) {
            void *q = serve(size, s, 0);
            if (q) {
                size_t old = malloc_usable_size(ptr);
                memcpy(q, ptr, old < size ? old : size);
                __libc_free(ptr);
                return q;
            }
        }
    }
    return __libc_realloc(ptr, size);
}

/* ---- the program's allocator (preloaded) ---- */

#define RET ((uintptr_t)__builtin_return_address(0))

void *malloc(size_t size) { return alloc_at(size, RET, 0, 0, 0, 0); }

void *calloc(size_t n, size_t size)
{
    size_t total;
    if (__builtin_mul_overflow(n, size, &total)) {
        errno = ENOMEM;
        return NULL;
    }
    return alloc_at(total, RET, 0, 0, 0, 1);
}

void *realloc(void *ptr, size_t size) { return realloc_at(ptr, size, RET, 0, 0); }

void *reallocarray(void *ptr, size_t n, size_t size)
{
    size_t total;
    if (__builtin_mul_overflow(n, size, &total)) {
        errno = ENOMEM;
        return NULL;
    }
    return realloc_at(ptr, total, RET, 0, 0);
}

void free(void *ptr)
{
    if (!ours_free(ptr)) __libc_free(ptr);
}

void *memalign(size_t align, size_t size) { return alloc_at(size, RET, 0, 0, align, 0); }
void *aligned_alloc(size_t align, size_t size) { return alloc_at(size, RET, 0, 0, align, 0); }
void *valloc(size_t size) { return alloc_at(size, RET, 0, 0, PAGE, 0); }
void *pvalloc(size_t size) { return alloc_at(ROUND(size), RET, 0, 0, PAGE, 0); }

int posix_memalign(void **out, size_t align, size_t size)
{
    if (align < sizeof(void *) || (align & (align - 1))) return EINVAL;
    void *p = alloc_at(size, RET, 0, 0, align, 0);
    if (!p && size) return ENOMEM;
    *out = p;
    return 0;
}

size_t malloc_usable_size(void *ptr)
{
    if (!ptr) return 0;
    if (MAYBE_OURS(ptr)) {
        pthread_mutex_lock(&lock);
        struct block *b = slot((uintptr_t)ptr, 0);
        size_t len = b ? b->len : 0;
        pthread_mutex_unlock(&lock);
        if (len) return len;
    }
    static size_t (*next)(void *);
    if (!next) next = (size_t(*)(void *))dlsym(RTLD_NEXT, "malloc_usable_size");
    return next ? next(ptr) : 0;
}

/* operator new: its caller is the site (STL code is instantiated in the program). A failure
   aborts instead of throwing std::bad_alloc (C code cannot throw); the nothrow forms return NULL. */
static void *new_at(size_t size, uintptr_t ret, size_t align, int nothrow)
{
    void *p = alloc_at(size ? size : 1, ret, 0, 0, align, 0);
    if (!p && !nothrow) {
        fprintf(stderr, "libmpr: operator new(%zu) failed\n", size);
        abort();
    }
    return p;
}
void *_Znwm(size_t size) { return new_at(size, RET, 0, 0); }
void *_Znam(size_t size) { return new_at(size, RET, 0, 0); }
void *_ZnwmRKSt9nothrow_t(size_t size, const void *nt) { (void)nt; return new_at(size, RET, 0, 1); }
void *_ZnamRKSt9nothrow_t(size_t size, const void *nt) { (void)nt; return new_at(size, RET, 0, 1); }
void *_ZnwmSt11align_val_t(size_t size, size_t align) { return new_at(size, RET, align, 0); }
void *_ZnamSt11align_val_t(size_t size, size_t align) { return new_at(size, RET, align, 0); }

/* ---- the MemPrintResize pass's entry points ---- */

void *__mpr_alloc(size_t size, uint64_t site, int kind, size_t align)
{
    void *p = alloc_at(size ? size : 1, RET, site, 1, align, kind == MPR_CALLOC);
    if (!p && kind == MPR_NEW) {
        fprintf(stderr, "libmpr: operator new(%zu) failed\n", size);
        abort();
    }
    return p;
}

void *__mpr_realloc(void *ptr, size_t size, uint64_t site) { return realloc_at(ptr, size, RET, site, 1); }

void *__mpr_calloc(size_t n, size_t size, uint64_t site)
{
    size_t total;
    if (__builtin_mul_overflow(n, size, &total)) {
        errno = ENOMEM;
        return NULL;
    }
    return alloc_at(total, RET, site, 1, 0, 1);
}

int __mpr_posix_memalign(void **out, size_t align, size_t size, uint64_t site)
{
    if (align < sizeof(void *) || (align & (align - 1))) return EINVAL;
    void *p = alloc_at(size, RET, site, 1, align, 0);
    if (!p && size) return ENOMEM;
    *out = p;
    return 0;
}

/* The program's call site that is growing a container on this thread (the pass brackets calls to
   container members from the program's own code with __mpr_growth(id) ... __mpr_growth(0)). */
static __thread uint64_t growth_site;

void __mpr_growth(uint64_t site) { growth_site = site; }

/* A container's new capacity: `len` elements of `elem` bytes, at growth point `site` (a template
   instantiation). The reservation is the call site's ("cap:0x<growth site>"), else the growth
   point's ("id:0x<site>"). A cap model, "cap A B", reserves A bytes (the final size the profile saw)
   once the container asks for more than B bytes (the largest the call site's small containers ever
   grew to), so that small containers at the same call site keep their own growth. Other models
   predict the final bytes from the request. Capacity only grows: a wrong prediction cannot shrink a
   container, and an excess is backed lazily. */
size_t __mpr_capacity(size_t len, size_t size, size_t elem, uint64_t site)
{
    (void)size;
    if (!ready || !elem) return len;
    struct site *s = NULL;
    if (growth_site) {
        char key[64];
        snprintf(key, sizeof key, "cap:0x%lx", (unsigned long)growth_site);
        pthread_mutex_lock(&lock);
        s = find_key(key);
        pthread_mutex_unlock(&lock);
    }
    if (!s) s = site_of_id(site);
    if (!s || s->model == M_NONE || s->model == M_DEAD) return len;
    double bytes;
    if (s->model == M_CAP) {
        if ((double)len * elem <= s->b) return len;
        bytes = s->a * scale;
    } else
        bytes = model_value(s, len * elem);
    size_t final = (size_t)((bytes + elem - 1) / elem); /* above the request: that is the point */
    TRACE("capacity %zu x %zu -> %zu (%s)\n", len, elem, final > len ? final : len, s->key);
    return final > len ? final : len;
}

size_t __mpr_committed(void) { return atomic_load(&committed); }

/* ---- start and end ---- */

static void parse_line(const char *line)
{
    char key[128], model[32];
    double a, b;
    if (!strncmp(line, "# image ", 8)) parse_image(line);
    if (line[0] == '#' || sscanf(line, "%127s %31s %lf %lf", key, model, &a, &b) != 4) return;
    int m = !strcmp(model, "dead") ? M_DEAD : !strcmp(model, "ratio") ? M_RATIO
          : !strcmp(model, "affine") ? M_AFFINE : !strcmp(model, "const") ? M_CONST
          : !strcmp(model, "cap") ? M_CAP : !strcmp(model, "runs") ? M_RUNS : M_NONE;
    if (m == M_NONE) return;
    struct site *s = find_key(key);
    if (s) /* a base created for a context line before its own line */
        s->model = m, s->a = a, s->b = b, s->has_model = 1;
    else
        s = add_site(key, m, a, b);
    if (s && m == M_RUNS) { /* "key runs K lo1 hi1 ... loK hiK" */
        const char *p = line;
        for (int skip = 0; skip < 3 && *p; skip++) { /* past the key, the model and K */
            while (*p && *p != ' ' && *p != '\t') p++;
            while (*p == ' ' || *p == '\t') p++;
        }
        int k = (int)a < MAX_RUNS ? (int)a : MAX_RUNS;
        for (s->nruns = 0; s->nruns < k; s->nruns++) {
            char *end;
            float lo = strtof(p, &end), hi = strtof(end, &end);
            if (end == p) break;
            s->runs[s->nruns][0] = lo, s->runs[s->nruns][1] = hi, p = end;
        }
    }
    /* a context line: its base site must look at the calling context */
    char *at = strchr(key, '@');
    if (at) {
        int depth = 0;
        for (char *c = at; c; c = strchr(c + 1, '@')) depth++;
        *at = 0;
        struct site *base = find_key(key);
        if (!base && (base = add_site(key, M_NONE, 0, 0))) base->has_model = 0;
        if (base && depth > base->ctx_depth) base->ctx_depth = depth;
    }
}

/* reservations built into the program by the pass (apply mode) */
static void load_embedded(const char *text)
{
    char line[512];
    while (*text) {
        size_t n = strcspn(text, "\n");
        if (n < sizeof line) {
            memcpy(line, text, n), line[n] = 0;
            parse_line(line);
        }
        text += n + (text[n] == '\n');
    }
}

static void load(const char *file)
{
    FILE *f = fopen(file, "r");
    if (!f) {
        fprintf(stderr, "libmpr: cannot read %s\n", file);
        return;
    }
    char line[512];
    while (fgets(line, sizeof line, f)) {
        parse_line(line);
    }
    fclose(f);
}

static void *table(size_t bytes)
{
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

__attribute__((constructor)) static void init(void)
{
    const char *v;
    sites = table(MAX_SITES * sizeof *sites);
    cache = table(CACHE * sizeof *cache);
    live  = table(LIVE * sizeof *live);
    starts = table(LIVE * sizeof *starts);
    if (!sites || !cache || !live || !starts) return; /* not ready: everything goes to glibc */
    if ((v = getenv("MPR_MIN"))) min_size = strtoull(v, NULL, 10);
    if ((v = getenv("MPR_SCALE"))) scale = strtod(v, NULL);
    if ((v = getenv("MPR_ALL"))) serve_all = atoi(v);
    if ((v = getenv("MPR_DEBUG"))) debug = atoi(v);
    if ((v = getenv("MPR_ZERO"))) zero_skip = atoi(v);
    if ((v = getenv("MPR_THP"))) thp = atoi(v);
    report = getenv("MPR_REPORT");
    libc_memset = (void *(*)(void *, int, size_t))dlsym(RTLD_NEXT, "memset");
    void *prime[2];
    in_backtrace = 1, backtrace(prime, 2), in_backtrace = 0; /* load libgcc_s now, not inside an allocation */
    if (zero_skip) pagemap = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    if ((v = getenv("MPR_RESERVATIONS")))
        load(v);
    else if (__mpr_embedded_reservations)
        load_embedded(__mpr_embedded_reservations);
    ready = 1;
}

__attribute__((destructor)) static void fini(void)
{
    if (!report) return;
    pthread_mutex_lock(&lock);
    for (size_t i = 0; i < LIVE; i++) /* blocks still live: their residency now */
        if (live[i].ptr) {
            struct block b = live[i];
            struct site *s = b.site;
            size_t res = resident_bytes((void *)b.ptr, b.len);
            s->resident += res;
            if (res > b.commit) s->under++;
            double r = b.requested ? (double)res / b.requested : 0;
            if (r > s->max_resident_ratio) s->max_resident_ratio = r;
        }
    pthread_mutex_unlock(&lock); /* fclose frees its buffer through free(): no lock held from here */
    FILE *f = fopen(report, "w");
    if (f) {
        fprintf(f, "site,model,count,requested,predicted,resident,under,max_resident_ratio,elided\n");
        static const char *names[] = {"none", "dead", "ratio", "affine", "const", "cap", "runs"};
        for (int i = 0; i < nsites; i++)
            if (sites[i].count)
                fprintf(f, "%s,%s,%lu,%lu,%lu,%lu,%lu,%.4f,%lu\n", sites[i].key, names[sites[i].model],
                        (unsigned long)sites[i].count, (unsigned long)sites[i].requested,
                        (unsigned long)sites[i].predicted, (unsigned long)sites[i].resident,
                        (unsigned long)sites[i].under, sites[i].max_resident_ratio,
                        (unsigned long)atomic_load(&sites[i].elided));
        fclose(f);
    }
}

/* libmpr: allocations served according to MemPrint's per-site reservations (see mpr.h).

   Environment:
     MPR_RESERVATIONS=<file>  reservations: "site model a b" per line (analysis/memprint/sites.py)
     MPR_MIN=<bytes>          blocks smaller than this go to glibc unchanged (default 65536)
     MPR_SCALE=<x>            multiply every prediction (robustness tests)
     MPR_ALL=1                serve every large block, with a reservation or not (no reservation:
                              committed in full); measures mmap, zero pages and mremap alone
     MPR_REPORT=<file>        at exit, per site: blocks, requested, predicted (committed), resident
                              at free (mincore), blocks that used more than was predicted

   A site is the id the MemPrintResize pass passes (key "id:0x..."), or in a preloaded run the
   caller's return address as "image+0xaddress", the address in the image file: the key MemPrint's
   Pin tool writes (-sites).

   Models (bytes predicted for a request of n bytes, clamped to [0, n]): dead (0), ratio (a n),
   affine (a n + b), const (a). */
#define _GNU_SOURCE
#include "mpr.h"

#include <dlfcn.h>
#include <elf.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <sys/mman.h>
#include <unistd.h>

extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void __libc_free(void *);
extern void *__libc_memalign(size_t, size_t);

#define PAGE 4096UL
#define ROUND(n) (((n) + PAGE - 1) & ~(PAGE - 1))

enum model { M_NONE, M_DEAD, M_RATIO, M_AFFINE, M_CONST };

struct site {
    char key[128];
    int model;
    double a, b;
    /* telemetry */
    uint64_t count, requested, predicted, resident, under;
    double max_resident_ratio;
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
static struct block { uintptr_t ptr; size_t len, commit, requested; struct site *site; } *live;
static size_t nlive;

static int ready;
static size_t min_size = 65536;
static double scale = 1;
static int serve_all;
static const char *report;
static atomic_size_t committed;
static int debug; /* MPR_DEBUG=1: every runtime block's events on stderr */
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
    return s;
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

static void unslot(struct block *b)
{
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
    if (tail == p + len) return p;
    if (tail != MAP_FAILED) munmap(tail, newlen - len); /* an old kernel took the flag as a hint */
    char *q = mmap(NULL, newlen, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (q == MAP_FAILED) return MAP_FAILED;
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
static void *serve(size_t size, struct site *s, size_t align)
{
    size_t len = ROUND(size), commit = ROUND(predict(s, size));
    if (commit > len) commit = len;
    void *p = map_block(len, commit, align);
    if (!p) return NULL;
    pthread_mutex_lock(&lock);
    struct block *b = slot((uintptr_t)p, 1);
    if (!b) {
        pthread_mutex_unlock(&lock);
        munmap(p, len);
        return NULL;
    }
    *b = (struct block){(uintptr_t)p, len, commit, size, s};
    nlive++;
    s->count++, s->requested += size, s->predicted += commit;
    pthread_mutex_unlock(&lock);
    atomic_fetch_add(&committed, commit);
    TRACE("serve %p len %zu commit %zu site %s\n", p, len, commit, s->key);
    return p;
}

static void *alloc_at(size_t size, uintptr_t ret, uint64_t id, int by_id, size_t align, int zero)
{
    if (ready && size >= min_size) {
        struct site *s = by_id ? site_of_id(id) : site_of_return(ret);
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

static void *realloc_at(void *ptr, size_t size, uintptr_t ret, uint64_t id, int by_id)
{
    if (!ptr) return alloc_at(size, ret, id, by_id, 0, 0);
    if (MAYBE_OURS(ptr)) {
        pthread_mutex_lock(&lock);
        struct block *b = slot((uintptr_t)ptr, 0);
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
            nlive++;
            pthread_mutex_unlock(&lock);
            return q;
        }
        pthread_mutex_unlock(&lock);
    }
    if (ready && size >= min_size) {
        /* a glibc block growing past the threshold at a site with a reservation */
        struct site *s = by_id ? site_of_id(id) : site_of_return(ret);
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
    void *p = alloc_at(size ? size : 1, 0, site, 1, align, kind == MPR_CALLOC);
    if (!p && kind == MPR_NEW) {
        fprintf(stderr, "libmpr: operator new(%zu) failed\n", size);
        abort();
    }
    return p;
}

void *__mpr_realloc(void *ptr, size_t size, uint64_t site) { return realloc_at(ptr, size, 0, site, 1); }

void *__mpr_calloc(size_t n, size_t size, uint64_t site)
{
    size_t total;
    if (__builtin_mul_overflow(n, size, &total)) {
        errno = ENOMEM;
        return NULL;
    }
    return alloc_at(total, 0, site, 1, 0, 1);
}

int __mpr_posix_memalign(void **out, size_t align, size_t size, uint64_t site)
{
    if (align < sizeof(void *) || (align & (align - 1))) return EINVAL;
    void *p = alloc_at(size, 0, site, 1, align, 0);
    if (!p && size) return ENOMEM;
    *out = p;
    return 0;
}

size_t __mpr_capacity(size_t len, size_t size, size_t elem, uint64_t site)
{
    (void)size;
    if (!ready || !elem) return len;
    struct site *s = site_of_id(site);
    /* a growth site's model predicts the final bytes from the first request: reserve them at once */
    if (!s || s->model == M_NONE || s->model == M_DEAD) return len;
    size_t final = (size_t)(model_value(s, len * elem) / elem); /* above the request: that is the point */
    return final > len ? final : len;
}

size_t __mpr_committed(void) { return atomic_load(&committed); }

/* ---- start and end ---- */

static void parse_line(const char *line)
{
    char key[128], model[32];
    double a, b;
    if (line[0] == '#' || sscanf(line, "%127s %31s %lf %lf", key, model, &a, &b) != 4) return;
    int m = !strcmp(model, "dead") ? M_DEAD : !strcmp(model, "ratio") ? M_RATIO
          : !strcmp(model, "affine") ? M_AFFINE : !strcmp(model, "const") ? M_CONST : M_NONE;
    if (m != M_NONE) add_site(key, m, a, b);
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
    if (!sites || !cache || !live) return; /* not ready: everything goes to glibc */
    if ((v = getenv("MPR_MIN"))) min_size = strtoull(v, NULL, 10);
    if ((v = getenv("MPR_SCALE"))) scale = strtod(v, NULL);
    if ((v = getenv("MPR_ALL"))) serve_all = atoi(v);
    if ((v = getenv("MPR_DEBUG"))) debug = atoi(v);
    report = getenv("MPR_REPORT");
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
    FILE *f = fopen(report, "w");
    if (f) {
        fprintf(f, "site,model,count,requested,predicted,resident,under,max_resident_ratio\n");
        static const char *names[] = {"none", "dead", "ratio", "affine", "const"};
        for (int i = 0; i < nsites; i++)
            if (sites[i].count)
                fprintf(f, "%s,%s,%lu,%lu,%lu,%lu,%lu,%.4f\n", sites[i].key, names[sites[i].model],
                        (unsigned long)sites[i].count, (unsigned long)sites[i].requested,
                        (unsigned long)sites[i].predicted, (unsigned long)sites[i].resident,
                        (unsigned long)sites[i].under, sites[i].max_resident_ratio);
        fclose(f);
    }
    pthread_mutex_unlock(&lock);
}

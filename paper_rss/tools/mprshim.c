/* LD_PRELOAD shim for the RSS footprint model. Every MPR_INT seconds (default 0.005), from one thread:
   MPR_MODE=pss:  write "seconds,rss_bytes,pss_bytes,anon_bytes,free_bytes,lazyfree_bytes" to MPR_OUT (from
                  smaps_rollup; LazyFree: pages released with MADV_FREE, which stay resident until reclaimed);
   MPR_MODE=info: write "seconds,rss_bytes,anon_bytes,free_bytes" to MPR_OUT (from statm);
   MPR_MODE=files: as pss, from smaps, with the RSS of mapped data files (file mappings other than the executable,
                  shared objects and the C library's locale and gconv data) and their Referenced bytes (pages
                  accessed since they were mapped): "...,free_bytes,datafile_bytes,lazyfree_bytes,datafile_ref_bytes";
   MPR_MODE=trim: call malloc_trim(0) (glibc returns its free pages to the kernel);
   MPR_MODE=fast: "seconds,rss_bytes,anon_bytes,free_bytes" from /proc/self/status (VmRSS, RssAnon) every MPR_INT
                  (default here 0.0001), free_bytes every MPR_FINT seconds (default 0.001; the last value in between);
   MPR_MODE=faults: the same columns, a sample after every MPR_FAULTS page faults of the process (default 256; a
                  software event per CPU, inherited by the program's threads) and at least every MPR_INT (default
                  here 0.005).
   MPR_MODE=event: the pss columns and "trigger", from smaps_rollup (the page tables, not the status counters) at
                  these events only: after every MPR_FAULTS page faults of the process (default 256, as faults; at
                  least every MPR_INT, default here 0.005: trigger t or f); before a free of a block of at least MPR_BIG
                  bytes (default 4096) or an munmap of at least MPR_BIG bytes, in the calling thread, if the process
                  has faulted since the last sample (x, m; free is interposed only in a build with -DMPR_EVENT_FREE)
                  and, to bound the cost, if either MPR_GAP seconds (default 0.001) or ten times the last sample's
                  own duration, whichever is longer, have passed since the last sample, or the process has faulted in
                  at least MPR_GROW (default 0.01) of the last sample's anonymous bytes since it (4 KiB per fault), so
                  that a skipped sample hides at most about that share of the footprint; and once at exit (e).
   MPR_MODE=refs: hot and cold memory without Pin. Every MPR_WIN seconds (default 0.1) read /proc/self/smaps, write
                  "seconds,window,rss_bytes,anon_bytes,ref_bytes,ref_anon_bytes,free_bytes,cost_s" to MPR_OUT and
                  "window,zone,rss_bytes,ref_bytes" to MPR_OUT.zones, then clear the process's accessed bits
                  (/proc/self/clear_refs 1), so ref_bytes are the resident bytes touched in that window and
                  rss - ref the cold ones. Zones: heap (brk), anon (other anonymous mappings), stack, data (data
                  files), code (the executable, shared objects, vdso). cost_s is the sample's own time (smaps read
                  and clear). In a build with -DMPR_SITES and MPR_SITES=1, each allocation of at least MPR_BIG bytes
                  (default here 262144) gets its own mapping, and with MPR_SITE_MMAP=1 so does each anonymous private
                  mapping of at least MPR_MMAP_MIN bytes (default 65536) the program makes itself (Python's arenas,
                  GCC's GC pages; glibc's own mmaps are internal calls and stay in the anon zone). Every such block is
                  followed by a PROT_NONE guard page and starts with a MADV_DONTDUMP chunk, so the kernel never merges
                  it with a neighbour, and is attributed to its call site (the zone site<N>); MPR_CHUNK=<bytes> splits
                  each block into mappings of that size (alternate chunks MADV_DONTDUMP) and writes
                  "window,site,block,chunk,rss_bytes,ref_bytes" per mapping to MPR_OUT.chunks. The sites' stacks are
                  written at exit to MPR_OUT.sites as "site,frame,object,symbol,offset".
   MPR_MMAP=1 (with pss): interpose mmap and munmap and add "mapped_bytes", the bytes the program itself maps
   anonymously and has not unmapped (glibc's malloc maps through internal calls, which are not counted).
   Every sampling mode writes "vmhwm_bytes" (the kernel's peak RSS) at exit to <MPR_OUT>.hwm.
   MPR_NOTHP=1 also disables transparent huge pages for the process (prctl PR_SET_THP_DISABLE), so that RSS counts
   4 KB pages, as the pages-touched truth does.
   free_bytes: the bytes the allocator holds free. glibc: mallinfo's fordblks (32 bits on glibc < 2.33). jemalloc
   (mallctl found): stats.resident - stats.allocated - stats.metadata. TCMalloc (MallocExtension found):
   generic.heap_size - tcmalloc.pageheap_unmapped_bytes - generic.current_allocated_bytes. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <poll.h>
#include <malloc.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static FILE *out;
static int trim, pss, files, fast, faults, mmaps, event, refs, place;
static double win = 0.1;
static unsigned long placed;   /* bytes bound by the place mode */
static int migrate_on __attribute__((unused)), migrate_hot __attribute__((unused));
static double migrate __attribute__((unused)), alpha __attribute__((unused)) = 0.5, migrate_max __attribute__((unused)) = 2;
static unsigned long moved_out, moved_in;
static FILE *zout, *cout;
static size_t big = 4096;
static void event_sample(char trig, int only_if_faulted);
static char exe[4096];
static double interval = 0.005, finterval = 0.001;
static long fault_period = 256;
static long mapped;
static struct timespec t0;
static int (*je_mallctl)(const char *, void *, size_t *, void *, size_t);
static int (*tc_property)(const char *, size_t *);

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) * 1e-9;
}

static size_t je_stat(const char *name) {
    size_t v = 0, sz = sizeof v;
    je_mallctl(name, &v, &sz, NULL, 0);
    return v;
}

static size_t tc_stat(const char *name) {
    size_t v = 0;
    tc_property(name, &v);
    return v;
}

static unsigned long free_bytes(void) {
    long f;
    if (je_mallctl) {
        uint64_t epoch = 1;
        size_t sz = sizeof epoch;
        je_mallctl("epoch", &epoch, &sz, &epoch, sz);   /* refresh the statistics */
        f = (long)je_stat("stats.resident") - (long)je_stat("stats.allocated") - (long)je_stat("stats.metadata");
    } else if (tc_property) {
        f = (long)tc_stat("generic.heap_size") - (long)tc_stat("tcmalloc.pageheap_unmapped_bytes")
            - (long)tc_stat("generic.current_allocated_bytes");
    } else {
        struct mallinfo m = mallinfo();
        f = (unsigned)m.fordblks;
    }
    return f > 0 ? (unsigned long)f : 0;
}

static int is_data_file(const char *path) {
    if (path[0] != '/' || !strcmp(path, exe) || strstr(path, ".so") ||
        !strncmp(path, "/usr/lib/locale/", 16) || !strncmp(path, "/usr/lib64/gconv/", 17))
        return 0;
    return 1;
}

/* Rss, Pss, Anonymous, the Rss of data files, LazyFree and the Referenced bytes of data files, summed over the
   mappings of /proc/self/smaps. */
static void smaps(unsigned long v[6]) {
    char line[4352], path[4096];
    unsigned long kb;
    int data = 0;
    FILE *f = fopen("/proc/self/smaps", "r");
    while (f && fgets(line, sizeof line, f)) {
        if (sscanf(line, "Rss: %lu", &kb) == 1) { v[0] += kb << 10; if (data) v[3] += kb << 10; }
        else if (sscanf(line, "Pss: %lu", &kb) == 1) v[1] += kb << 10;
        else if (sscanf(line, "Anonymous: %lu", &kb) == 1) v[2] += kb << 10;
        else if (sscanf(line, "LazyFree: %lu", &kb) == 1) v[4] += kb << 10;
        else if (sscanf(line, "Referenced: %lu", &kb) == 1) { if (data) v[5] += kb << 10; }
        else if (strchr(line, '-') && strchr(line, '-') < strchr(line, ' ')) {   /* a mapping's first line */
            path[0] = 0;
            sscanf(line, "%*s %*s %*s %*s %*s %4095[^\n]", path);
            data = is_data_file(path);
        }
    }
    if (f) fclose(f);
}

/* VmRSS, RssAnon and VmHWM from /proc/self/status, in bytes. */
static void status(unsigned long v[3]) {
    char line[256];
    unsigned long kb;
    FILE *f = fopen("/proc/self/status", "r");
    while (f && fgets(line, sizeof line, f)) {
        if (sscanf(line, "VmRSS: %lu", &kb) == 1) v[0] = kb << 10;
        else if (sscanf(line, "RssAnon: %lu", &kb) == 1) v[1] = kb << 10;
        else if (sscanf(line, "VmHWM: %lu", &kb) == 1) v[2] = kb << 10;
    }
    if (f) fclose(f);
}

/* The program's own anonymous mappings (MPR_MMAP=1): the ranges it maps, less the parts it unmaps. */
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static int (*real_munmap)(void *, size_t);
#define MAXRANGE 65536
static struct { unsigned long lo, hi; } rng[MAXRANGE];
static int nrange;
static pthread_mutex_t rlock = PTHREAD_MUTEX_INITIALIZER;

static void *track(void *p, size_t n, int fl) {
    if (mmaps && p != MAP_FAILED && (fl & MAP_ANONYMOUS)) {
        pthread_mutex_lock(&rlock);
        if (nrange < MAXRANGE) { rng[nrange].lo = (unsigned long)p; rng[nrange++].hi = (unsigned long)p + n; mapped += n; }
        pthread_mutex_unlock(&rlock);
    }
    return p;
}

#ifdef MPR_SITES
static void *sites_mmap(void *a, size_t n, int prot, int fl, int fd, off_t off);
static void sites_unmap(unsigned long lo, unsigned long n);
#endif
void *mmap(void *a, size_t n, int prot, int fl, int fd, off_t off) {
    if (!real_mmap) real_mmap = dlsym(RTLD_NEXT, "mmap");
#ifdef MPR_SITES
    void *sp = sites_mmap(a, n, prot, fl, fd, off);
    if (sp) return track(sp, n, fl);
#endif
    return track(real_mmap(a, n, prot, fl, fd, off), n, fl);
}

void *mmap64(void *a, size_t n, int prot, int fl, int fd, off_t off) { return mmap(a, n, prot, fl, fd, off); }

int munmap(void *a, size_t n) {
    if (!real_munmap) real_munmap = dlsym(RTLD_NEXT, "munmap");
    if (event && n >= big) event_sample('m', 1);
#ifdef MPR_SITES
    sites_unmap((unsigned long)a, n);
#endif
    int r = real_munmap(a, n);
    if (mmaps && r == 0) {
        unsigned long lo = (unsigned long)a, hi = lo + n;
        pthread_mutex_lock(&rlock);
        for (int i = 0; i < nrange; i++) {   /* subtract the overlap; a hole in the middle splits the range */
            unsigned long l = rng[i].lo > lo ? rng[i].lo : lo, h = rng[i].hi < hi ? rng[i].hi : hi;
            if (l >= h) continue;
            mapped -= h - l;
            if (l > rng[i].lo && h < rng[i].hi && nrange < MAXRANGE) {
                rng[nrange].lo = h; rng[nrange++].hi = rng[i].hi; rng[i].hi = l;
            } else if (l > rng[i].lo) rng[i].hi = l;
            else rng[i].lo = h;
        }
        pthread_mutex_unlock(&rlock);
    }
    return r;
}

static unsigned long held_free;
static double free_at = -1;

/* One sample of the fast and faults modes; the free bytes are refreshed when due (always in the faults mode). */
static void quick(int refresh) {
    unsigned long v[3] = {0, 0, 0};
    double t = now();
    if (refresh || t - free_at >= finterval) { held_free = free_bytes(); free_at = t; }
    status(v);
    fprintf(out, "%.6f,%lu,%lu,%lu\n", t, v[0], v[1], held_free);
}

/* One event-mode sample from smaps_rollup; trig names the event. */
static long last_flt = -1;
static double last_t = -1, last_dur, gap = 0.001, grow = 0.01;
static unsigned long last_anon;
static __thread int busy;
static pthread_mutex_t elock = PTHREAD_MUTEX_INITIALIZER;
static long minflt(void) {
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    return u.ru_minflt + u.ru_majflt;
}

static void event_sample(char trig, int only_if_faulted) {
    if (!out || busy) return;
    busy = 1;
    long fl = minflt();
    pthread_mutex_lock(&elock);
    double t0s = now(), wait = 10 * last_dur > gap ? 10 * last_dur : gap;
    int skip = only_if_faulted && (fl == last_flt || (t0s - last_t < wait && (fl - last_flt) * 4096.0 < grow * last_anon));
    if (!skip) { last_flt = fl; last_t = t0s; }
    pthread_mutex_unlock(&elock);
    if (!skip) {
        /* open and read, not stdio: the sample allocates nothing, so the free bytes are the program's */
        static const char *key[4] = {"\nRss:", "\nPss:", "\nAnonymous:", "\nLazyFree:"};
        unsigned long v[4] = {0, 0, 0, 0};
        char buf[4096];
        double t = now();
        int fd = open("/proc/self/smaps_rollup", O_RDONLY), n = 0, r;
        while (fd >= 0 && n < (int)sizeof buf - 1 && (r = (int)read(fd, buf + n, sizeof buf - 1 - n)) > 0) n += r;
        if (fd >= 0) close(fd);
        buf[n] = 0;
        for (int i = 0; i < 4; i++) {
            char *q = strstr(buf, key[i]);
            if (q) v[i] = strtoul(q + strlen(key[i]), NULL, 10) << 10;
        }
        unsigned long fb = free_bytes();
        fprintf(out, "%.6f,%lu,%lu,%lu,%lu,%lu,%c\n", t, v[0], v[1], v[2], fb, v[3], trig);
        last_dur = now() - t0s;
        last_anon = v[2];
    }
    busy = 0;
}

#ifdef MPR_EVENT_FREE
extern void __libc_free(void *);
void free(void *p) {
    if (event && p && malloc_usable_size(p) >= big) event_sample('x', 1);
    __libc_free(p);
}
#endif

/* The refs mode. Big blocks (MPR_SITES build): the shim's own mappings, one per allocation, keyed by base address. */
#define MAXSITE 4096
#define DEPTH 8
static int nsite;
static void *site_pc[MAXSITE][DEPTH];
static int site_n[MAXSITE];
static unsigned long site_sig[MAXSITE];   /* FNV-1a of the frames' object paths and offsets: the same in every run */
#ifdef MPR_SITES
#include <execinfo.h>
#define NBLK (1 << 18)
#define HDR 64
static int sites, site_mmap;
static unsigned long chunk, mmap_min = 65536;
/* A block: an allocation of at least MPR_BIG bytes the shim mapped (kind 0, the program's pointer is base + HDR) or an
   anonymous private mapping of at least MPR_MMAP_MIN bytes the program made itself (kind 1, MPR_SITE_MMAP=1). Each is
   followed by a PROT_NONE guard page (guard 1) and starts with a MADV_DONTDUMP chunk, so the kernel never merges it
   with a neighbour; with MPR_CHUNK, every other chunk of MPR_CHUNK bytes is marked MADV_DONTDUMP, so each chunk is a
   mapping of its own in smaps. */
static struct blk { unsigned long base, len, size; int site, kind, guard; long id, ord; } blk[NBLK];   /* base 0 empty, 1 deleted */
static long nblkid;
static pthread_mutex_t block = PTHREAD_MUTEX_INITIALIZER;
static __thread int in_shim;
static long pg = 4096;
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void __libc_free(void *);
static unsigned hblk(unsigned long b) { return (unsigned)((b >> 12) * 2654435761u) & (NBLK - 1); }

static void *self_base;
static int site_of(void) {   /* the caller's stack, without the shim's own frames */
    void *pc[DEPTH + 6];
    int m = backtrace(pc, DEPTH + 6), k = 0, s;
    Dl_info d;
    while (k < m && dladdr(pc[k], &d) && d.dli_fbase == self_base) k++;
    int n = m - k < DEPTH ? m - k : DEPTH;
    pthread_mutex_lock(&block);
    for (s = 0; s < nsite; s++)
        if (site_n[s] == n && !memcmp(site_pc[s], pc + k, n * sizeof(void *))) break;
    if (s == nsite && nsite < MAXSITE) {
        memcpy(site_pc[s], pc + k, n * sizeof(void *)); site_n[s] = n;
        unsigned long h = 1469598103934665603UL;
        for (int i = 0; i < n; i++) {
            Dl_info e;
            memset(&e, 0, sizeof e);
            dladdr(pc[k + i], &e);
            unsigned long off = (unsigned long)pc[k + i] - (unsigned long)e.dli_fbase;
            for (const char *c = e.dli_fname ? e.dli_fname : "?"; *c; c++) h = (h ^ (unsigned char)*c) * 1099511628211UL;
            for (int j = 0; j < 8; j++) h = (h ^ ((off >> (8 * j)) & 255)) * 1099511628211UL;
        }
        site_sig[s] = h;
        nsite++;
    }
    pthread_mutex_unlock(&block);
    return s < MAXSITE ? s : MAXSITE - 1;
}

/* The even chunks (the whole block without MPR_CHUNK) get VM_DONTDUMP, which other mappings rarely have: the block's
   first chunk then cannot merge with the mapping below it, nor two neighbouring chunks with each other. */
static void split(unsigned long b, unsigned long len) {
    unsigned long c = chunk ? chunk : len;
    for (unsigned long o = 0; o < len; o += 2 * c)
        madvise((void *)(b + o), (o + c < len ? c : len - o), MADV_DONTDUMP);
}

/* The place mode: MPR_PLAN lists chunks as "sig,ord,chunk" (sig: the site's stack hash in hex, as in MPR_OUT.sites; ord: the block's place among its site's blocks, in the
   order they were made); each listed chunk of a new block is bound to node MPR_NODE (default 1) before it is touched. */
static unsigned long *plan;
static long nplan, site_cnt[MAXSITE];
static int place_node = 1;
static unsigned long pkey(unsigned long sig, long ord, unsigned long c) {   /* splitmix64 of the three */
    unsigned long z = sig ^ ((unsigned long)ord * 0x9E3779B97F4A7C15UL) ^ (c * 0xC2B2AE3D27D4EB4FUL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
    return z ^ (z >> 31);
}
static int in_plan(unsigned long k) {
    long l = 0, h = nplan - 1;
    while (l <= h) { long m = (l + h) / 2; if (plan[m] == k) return 1; if (plan[m] < k) l = m + 1; else h = m - 1; }
    return 0;
}
/* MPR_RPLAN: a plan that carries over to other inputs, "sig,f0,f1": every chunk of every block of the site whose
   middle lies in [f0, f1) of the block's length goes to node MPR_NODE. */
static struct rp { unsigned long sig; double f0, f1; } *rplan;
static long nrplan;
static int in_rplan(unsigned long sig, double mid) {
    long l = 0, h = nrplan - 1, i;
    while (l <= h) { long m = (l + h) / 2; if (rplan[m].sig < sig) l = m + 1; else h = m - 1; }
    for (i = l; i < nrplan && rplan[i].sig == sig; i++)
        if (mid >= rplan[i].f0 && mid < rplan[i].f1) return 1;
    return 0;
}
static void bind_plan(unsigned long b, unsigned long len, int site, long ord) {
    unsigned long c = chunk ? chunk : len, mask = 1UL << place_node;
    for (unsigned long o = 0, i = 0; o < len; o += c, i++)
        if ((nplan && in_plan(pkey(site_sig[site], ord, i))) ||
            (nrplan && in_rplan(site_sig[site], (o + (o + c < len ? c : len - o) / 2.0) / len))) {
            unsigned long n = o + c < len ? c : len - o;
            if (syscall(SYS_mbind, b + o, n, 2 /* MPOL_BIND */, &mask, 65, 0) == 0) __sync_fetch_and_add(&placed, n);
        }
}

/* ord < 0: a new block, numbered after its site's earlier ones (and placed, in the place mode); else a block re-made
   from an old one (a partial munmap or an mremap), which keeps its number. */
static void add_block(unsigned long b, unsigned long len, unsigned long size, int site, int kind, int guard, long ord) {
    int fresh = ord < 0;
    pthread_mutex_lock(&block);
    if (fresh) ord = site_cnt[site]++;
    unsigned h = hblk(b);
    while (blk[h].base > 1) h = (h + 1) & (NBLK - 1);
    blk[h].base = b; blk[h].len = len; blk[h].size = size; blk[h].site = site; blk[h].kind = kind;
    blk[h].guard = guard; blk[h].id = nblkid++; blk[h].ord = ord;
    pthread_mutex_unlock(&block);
    if ((nplan || nrplan) && fresh) bind_plan(b, len, site, ord);
}

static struct blk *find_base(unsigned long b) {   /* caller holds block */
    for (unsigned h = hblk(b), i = 0; blk[h].base && i < NBLK; h = (h + 1) & (NBLK - 1), i++)
        if (blk[h].base == b) return &blk[h];
    return NULL;
}

static void *big_alloc(size_t size) {
    if (!real_mmap) real_mmap = dlsym(RTLD_NEXT, "mmap");
    unsigned long len = (size + HDR + pg - 1) / pg * pg;
    char *b = real_mmap(NULL, len + pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (b == MAP_FAILED) return NULL;
    mprotect(b + len, pg, PROT_NONE);   /* the guard page */
    split((unsigned long)b, len);
    add_block((unsigned long)b, len, size, site_of(), 0, 1, -1);
    return b + HDR;
}

static struct blk *find(void *p) {   /* a kind-0 block from the program's pointer; caller holds block */
    if (((unsigned long)p & (pg - 1)) != HDR) return NULL;
    struct blk *k = find_base((unsigned long)p - HDR);
    return k && k->kind == 0 ? k : NULL;
}

static int big_free(void *p) {
    if (!p || ((unsigned long)p & (pg - 1)) != HDR) return 0;
    pthread_mutex_lock(&block);
    struct blk *k = find(p);
    unsigned long b = 0, len = 0;
    if (k) { b = k->base; len = k->len; k->base = 1; }
    pthread_mutex_unlock(&block);
    if (!k) return 0;
    if (!real_munmap) real_munmap = dlsym(RTLD_NEXT, "munmap");
    real_munmap((void *)b, len + pg);
    return 1;
}

static size_t big_size(void *p) {
    if (!p || ((unsigned long)p & (pg - 1)) != HDR) return 0;
    pthread_mutex_lock(&block);
    struct blk *k = find(p);
    size_t n = k ? k->size : 0;
    pthread_mutex_unlock(&block);
    return n;
}

static int take(size_t n) { return sites && !in_shim && n >= big; }

void *malloc(size_t n) {
    if (!take(n)) return __libc_malloc(n);
    in_shim = 1; void *p = big_alloc(n); in_shim = 0;
    return p ? p : __libc_malloc(n);
}

void *calloc(size_t a, size_t b) {
    size_t n = a * b;
    if ((b && n / b != a) || !take(n)) return __libc_calloc(a, b);
    in_shim = 1; void *p = big_alloc(n); in_shim = 0;   /* fresh mappings are zero */
    return p ? p : __libc_calloc(a, b);
}

void free(void *p) {
    if (!big_free(p)) __libc_free(p);
}

void *realloc(void *p, size_t n) {
    size_t old = big_size(p);
    if (!old) {
        if (!take(n) || !p) return p ? __libc_realloc(p, n) : malloc(n);
        old = malloc_usable_size(p);   /* a glibc block growing past MPR_BIG becomes a big block */
    }
    if (!n) { free(p); return NULL; }
    void *q = malloc(n);
    if (!q) return NULL;
    memcpy(q, p, old < n ? old : n);
    free(p);
    return q;
}

size_t malloc_usable_size(void *p) {
    size_t n = big_size(p);
    if (n) return n;
    static size_t (*real)(void *);
    if (!real) real = dlsym(RTLD_NEXT, "malloc_usable_size");
    return real(p);
}

/* Aligned allocations (ggml's buffers: posix_memalign with 64-byte alignment). A block's pointer is HDR bytes into a
   page, so any power-of-two alignment up to HDR is met; a larger one goes to glibc. */
extern void *__libc_memalign(size_t, size_t);
static void *big_aligned(size_t al, size_t n) {
    if (!take(n) || al > HDR || (al & (al - 1))) return NULL;
    in_shim = 1; void *p = big_alloc(n); in_shim = 0;
    return p;
}

int posix_memalign(void **out, size_t al, size_t n) {
    void *p = big_aligned(al, n);
    if (p) { *out = p; return 0; }
    static int (*real)(void **, size_t, size_t);
    if (!real) real = dlsym(RTLD_NEXT, "posix_memalign");
    return real(out, al, n);
}

void *aligned_alloc(size_t al, size_t n) {
    void *p = big_aligned(al, n);
    return p ? p : __libc_memalign(al, n);
}

void *memalign(size_t al, size_t n) {
    void *p = big_aligned(al, n);
    return p ? p : __libc_memalign(al, n);
}

/* The program's own mappings (kind 1). mmap: an anonymous private mapping of at least MPR_MMAP_MIN bytes at an
   address of the kernel's choosing gets one more page, the guard. munmap of a whole block also unmaps its guard; a
   partial munmap trims or splits the block (the guard of a trimmed tail stays mapped: one page of address space).
   mremap of a block re-registers the new range without a guard. */
static void *sites_mmap(void *a, size_t n, int prot, int fl, int fd, off_t off) {
    if (!sites || !site_mmap || in_shim || a || fd != -1 || !(fl & MAP_ANONYMOUS) || !(fl & MAP_PRIVATE) ||
        (fl & (MAP_FIXED | MAP_STACK | MAP_GROWSDOWN)) || n < mmap_min)
        return NULL;
    unsigned long len = (n + pg - 1) / pg * pg;
    in_shim = 1;
    char *b = real_mmap(NULL, len + pg, prot, fl, fd, off);
    if (b != MAP_FAILED) {
        mprotect(b + len, pg, PROT_NONE);
        split((unsigned long)b, len);
        add_block((unsigned long)b, len, n, site_of(), 1, 1, -1);
    }
    in_shim = 0;
    return b;
}

static void sites_unmap(unsigned long lo, unsigned long n) {   /* before the program's munmap of [lo, lo + n) */
    unsigned long hi = lo + (n + pg - 1) / pg * pg, gl = 0;
    pthread_mutex_lock(&block);
    struct blk *k = find_base(lo);
    if (k && k->kind == 1 && hi >= k->base + k->len) {   /* the common case: the whole block */
        if (k->guard) gl = k->base + k->len;
        k->base = 1;
    } else {   /* partial or not at a block's start: scan for overlaps */
        for (unsigned h = 0; h < NBLK; h++) {
            struct blk *q = &blk[h];
            if (q->base <= 1 || q->kind != 1) continue;
            unsigned long qlo = q->base, qhi = q->base + q->len;
            if (hi <= qlo || lo >= qhi) continue;
            struct blk c = *q;
            q->base = 1;
            pthread_mutex_unlock(&block);
            if (lo > qlo) add_block(qlo, lo - qlo, lo - qlo, c.site, 1, 0, c.ord);
            if (hi < qhi) add_block(hi, qhi - hi, qhi - hi, c.site, 1, c.guard, c.ord);
            else if (c.guard && hi == qhi) gl = qhi;
            pthread_mutex_lock(&block);
        }
    }
    pthread_mutex_unlock(&block);
    if (gl) real_munmap((void *)gl, pg);
}

void *mremap(void *old, size_t on, size_t nn, int flags, ...) {
    static void *(*real)(void *, size_t, size_t, int, ...);
    if (!real) real = dlsym(RTLD_NEXT, "mremap");
    void *na = NULL;
    if (flags & MREMAP_FIXED) { va_list ap; va_start(ap, flags); na = va_arg(ap, void *); va_end(ap); }
    struct blk c = {0};
    if (sites && site_mmap) {
        pthread_mutex_lock(&block);
        struct blk *k = find_base((unsigned long)old);
        if (k && k->kind == 1) { c = *k; k->base = 1; }
        pthread_mutex_unlock(&block);
    }
    void *r = (flags & MREMAP_FIXED) ? real(old, on, nn, flags, na) : real(old, on, nn, flags);
    if (c.len) {
        unsigned long nl = (nn + pg - 1) / pg * pg;
        if (r != MAP_FAILED) add_block((unsigned long)r, nl, nn, c.site, 1, 0, c.ord);
        else add_block(c.base, c.len, c.size, c.site, 1, c.guard, c.ord);
        if (r != MAP_FAILED && c.guard && r != old) real_munmap((void *)(c.base + c.len), pg);
    }
    return r;
}

/* Each sample takes a sorted copy of the live blocks, then finds the block a mapping starts in by binary search (a
   mapping split by the program, with mprotect or madvise, or by MPR_CHUNK, starts inside its block). */
static struct sb { unsigned long lo, hi; int site; long id, ord; } *snap;
static int nsnap;
static int cmp_sb(const void *a, const void *b) {
    unsigned long x = ((const struct sb *)a)->lo, y = ((const struct sb *)b)->lo;
    return x < y ? -1 : x > y;
}
static void snapshot(void) {
    if (!snap) { snap = real_mmap(NULL, NBLK * sizeof *snap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                 if (snap == MAP_FAILED) { snap = NULL; return; } }
    nsnap = 0;
    pthread_mutex_lock(&block);
    for (unsigned h = 0; h < NBLK; h++)
        if (blk[h].base > 1) { snap[nsnap].lo = blk[h].base; snap[nsnap].hi = blk[h].base + blk[h].len;
                               snap[nsnap].site = blk[h].site; snap[nsnap].ord = blk[h].ord; snap[nsnap++].id = blk[h].id; }
    pthread_mutex_unlock(&block);
    qsort(snap, nsnap, sizeof *snap, cmp_sb);
}
static struct sb *block_at(unsigned long a) {   /* the block that holds address a, or NULL */
    int l = 0, h = nsnap - 1;
    while (l <= h) {
        int m = (l + h) / 2;
        if (a < snap[m].lo) h = m - 1;
        else if (a >= snap[m].hi) l = m + 1;
        else return &snap[m];
    }
    return NULL;
}
#endif

/* Zones: 0 heap, 1 anon, 2 stack, 3 data, 4 code, 5.. site (zone - 5). */
#define NZ (5 + MAXSITE)
static unsigned long zrss[NZ], zref[NZ];
static long window;

static void refs_sample(void) {
    char line[4352], path[4096];
    unsigned long kb, lo, hi, v[4] = {0, 0, 0, 0};   /* rss, anon, ref, ref of anonymous memory */
    int z = 4;
    unsigned long crss = 0, cref = 0, clo = 0;
#ifdef MPR_SITES
    struct sb *cb = NULL;
    in_shim = 1;
#endif
    (void)crss; (void)cref; (void)clo;
    static pthread_mutex_t zl = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&zl);
    double t = now();
    memset(zrss, 0, sizeof zrss); memset(zref, 0, sizeof zref);
#ifdef MPR_SITES
    if (sites) snapshot();
#endif
    FILE *f = fopen("/proc/self/smaps", "r");
    while (f && fgets(line, sizeof line, f)) {
        if (sscanf(line, "Rss: %lu", &kb) == 1) { v[0] += kb << 10; zrss[z] += kb << 10; crss = kb << 10; }
        else if (sscanf(line, "Anonymous: %lu", &kb) == 1) v[1] += kb << 10;
        else if (sscanf(line, "Referenced: %lu", &kb) == 1) {
            v[2] += kb << 10; zref[z] += kb << 10; cref = kb << 10;
            if (z <= 2 || z >= 5) v[3] += kb << 10;   /* heap, anon, stack, sites */
        }
        else if (sscanf(line, "%lx-%lx ", &lo, &hi) == 2 && strchr(line, '-') < strchr(line, ' ')) {
#ifdef MPR_SITES
            if (cb && cout) fprintf(cout, "%ld,%d,%ld,%lu,%lu,%lu,%ld,%lu\n", window, cb->site, cb->id,
                                    chunk ? (clo - cb->lo) / chunk : 0, crss, cref, cb->ord, cb->hi - cb->lo);
            cb = NULL; crss = cref = 0;
#endif
            path[0] = 0;
            sscanf(line, "%*s %*s %*s %*s %*s %4095[^\n]", path);
            if (!strcmp(path, "[heap]")) z = 0;
            else if (!strncmp(path, "[stack", 6)) z = 2;
            else if (!path[0]) {
                z = 1;
#ifdef MPR_SITES
                if (sites) {
                    struct sb *k = block_at(lo);
                    if (k) { z = 5 + k->site; cb = k; clo = lo; }
                }
#endif
            }
            else z = is_data_file(path) ? 3 : 4;
        }
    }
    if (f) fclose(f);
#ifdef MPR_SITES
    if (cb && cout) fprintf(cout, "%ld,%d,%ld,%lu,%lu,%lu,%ld,%lu\n", window, cb->site, cb->id,
                            chunk ? (clo - cb->lo) / chunk : 0, crss, cref, cb->ord, cb->hi - cb->lo);
#endif
    int fd = open("/proc/self/clear_refs", O_WRONLY);
    if (fd >= 0) { if (write(fd, "1", 1) < 0) {} close(fd); }
    fprintf(out, "%.4f,%ld,%lu,%lu,%lu,%lu,%lu,%.6f\n", t, window, v[0], v[1], v[2], v[3], free_bytes(), now() - t);
    static const char *zn[5] = {"heap", "anon", "stack", "data", "code"};
    for (int i = 0; i < NZ; i++)
        if (zrss[i] || zref[i]) {
            if (i < 5) fprintf(zout, "%ld,%s,%lu,%lu\n", window, zn[i], zrss[i], zref[i]);
            else fprintf(zout, "%ld,site%d,%lu,%lu\n", window, i - 5, zrss[i], zref[i]);
        }
    window++;
    pthread_mutex_unlock(&zl);
#ifdef MPR_SITES
    in_shim = 0;
#endif
}

/* The place mode's sample: anonymous bytes on node 0 and node 1 (from numa_maps). */
static void numa_sample(void) {
    char line[4352];
    unsigned long v[2] = {0, 0};
    FILE *f = fopen("/proc/self/numa_maps", "r");
    while (f && fgets(line, sizeof line, f)) {
        if (!strstr(line, " anon=")) continue;
        unsigned long kbp = 4, n;
        char *q = strstr(line, "kernelpagesize_kB=");
        if (q) kbp = strtoul(q + 18, NULL, 10);
        for (int i = 0; i < 2; i++) {
            char key[8];
            snprintf(key, sizeof key, " N%d=", i);
            if ((q = strstr(line, key)) && (n = strtoul(q + strlen(key), NULL, 10))) v[i] += n * kbp << 10;
        }
    }
    if (f) fclose(f);
    fprintf(out, "%.4f,%lu,%lu,%lu,%lu,%lu\n", now(), v[0], v[1], placed, moved_out, moved_in);
}

#ifdef MPR_SITES
/* MPR_MIGRATE=f (place mode): every window, each resident site chunk gets a score, alpha x (its referenced share in the
   window) + (1 - alpha) x its last score (a new chunk: its share), and the lowest-scored chunks (highest with
   MPR_MIGRATE_HOT=1), up to f x the anonymous RSS and only those scored at most MPR_MIGRATE_MAX (default: any), are
   moved to node MPR_NODE, the others back to node 0 (mbind with MPOL_MF_MOVE); the accessed bits are cleared before.
   f = 0 only monitors. */
struct mc { unsigned long lo, hi, rss; long id; float s; int remote; };
static struct mc *mprev, *mcur;
static long nprev, ncur;
#define MAXMC (1 << 18)
static int cmp_mc(const void *a, const void *b) {
    unsigned long x = ((const struct mc *)a)->lo, y = ((const struct mc *)b)->lo;
    return x < y ? -1 : x > y;
}
static int cmp_score(const void *a, const void *b) {
    const struct mc *x = *(struct mc *const *)a, *y = *(struct mc *const *)b;
    float d = migrate_hot ? y->s - x->s : x->s - y->s;
    if (d) return d < 0 ? -1 : 1;
    return x->rss > y->rss ? -1 : x->rss < y->rss;
}
static void migrate_sample(void) {
    if (!mprev) {
        mprev = real_mmap(NULL, MAXMC * sizeof *mprev, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        mcur = real_mmap(NULL, MAXMC * sizeof *mcur, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mprev == MAP_FAILED || mcur == MAP_FAILED) { mprev = NULL; migrate_on = 0; return; }
    }
    static struct mc **ord;
    if (!ord) ord = real_mmap(NULL, MAXMC * sizeof *ord, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char line[4352], path[4096];
    unsigned long kb, lo, hi, anon = 0, rss = 0;
    int z = 4;
    struct sb *k = NULL;
    ncur = 0;
    snapshot();
    FILE *f = fopen("/proc/self/smaps", "r");
    while (f && fgets(line, sizeof line, f)) {
        if (sscanf(line, "Rss: %lu", &kb) == 1) {
            rss = kb << 10;
            if (z <= 2 || z >= 5) anon += rss;
        } else if (sscanf(line, "Referenced: %lu", &kb) == 1) {
            if (k && rss && ncur < MAXMC) {
                struct mc *c = &mcur[ncur++];
                c->lo = lo; c->hi = hi; c->rss = rss; c->id = k->id; c->remote = 0;
                c->s = (float)((double)(kb << 10) / rss);
            }
        } else if (sscanf(line, "%lx-%lx ", &lo, &hi) == 2 && strchr(line, '-') < strchr(line, ' ')) {
            path[0] = 0; k = NULL; rss = 0;
            sscanf(line, "%*s %*s %*s %*s %*s %4095[^\n]", path);
            if (!strcmp(path, "[heap]")) z = 0;
            else if (!strncmp(path, "[stack", 6)) z = 2;
            else if (!path[0]) { z = 1; if ((k = block_at(lo))) z = 5; }
            else z = 4;
        }
    }
    if (f) fclose(f);
    int fd = open("/proc/self/clear_refs", O_WRONLY);
    if (fd >= 0) { if (write(fd, "1", 1) < 0) {} close(fd); }
    /* carry each chunk's score and node over from the last window (mprev is sorted by lo) */
    for (long i = 0; i < ncur; i++) {
        long l = 0, h = nprev - 1;
        while (l <= h) {
            long m = (l + h) / 2;
            if (mprev[m].lo == mcur[i].lo) { l = m; break; }
            if (mprev[m].lo < mcur[i].lo) l = m + 1; else h = m - 1;
        }
        if (l < nprev && mprev[l].lo == mcur[i].lo && mprev[l].id == mcur[i].id) {
            mcur[i].s = (float)(alpha * mcur[i].s + (1 - alpha) * mprev[l].s);
            mcur[i].remote = mprev[l].remote;
        }
        ord[i] = &mcur[i];
    }
    qsort(ord, ncur, sizeof *ord, cmp_score);
    unsigned long budget = (unsigned long)(migrate * anon), sum = 0;
    for (long i = 0; i < ncur; i++) {
        struct mc *c = ord[i];
        int want = sum + c->rss <= budget && c->s <= migrate_max;   /* greedy: a chunk that does not fit is skipped */
        if (want) sum += c->rss;
        if (want != c->remote) {
            unsigned long mask = 1UL << (want ? place_node : 0);
            if (syscall(SYS_mbind, c->lo, c->hi - c->lo, 2 /* MPOL_BIND */, &mask, 65, 2 /* MPOL_MF_MOVE */) == 0) {
                c->remote = want;
                if (want) moved_out += c->rss; else moved_in += c->rss;
            }
        }
    }
    struct mc *t = mprev; mprev = mcur; mcur = t; nprev = ncur;
    qsort(mprev, nprev, sizeof *mprev, cmp_mc);   /* already in address order from smaps; kept for safety */
}
#endif

static void write_sites(void) {
    char path[4200];
    snprintf(path, sizeof path, "%s.sites", getenv("MPR_OUT"));
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "site,frame,object,symbol,offset,sig\n");
    for (int s = 0; s < nsite; s++)
        for (int i = 0; i < site_n[s]; i++) {
            Dl_info d;
            memset(&d, 0, sizeof d);
            dladdr(site_pc[s][i], &d);
            fprintf(f, "%d,%d,%s,%s,0x%lx,%016lx\n", s, i, d.dli_fname ? d.dli_fname : "?",
                    d.dli_sname ? d.dli_sname : "?", (unsigned long)site_pc[s][i] - (unsigned long)d.dli_fbase, site_sig[s]);
        }
    fclose(f);
}

/* One sample (or, in the trim mode, one trim). */
static void sample(void) {
    long page = sysconf(_SC_PAGESIZE);
    {
        if (refs) {
            refs_sample();
        } else if (place) {
#ifdef MPR_SITES
            if (migrate_on) { in_shim = 1; migrate_sample(); in_shim = 0; }
#endif
            numa_sample();
        } else if (event) {
            event_sample('e', 0);
        } else if (trim) {
            malloc_trim(0);
        } else if (fast || faults) {
            quick(1);
        } else if (files) {
            unsigned long v[6] = {0, 0, 0, 0, 0, 0};
            smaps(v);
            fprintf(out, "%.4f,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n", now(), v[0], v[1], v[2], free_bytes(), v[3], v[4], v[5]);
        } else if (pss) {
            unsigned long v[4] = {0, 0, 0, 0}, kb;
            char line[256];
            FILE *f = fopen("/proc/self/smaps_rollup", "r");
            while (f && fgets(line, sizeof line, f)) {
                if (sscanf(line, "Rss: %lu", &kb) == 1) v[0] = kb << 10;
                else if (sscanf(line, "Pss: %lu", &kb) == 1) v[1] = kb << 10;
                else if (sscanf(line, "Anonymous: %lu", &kb) == 1) v[2] = kb << 10;
                else if (sscanf(line, "LazyFree: %lu", &kb) == 1) v[3] = kb << 10;
            }
            if (f) fclose(f);
            if (mmaps)
                fprintf(out, "%.4f,%lu,%lu,%lu,%lu,%lu,%ld\n", now(), v[0], v[1], v[2], free_bytes(), v[3], mapped);
            else
                fprintf(out, "%.4f,%lu,%lu,%lu,%lu,%lu\n", now(), v[0], v[1], v[2], free_bytes(), v[3]);
        } else {
            unsigned long size, res, shared;
            FILE *f = fopen("/proc/self/statm", "r");
            if (f && fscanf(f, "%lu %lu %lu", &size, &res, &shared) == 3)
                fprintf(out, "%.4f,%lu,%lu,%lu\n", now(), res * page, (res - shared) * page, free_bytes());
            if (f) fclose(f);
        }
    }
}

static void *loop(void *arg) {
    (void)arg;
    struct timespec d = {(time_t)interval, (long)((interval - (time_t)interval) * 1e9)};
    for (;;) {
        if (fast) quick(0);
        else if (event) event_sample('t', 0);
        else sample();
        nanosleep(&d, NULL);
    }
    return NULL;
}

/* The faults mode: one page-fault event per CPU for this process and the threads it creates, each with a small ring
   buffer, so that poll() returns after every fault_period faults on any CPU. */
#define MAXCPU 256
static struct pollfd pfd[MAXCPU];
static struct perf_event_mmap_page *ring[MAXCPU];
static int ncpu;

static int open_faults(void) {
    struct perf_event_attr a;
    long page = sysconf(_SC_PAGESIZE);
    memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.type = PERF_TYPE_SOFTWARE;
    a.config = PERF_COUNT_SW_PAGE_FAULTS;
    a.sample_period = fault_period;
    a.wakeup_events = 1;
    a.inherit = 1;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    int n = (int)sysconf(_SC_NPROCESSORS_CONF);
    for (int c = 0; c < n && ncpu < MAXCPU; c++) {
        int fd = (int)syscall(SYS_perf_event_open, &a, 0, c, -1, PERF_FLAG_FD_CLOEXEC);
        if (fd < 0) continue;
        void *m = mmap(NULL, 2 * page, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED) { close(fd); continue; }
        ring[ncpu] = m;
        pfd[ncpu].fd = fd;
        pfd[ncpu].events = POLLIN;
        ncpu++;
    }
    return ncpu;
}

static void *loop_faults(void *arg) {
    (void)arg;
    int ms = (int)(interval * 1000) > 0 ? (int)(interval * 1000) : 1;
    for (;;) {
        int r = poll(pfd, ncpu, ms);
        for (int i = 0; i < ncpu; i++) {   /* consume the records, so the buffers never fill */
            __sync_synchronize();
            ring[i]->data_tail = ring[i]->data_head;
        }
        if (event) event_sample(r ? 'f' : 't', 0);
        else quick(1);
    }
    return NULL;
}

/* At exit, one last sample: a run shorter than the interval still has its final state; and the kernel's peak RSS. */
static void flush(void) {
    if (!out) return;
    sample();
    fflush(out);
    if ((zout || place) && nsite) write_sites();
    if (zout) fflush(zout);
    if (cout) fflush(cout);
    char *p = getenv("MPR_OUT"), path[4200];
    unsigned long v[3] = {0, 0, 0};
    status(v);
    snprintf(path, sizeof path, "%s.hwm", p);
    FILE *h = fopen(path, "w");
    if (h) { fprintf(h, "vmhwm_bytes\n%lu\n", v[2]); fclose(h); }
}

__attribute__((constructor)) static void start(void) {
    const char *mode = getenv("MPR_MODE"), *path = getenv("MPR_OUT"), *iv = getenv("MPR_INT");
    if (getenv("MPR_NOTHP"))
        prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
    if (!mode) return;
    unsetenv("LD_PRELOAD");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    trim = !strcmp(mode, "trim");
    pss = !strcmp(mode, "pss");
    files = !strcmp(mode, "files");
    fast = !strcmp(mode, "fast");
    faults = !strcmp(mode, "faults");
    event = !strcmp(mode, "event");
    refs = !strcmp(mode, "refs");
    place = !strcmp(mode, "place");
    if (getenv("MPR_WIN")) win = atof(getenv("MPR_WIN"));
    if (refs || place) interval = win;
#ifdef MPR_SITES
    pg = sysconf(_SC_PAGESIZE);
    if ((refs || place) && getenv("MPR_SITES") && atoi(getenv("MPR_SITES"))) {
        void *pc[4];
        Dl_info d;
        if (dladdr((void *)site_of, &d)) self_base = d.dli_fbase;
        backtrace(pc, 4);   /* loads the unwinder now, not inside the program's first big malloc */
        big = 262144;
        if (getenv("MPR_BIG")) big = (size_t)atol(getenv("MPR_BIG"));
        if (getenv("MPR_CHUNK")) chunk = (unsigned long)atol(getenv("MPR_CHUNK")) / pg * pg;
        if (getenv("MPR_MMAP_MIN")) mmap_min = (unsigned long)atol(getenv("MPR_MMAP_MIN"));
        site_mmap = getenv("MPR_SITE_MMAP") && atoi(getenv("MPR_SITE_MMAP"));
        if (!real_mmap) real_mmap = dlsym(RTLD_NEXT, "mmap");
        if (!real_munmap) real_munmap = dlsym(RTLD_NEXT, "munmap");
        sites = 1;
        const char *pp = getenv("MPR_PLAN");
        if (getenv("MPR_NODE")) place_node = atoi(getenv("MPR_NODE"));
        FILE *pf = place && pp ? fopen(pp, "r") : NULL;
        if (pf) {
            long b; unsigned long a, c, cap = 1024;
            plan = malloc(cap * sizeof *plan);
            char line[256];
            while (fgets(line, sizeof line, pf))
                if (sscanf(line, "%lx,%ld,%lu", &a, &b, &c) == 3) {
                    if (nplan == (long)cap) plan = realloc(plan, (cap *= 2) * sizeof *plan);
                    plan[nplan++] = pkey(a, b, c);
                }
            fclose(pf);
            int cmp(const void *x, const void *y) { unsigned long p = *(const unsigned long *)x, q = *(const unsigned long *)y;
                                                    return p < q ? -1 : p > q; }
            qsort(plan, nplan, sizeof *plan, cmp);
        }
        const char *rpp = getenv("MPR_RPLAN");
        FILE *rf = place && rpp ? fopen(rpp, "r") : NULL;
        if (rf) {
            unsigned long a, cap = 1024; double f0, f1;
            rplan = malloc(cap * sizeof *rplan);
            char line[256];
            while (fgets(line, sizeof line, rf))
                if (sscanf(line, "%lx,%lf,%lf", &a, &f0, &f1) == 3) {
                    if (nrplan == (long)cap) rplan = realloc(rplan, (cap *= 2) * sizeof *rplan);
                    rplan[nrplan].sig = a; rplan[nrplan].f0 = f0; rplan[nrplan++].f1 = f1;
                }
            fclose(rf);
            int rcmp(const void *x, const void *y) { unsigned long p = ((const struct rp *)x)->sig, q = ((const struct rp *)y)->sig;
                                                     return p < q ? -1 : p > q; }
            qsort(rplan, nrplan, sizeof *rplan, rcmp);
        }
        if (place && getenv("MPR_MIGRATE")) { migrate = atof(getenv("MPR_MIGRATE")); migrate_on = 1; }
        migrate_hot = getenv("MPR_MIGRATE_HOT") && atoi(getenv("MPR_MIGRATE_HOT"));
        if (getenv("MPR_ALPHA")) alpha = atof(getenv("MPR_ALPHA"));
        if (getenv("MPR_MIGRATE_MAX")) migrate_max = atof(getenv("MPR_MIGRATE_MAX"));
    }
#endif
    if (getenv("MPR_GROW")) grow = atof(getenv("MPR_GROW"));
    if (getenv("MPR_GAP")) gap = atof(getenv("MPR_GAP"));
    if (getenv("MPR_BIG") && !refs) big = (size_t)atol(getenv("MPR_BIG"));
    mmaps = pss && getenv("MPR_MMAP") && atoi(getenv("MPR_MMAP"));
    if (fast) interval = 0.0001;
    if (iv) interval = atof(iv);
    if (getenv("MPR_FINT")) finterval = atof(getenv("MPR_FINT"));
    if (getenv("MPR_FAULTS")) fault_period = atol(getenv("MPR_FAULTS"));
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    exe[n > 0 ? n : 0] = 0;
    je_mallctl = dlsym(RTLD_DEFAULT, "mallctl");
    tc_property = dlsym(RTLD_DEFAULT, "MallocExtension_GetNumericProperty");
    if (!trim) {
        if (!path || !(out = fopen(path, "w"))) return;
        setvbuf(out, NULL, _IOFBF, 1 << 20);
        if (refs) {
            char zp[4200];
            snprintf(zp, sizeof zp, "%s.zones", path);
            if (!(zout = fopen(zp, "w"))) return;
            setvbuf(zout, NULL, _IOFBF, 1 << 20);
            fprintf(zout, "window,zone,rss_bytes,ref_bytes\n");
#ifdef MPR_SITES
            if (sites) {
                snprintf(zp, sizeof zp, "%s.chunks", path);
                if ((cout = fopen(zp, "w"))) {
                    setvbuf(cout, NULL, _IOFBF, 1 << 20);
                    fprintf(cout, "window,site,block,chunk,rss_bytes,ref_bytes,ord,len\n");
                }
            }
#endif
            int fd = open("/proc/self/clear_refs", O_WRONLY);
            if (fd >= 0) { if (write(fd, "1", 1) < 0) {} close(fd); }
        }
        fprintf(out, place ? "seconds,node0_anon_bytes,node1_anon_bytes,placed_bytes,moved_out_bytes,moved_in_bytes\n"
                     : refs ? "seconds,window,rss_bytes,anon_bytes,ref_bytes,ref_anon_bytes,free_bytes,cost_s\n"
                     : event ? "seconds,rss_bytes,pss_bytes,anon_bytes,free_bytes,lazyfree_bytes,trigger\n"
                     : files ? "seconds,rss_bytes,pss_bytes,anon_bytes,free_bytes,datafile_bytes,lazyfree_bytes,"
                             "datafile_ref_bytes\n"
                     : mmaps ? "seconds,rss_bytes,pss_bytes,anon_bytes,free_bytes,lazyfree_bytes,mapped_bytes\n"
                     : pss ? "seconds,rss_bytes,pss_bytes,anon_bytes,free_bytes,lazyfree_bytes\n"
                           : "seconds,rss_bytes,anon_bytes,free_bytes\n");
        atexit(flush);
    }
    pthread_t th;
    if ((faults || event) && open_faults() > 0) {
        if (event) event_sample('t', 0);
        else quick(1);
        pthread_create(&th, NULL, loop_faults, NULL);
        pthread_detach(th);
        return;
    }
    pthread_create(&th, NULL, loop, NULL);
    pthread_detach(th);
}

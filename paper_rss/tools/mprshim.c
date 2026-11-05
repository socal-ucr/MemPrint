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
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static FILE *out;
static int trim, pss, files, fast, faults, mmaps;
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

void *mmap(void *a, size_t n, int prot, int fl, int fd, off_t off) {
    if (!real_mmap) real_mmap = dlsym(RTLD_NEXT, "mmap");
    return track(real_mmap(a, n, prot, fl, fd, off), n, fl);
}

void *mmap64(void *a, size_t n, int prot, int fl, int fd, off_t off) { return mmap(a, n, prot, fl, fd, off); }

int munmap(void *a, size_t n) {
    if (!real_munmap) real_munmap = dlsym(RTLD_NEXT, "munmap");
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

/* One sample (or, in the trim mode, one trim). */
static void sample(void) {
    long page = sysconf(_SC_PAGESIZE);
    {
        if (trim) {
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
        poll(pfd, ncpu, ms);
        for (int i = 0; i < ncpu; i++) {   /* consume the records, so the buffers never fill */
            __sync_synchronize();
            ring[i]->data_tail = ring[i]->data_head;
        }
        quick(1);
    }
    return NULL;
}

/* At exit, one last sample: a run shorter than the interval still has its final state; and the kernel's peak RSS. */
static void flush(void) {
    if (!out) return;
    sample();
    fflush(out);
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
        fprintf(out, files ? "seconds,rss_bytes,pss_bytes,anon_bytes,free_bytes,datafile_bytes,lazyfree_bytes,"
                             "datafile_ref_bytes\n"
                     : mmaps ? "seconds,rss_bytes,pss_bytes,anon_bytes,free_bytes,lazyfree_bytes,mapped_bytes\n"
                     : pss ? "seconds,rss_bytes,pss_bytes,anon_bytes,free_bytes,lazyfree_bytes\n"
                           : "seconds,rss_bytes,anon_bytes,free_bytes\n");
        atexit(flush);
    }
    pthread_t th;
    if (faults && open_faults() > 0) {
        quick(1);
        pthread_create(&th, NULL, loop_faults, NULL);
        pthread_detach(th);
        return;
    }
    pthread_create(&th, NULL, loop, NULL);
    pthread_detach(th);
}

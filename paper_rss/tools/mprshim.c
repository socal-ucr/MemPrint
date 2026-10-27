/* LD_PRELOAD shim for the RSS footprint model. Every MPR_INT seconds (default 0.005), from one thread:
   MPR_MODE=pss:  write "seconds,rss_bytes,pss_bytes,anon_bytes,free_bytes,lazyfree_bytes" to MPR_OUT (from
                  smaps_rollup; LazyFree: pages released with MADV_FREE, which stay resident until reclaimed);
   MPR_MODE=info: write "seconds,rss_bytes,anon_bytes,free_bytes" to MPR_OUT (from statm);
   MPR_MODE=files: as pss, from smaps, with the RSS of mapped data files (file mappings other than the executable,
                  shared objects and the C library's locale and gconv data) and their Referenced bytes (pages
                  accessed since they were mapped): "...,free_bytes,datafile_bytes,lazyfree_bytes,datafile_ref_bytes";
   MPR_MODE=trim: call malloc_trim(0) (glibc returns its free pages to the kernel).
   MPR_NOTHP=1 also disables transparent huge pages for the process (prctl PR_SET_THP_DISABLE), so that RSS counts
   4 KB pages, as the pages-touched truth does.
   free_bytes: the bytes the allocator holds free. glibc: mallinfo's fordblks (32 bits on glibc < 2.33). jemalloc
   (mallctl found): stats.resident - stats.allocated - stats.metadata. TCMalloc (MallocExtension found):
   generic.heap_size - tcmalloc.pageheap_unmapped_bytes - generic.current_allocated_bytes. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <malloc.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>

static FILE *out;
static int trim, pss, files;
static char exe[4096];
static double interval = 0.005;
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

static void *loop(void *arg) {
    (void)arg;
    struct timespec d = {(time_t)interval, (long)((interval - (time_t)interval) * 1e9)};
    long page = sysconf(_SC_PAGESIZE);
    for (;;) {
        if (trim) {
            malloc_trim(0);
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
            fprintf(out, "%.4f,%lu,%lu,%lu,%lu,%lu\n", now(), v[0], v[1], v[2], free_bytes(), v[3]);
        } else {
            unsigned long size, res, shared;
            FILE *f = fopen("/proc/self/statm", "r");
            if (f && fscanf(f, "%lu %lu %lu", &size, &res, &shared) == 3)
                fprintf(out, "%.4f,%lu,%lu,%lu\n", now(), res * page, (res - shared) * page, free_bytes());
            if (f) fclose(f);
        }
        nanosleep(&d, NULL);
    }
    return NULL;
}

static void flush(void) { if (out) fflush(out); }

__attribute__((constructor)) static void start(void) {
    const char *mode = getenv("MPR_MODE"), *path = getenv("MPR_OUT"), *iv = getenv("MPR_INT");
    if (getenv("MPR_NOTHP"))
        prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
    if (!mode) return;
    unsetenv("LD_PRELOAD");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (iv) interval = atof(iv);
    trim = !strcmp(mode, "trim");
    pss = !strcmp(mode, "pss");
    files = !strcmp(mode, "files");
    ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    exe[n > 0 ? n : 0] = 0;
    je_mallctl = dlsym(RTLD_DEFAULT, "mallctl");
    tc_property = dlsym(RTLD_DEFAULT, "MallocExtension_GetNumericProperty");
    if (!trim) {
        if (!path || !(out = fopen(path, "w"))) return;
        setvbuf(out, NULL, _IOFBF, 1 << 20);
        fprintf(out, files ? "seconds,rss_bytes,pss_bytes,anon_bytes,free_bytes,datafile_bytes,lazyfree_bytes,"
                             "datafile_ref_bytes\n"
                     : pss ? "seconds,rss_bytes,pss_bytes,anon_bytes,free_bytes,lazyfree_bytes\n"
                           : "seconds,rss_bytes,anon_bytes,free_bytes\n");
        atexit(flush);
    }
    pthread_t th;
    pthread_create(&th, NULL, loop, NULL);
    pthread_detach(th);
}

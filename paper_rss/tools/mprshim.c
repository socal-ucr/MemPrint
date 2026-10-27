/* LD_PRELOAD shim for the RSS-information study. MPR_MODE=info: every MPR_INT seconds write
   "seconds,rss_bytes,anon_bytes,free_bytes" to MPR_OUT (MPR_MODE=pss: from smaps_rollup, with pss_bytes) (free_bytes: glibc's free bytes in its arenas, mallinfo
   fordblks). MPR_MODE=trim: every MPR_INT seconds call malloc_trim(0) (returns free pages to the kernel). */
#define _GNU_SOURCE
#include <malloc.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static FILE *out;
static int trim, pss;
static double interval = 0.005;
static struct timespec t0;

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) * 1e-9;
}

static void *loop(void *arg) {
    (void)arg;
    struct timespec d = {0, (long)(interval * 1e9)};
    long page = sysconf(_SC_PAGESIZE);
    for (;;) {
        if (trim) {
            malloc_trim(0);
        } else if (pss) {
            unsigned long v[3] = {0, 0, 0}, kb;
            char line[256];
            FILE *f = fopen("/proc/self/smaps_rollup", "r");
            while (f && fgets(line, sizeof line, f)) {
                if (sscanf(line, "Rss: %lu", &kb) == 1) v[0] = kb << 10;
                else if (sscanf(line, "Pss: %lu", &kb) == 1) v[1] = kb << 10;
                else if (sscanf(line, "Anonymous: %lu", &kb) == 1) v[2] = kb << 10;
            }
            if (f) fclose(f);
            struct mallinfo m = mallinfo();
            fprintf(out, "%.4f,%lu,%lu,%lu,%u\n", now(), v[0], v[1], v[2], (unsigned)m.fordblks);
        } else {
            unsigned long size, res, shared;
            FILE *f = fopen("/proc/self/statm", "r");
            if (f && fscanf(f, "%lu %lu %lu", &size, &res, &shared) == 3) {
                struct mallinfo m = mallinfo();
                fprintf(out, "%.4f,%lu,%lu,%u\n", now(), res * page, (res - shared) * page, (unsigned)m.fordblks);
            }
            if (f) fclose(f);
        }
        nanosleep(&d, NULL);
    }
    return NULL;
}

static void flush(void) { if (out) fflush(out); }

__attribute__((constructor)) static void start(void) {
    const char *mode = getenv("MPR_MODE"), *path = getenv("MPR_OUT"), *iv = getenv("MPR_INT");
    if (!mode) return;
    unsetenv("LD_PRELOAD");
    clock_gettime(CLOCK_MONOTONIC, &t0);
    if (iv) interval = atof(iv);
    trim = !strcmp(mode, "trim");
    pss = !strcmp(mode, "pss");
    if (!trim) {
        if (!path || !(out = fopen(path, "w"))) return;
        setvbuf(out, NULL, _IOFBF, 1 << 20);
        fprintf(out, pss ? "seconds,rss_bytes,pss_bytes,anon_bytes,free_bytes\n" : "seconds,rss_bytes,anon_bytes,free_bytes\n");
        atexit(flush);
    }
    pthread_t th;
    pthread_create(&th, NULL, loop, NULL);
    pthread_detach(th);
}

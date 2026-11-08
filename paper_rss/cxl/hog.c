// hog <threads> [stride] [seconds]: each thread reads its own 256 MB buffer in a loop (run under numactl on node 1).
// stride in 8-byte words (default 8: one read per 64-byte line). With seconds, it stops after that long and prints the
// GB/s its threads read (lines read x 64 bytes, after a 1 s warm-up).
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
static long st = 8;
static volatile uint64_t lines[64 * 8];   // per thread, a cache line apart
static void *run(void *x) {
    long id = (long)x; size_t n = (256UL << 20) / 8; volatile uint64_t *a = malloc(n * 8); memset((void *)a, 1, n * 8);
    uint64_t s = 0;
    for (;;) { for (size_t i = 0; i < n; i += st) s += a[i]; lines[id * 8] += n / st; }
    return (void *)s;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t total(int t) { uint64_t s = 0; for (int i = 0; i < t; i++) s += lines[i * 8]; return s; }
int main(int c, char **v) {
    int t = atoi(v[1]); if (c > 2) st = atol(v[2]); double secs = c > 3 ? atof(v[3]) : 0; pthread_t p[64];
    for (long i = 0; i < t; i++) pthread_create(&p[i], 0, run, (void *)i);
    if (secs <= 0) { pthread_join(p[0], 0); return 0; }
    sleep(1); uint64_t l0 = total(t); double t0 = now();
    usleep((useconds_t)((secs - 1) * 1e6));
    printf("%.2f\n", (total(t) - l0) * 64.0 / (now() - t0) / 1e9);
    return 0;
}

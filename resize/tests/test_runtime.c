/* libmpr runtime tests: run with LD_PRELOAD=libmpr.so MPR_RESERVATIONS=reservations.txt MPR_ALL=1
   (resize/tests/run_tests.sh writes the file: id:0x1 dead, id:0x2 ratio 0.25, id:0x3 const 8192). */
#include <fcntl.h>
#include <malloc.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif
#define PAGE_SIZE_TEST 4096

#include "../runtime/mpr.h"

#define KB 1024UL
#define MB (1024 * KB)
static int failed;

#define CHECK(cond, ...)                                                                     \
    do {                                                                                     \
        if (cond) printf("    ok   " __VA_ARGS__);                                           \
        else { printf("    FAIL " __VA_ARGS__); failed = 1; }                                \
        printf("\n");                                                                        \
    } while (0)

static void fill(unsigned char *p, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; i += 512) p[i] = (unsigned char)(i / 512 + seed);
}
static int same(const unsigned char *p, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; i += 512)
        if (p[i] != (unsigned char)(i / 512 + seed)) return 0;
    return 1;
}

static void *worker(void *arg)
{
    unsigned seed = (unsigned)(uintptr_t)arg;
    for (int i = 0; i < 200; i++) {
        size_t n = (64 + (i * 37 + seed) % 512) * KB;
        unsigned char *p = malloc(n);
        fill(p, n, seed + i);
        p = realloc(p, n * 2);
        if (!same(p, n, seed + i)) return (void *)1;
        free(p);
        char *small = malloc(100);
        free(small);
    }
    return NULL;
}

int main(void)
{
    size_t base = __mpr_committed();

    /* a dead site: nothing committed, yet the program may use all of it, also through a system call */
    unsigned char *dead = __mpr_alloc(1 * MB, 0x1, MPR_MALLOC, 0);
    CHECK(dead && __mpr_committed() - base == 0, "dead site: 1 MB served, 0 committed");
    int fd = open("/dev/zero", O_RDONLY);
    ssize_t got = read(fd, dead, 1 * MB);
    close(fd);
    fill(dead, 1 * MB, 7);
    CHECK(got == (ssize_t)MB && same(dead, 1 * MB, 7), "under-prediction: all of it written (read() into it too)");

    /* ratio 0.25: a quarter committed */
    unsigned char *quarter = __mpr_alloc(1 * MB, 0x2, MPR_MALLOC, 0);
    CHECK(__mpr_committed() - base == 256 * KB, "ratio site: 256 KB of 1 MB committed");

    /* calloc: fresh zero pages */
    unsigned char *z = __mpr_alloc(512 * KB, 0x2, MPR_CALLOC, 0);
    int zero = 1;
    for (size_t i = 0; i < 512 * KB; i++) zero &= z[i] == 0;
    CHECK(zero, "calloc: zero without a memset");

    /* realloc by mremap keeps the data, growing and shrinking */
    fill(quarter, 1 * MB, 3);
    quarter = __mpr_realloc(quarter, 4 * MB, 0x2);
    CHECK(quarter && same(quarter, 1 * MB, 3), "realloc grow (mremap): data kept");
    quarter = __mpr_realloc(quarter, 512 * KB, 0x2);
    CHECK(quarter && same(quarter, 512 * KB, 3), "realloc shrink: data kept");
    CHECK(malloc_usable_size(quarter) >= 512 * KB, "malloc_usable_size of a runtime block");

    /* growth when the address after the block is taken: the mappings move, nothing is copied */
    void *wall = mmap((char *)quarter + 512 * KB, PAGE_SIZE_TEST, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    unsigned char *before = quarter;
    quarter = __mpr_realloc(quarter, 2 * MB, 0x2);
    CHECK(wall == (char *)before + 512 * KB && quarter && quarter != before && same(quarter, 512 * KB, 3),
          "realloc grow with the next page taken: moved, data kept");
    munmap(wall, PAGE_SIZE_TEST);

    /* alignment above a page */
    void *aligned = __mpr_alloc(1 * MB, 0x3, MPR_MEMALIGN, 64 * KB);
    CHECK(((uintptr_t)aligned & (64 * KB - 1)) == 0, "64 KB alignment");

    /* small blocks go to glibc; free sorts them out */
    char *small = __mpr_alloc(100, 0x2, MPR_MALLOC, 0);
    strcpy(small, "glibc");
    free(small);

    free(dead), free(quarter), free(z), free(aligned);
    CHECK(__mpr_committed() == base, "committed back to %zu after the frees", base);

    /* preloaded allocator (MPR_ALL: every large block served), 8 threads */
    pthread_t t[8];
    for (uintptr_t i = 0; i < 8; i++) pthread_create(&t[i], NULL, worker, (void *)i);
    void *bad = NULL;
    for (int i = 0; i < 8; i++) {
        void *r;
        pthread_join(t[i], &r);
        if (r) bad = r;
    }
    CHECK(!bad, "8 threads: malloc, realloc, free of large and small blocks");

    printf(failed ? "some runtime tests FAILED\n" : "all runtime tests passed\n");
    return failed;
}

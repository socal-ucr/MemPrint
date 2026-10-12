/* -sites: one allocation site per function, each with a known requested and touched size.
   check_sites.py names a site by the function its return address is in (addr2line). */
#include <stdlib.h>
#include <string.h>
#include "common.h"

/* 1 MB requested, never touched */
__attribute__((noinline)) static char *dead(void) { return malloc(1 * MB); }

/* 1 MB requested, the first 256 KB touched: high water 256 KB */
__attribute__((noinline)) static char *prefix(void)
{
    char *p = malloc(1 * MB);
    touch(p, 256 * KB);
    return p;
}

/* 512 KB requested, only the last 64 KB touched: high water 512 KB, touched 64 KB */
__attribute__((noinline)) static char *tail(void)
{
    char *p = malloc(512 * KB);
    touch(p + 448 * KB, 64 * KB);
    return p;
}

/* calloc of 64 KB served from a freed, touched heap chunk: calloc zeroes it (64 KB touched by
   the allocator), the program then touches 4 KB */
__attribute__((noinline)) static char *zeroed(void)
{
    char *p = calloc(1, 64 * KB);
    touch(p, 4 * KB);
    return p;
}

/* realloc growth: 1 KB doubling to 1 MB, each block fully touched; 10 blocks, 9 moves */
__attribute__((noinline)) static char *grow(void)
{
    char *p = NULL;
    for (size_t n = 1 * KB; n <= 1 * MB; n *= 2)
    {
        p = realloc(p, n);
        touch(p, n);
    }
    return p;
}

/* four 16 KB blocks, each half touched */
__attribute__((noinline)) static char *half(void)
{
    char *p = malloc(16 * KB);
    touch(p, 8 * KB);
    return p;
}

int main(void)
{
    /* a 64 KB heap chunk, touched and freed, for zeroed() to reuse */
    char *warm = malloc(64 * KB);
    touch(warm, 64 * KB);
    char *keep = malloc(16);  /* keeps warm's chunk from merging into the top */
    free(warm);

    char *a = dead(), *b = prefix(), *c = tail(), *d = zeroed(), *e = grow();
    char *h[4];
    for (int i = 0; i < 4; i++) h[i] = half();
    free(a), free(b), free(c), free(d), free(e), free(keep);
    for (int i = 0; i < 4; i++) free(h[i]);
    return 0;
}

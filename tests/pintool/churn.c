/* Heap churn: a ring of live heap blocks, each replaced in turn, of which only
 * a fraction is ever touched. Freed blocks' pages stay resident and are handed
 * to later blocks, so page residency overstates what the later blocks touch.
 *
 *   churn FRACTION [STEPS]
 *
 * Keeps 64 live blocks of 16 to 112 KB (below glibc's mmap threshold, so they
 * come from the heap). Each step frees the oldest block, allocates a new one,
 * writes the first FRACTION of it, and reads the touched part of one other
 * live block. Live footprint: about FRACTION x 4 MB.
 */
#include <stdio.h>
#include <stdlib.h>

#define LIVE 64

int main(int argc, char** argv)
{
    double fraction = argc > 1 ? atof(argv[1]) : 0.25;
    long steps      = argc > 2 ? atol(argv[2]) : 4000;
    char* block[LIVE] = {0};
    size_t touched[LIVE] = {0};
    unsigned seed = 12345;
    volatile long sink = 0;
    for (long s = 0; s < steps; s++)
    {
        int k = s % LIVE;
        free(block[k]);
        seed = seed * 1103515245 + 12345;
        size_t size = (16 + (seed >> 8) % 97) * 1024;
        block[k]    = malloc(size);
        touched[k]  = (size_t)(fraction * size) & ~(size_t)7;
        for (size_t i = 0; i < touched[k]; i += 8)
            *(long*)(block[k] + i) = s;
        int j = (seed >> 4) % LIVE;
        for (size_t i = 0; block[j] && i < touched[j]; i += 8)
            sink += *(long*)(block[j] + i);
    }
    for (int k = 0; k < LIVE; k++)
        free(block[k]);
    printf("%ld\n", sink != 0);
    return 0;
}

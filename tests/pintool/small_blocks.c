/* Heap blocks below the mmap threshold, as PolyBench allocates its arrays:
   16 rounds of posix_memalign/malloc 64 KB + 64 KB, touch, free.
   Peak 128 KB (+ runtime); >= 2 MB freed. */
#include <stdlib.h>
#include "common.h"

int main(void)
{
    for (int round = 0; round < 16; round++)
    {
        void *a, *b = malloc(64 * KB);
        posix_memalign(&a, 32, 64 * KB);
        touch(a, 64 * KB);
        touch(b, 64 * KB);
        free(a);
        free(b);
    }
    return 0;
}

/* The program's first allocation is a large calloc. glibc then calls malloc
 * through its initialization hook and leaves calloc with a jump to memset, so
 * calloc's exit is never seen; the tool must still record the block (4 MB),
 * and must not treat later allocations as nested in it. */
#include <stdlib.h>

#include "common.h"

int main(void)
{
    char* a = calloc(4 << 20, 1); /* zeroed by calloc: that is its touch */
    volatile char sink = a[(4 << 20) - 1];
    (void)sink;
    free(a);
    char* b = malloc(2 << 20);
    touch(b, 2 << 20);
    free(b);
    return 0;
}

/* malloc 4 MB and write 8 bytes on each page of its first 2 MB, many times over, then free.
   Bytes touched: 4 KB; pages touched: 512, so the footprint with -footprint pages peaks at 2 MB. */
#include <stdlib.h>
#include "common.h"

int main(void)
{
    volatile double *a = malloc(4 * MB);
    for (int pass = 0; pass < 2000; pass++)
        for (size_t page = 0; page < 2 * MB / (4 * KB); page++) a[page * (4 * KB / sizeof(double))] = pass;
    free((void *)a);
    return 0;
}

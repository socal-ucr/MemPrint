#include <stddef.h>

#define KB (1024UL)
#define MB (1024UL * KB)

/* Write every 8-byte word of [p, p + bytes): the footprint grows by exactly `bytes`. */
static void touch(void *p, size_t bytes)
{
    volatile double *d = (volatile double *)p;
    for (size_t i = 0; i < bytes / sizeof(double); i++) d[i] = (double)i;
}

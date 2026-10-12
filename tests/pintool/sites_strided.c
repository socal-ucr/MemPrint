/* A block used like a KV cache: 8 layers of 1 MB, of which only the first 64 KB (the positions
   generated so far) are written in each layer. Its highest touched byte is near the end, so a
   prefix reservation commits almost all of it; a shaped one commits the 8 ranges, 512 KB. */
#include <stdlib.h>
#include "common.h"

__attribute__((noinline)) char *cache(void)
{
    char *p = malloc(8 * MB);
    for (int layer = 0; layer < 8; layer++) touch(p + layer * MB, 64 * KB);
    return p;
}

int main(void)
{
    char *c = cache();
    long sum = 0;
    for (int layer = 0; layer < 8; layer++) sum += c[layer * MB + 8];
    free(c);
    return sum == 0 ? 0 : 1;
}

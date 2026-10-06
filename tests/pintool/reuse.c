/* Footprint 4 MB, every 8-byte word written 10 times (10 sweeps). With
   uniform reuse, Chao1 on the sampled union recovers the true number of
   addresses although only a small fraction of them are ever sampled. */
#include <stdlib.h>
#include "common.h"

int main(void)
{
    char *a = malloc(4 * MB);
    for (int sweep = 0; sweep < 10; sweep++) touch(a, 4 * MB);
    free(a);
    return 0;
}

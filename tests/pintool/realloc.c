/* 64 KB heap block shrunk in place to 32 KB (32 KB tail freed), then grown to
   4 MB (moves; old 32 KB freed), touched, and freed. Peak 4 MB; >= 4 MB + 64 KB freed. */
#include <stdlib.h>
#include "common.h"

int main(void)
{
    char *a = malloc(64 * KB);
    touch(a, 64 * KB);
    a = realloc(a, 32 * KB);
    a = realloc(a, 4 * MB);
    touch(a, 4 * MB);
    free(a);
    return 0;
}

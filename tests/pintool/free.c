/* malloc 1 MB, touch, free; then malloc 2 MB, touch, free.
   Live footprint peaks at 2 MB; 3 MB is freed in total. */
#include <stdlib.h>
#include "common.h"

int main(void)
{
    char *a = malloc(1 * MB);
    touch(a, 1 * MB);
    free(a);
    char *b = malloc(2 * MB);
    touch(b, 2 * MB);
    free(b);
    return 0;
}

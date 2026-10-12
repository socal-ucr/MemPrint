/* Two callers of a wrapper in a shared library (libwrap.so): hot() writes all of its 1 MB block,
   cold() none of it. The site is in the library, the calling contexts in the program. */
#include <stdlib.h>
#include "common.h"

void *lib_alloc(size_t n);

__attribute__((noinline)) char *hot(void)
{
    char *p = lib_alloc(1 * MB);
    touch(p, 1 * MB);
    return p;
}

__attribute__((noinline)) char *cold(void)
{
    char *p = lib_alloc(1 * MB);
    __asm__ volatile("" ::: "memory");
    return p;
}

int main(void)
{
    char *a = hot(), *b = cold();
    free(a), free(b);
    return 0;
}

/* An allocation wrapper in a shared library (the program's build never sees its code): sites_lib.c
   calls it from two places, one that writes its block and one that does not. */
#include <stdlib.h>

__attribute__((noinline)) void *lib_alloc(size_t n)
{
    void *p = malloc(n);
    __asm__ volatile("" ::: "memory"); /* not a tail call */
    return p;
}

/* -sites calling contexts and zeroing.
   - my_alloc (not inlined) wraps malloc, so every block it returns has one site, the call in
     my_alloc. hot() writes all 1 MB of its block, cold() none: the caller frame tells them apart.
   - cleared(): malloc 1 MB, clear it with memset, then write its first 64 KB. The memset writes
     zeros over untouched memory: with -skip_zeroing the block's touched bytes are 64 KB. */
#include <stdlib.h>
#include <string.h>
#include "common.h"

__attribute__((noinline)) void *my_alloc(size_t n)
{
    void *p = malloc(n);
    __asm__ volatile("" ::: "memory"); /* not a tail call: my_alloc stays in the stack */
    return p;
}

__attribute__((noinline)) char *hot(void)
{
    char *p = my_alloc(1 * MB);
    touch(p, 1 * MB);
    return p;
}

__attribute__((noinline)) char *cold(void)
{
    char *p = my_alloc(1 * MB);
    __asm__ volatile("" ::: "memory");
    return p;
}

__attribute__((noinline)) char *cleared(void)
{
    char *p = malloc(1 * MB);
    void *(*volatile set)(void *, int, size_t) = memset; /* a real call, as from another library */
    set(p, 0, 1 * MB);
    touch(p, 64 * KB);
    return p;
}

int main(void)
{
    char *a = hot(), *b = cold(), *c = cleared();
    free(a), free(b), free(c);
    return 0;
}

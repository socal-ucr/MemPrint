/* Anonymous 4 MB mapping, touched; unmap the second MB, then the rest.
   Peak 4 MB; >= 4 MB freed. */
#include <sys/mman.h>
#include "common.h"

int main(void)
{
    char *a = mmap(0, 4 * MB, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    touch(a, 4 * MB);
    munmap(a + 1 * MB, 1 * MB);
    munmap(a, 1 * MB);
    munmap(a + 2 * MB, 2 * MB);
    return 0;
}

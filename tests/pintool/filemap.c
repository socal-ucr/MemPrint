/* -file_maps (windowed): a 4 MB file mapped read-only and read in full at the start, as an LLM engine reads
   its weights, then a long phase over a 64 KB array, then the mapping is removed. The file is read between
   windows: without -file_maps the windowed estimate sees it only through the windows' samples. */
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "common.h"

int main(int argc, char **argv)
{
    const size_t n = 4 * MB;
    static char buf[64 * KB];
    int fd = open(argc > 1 ? argv[1] : "filemap.bin", O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return 1;
    memset(buf, 1, sizeof buf);
    for (size_t k = 0; k < n / sizeof buf; k++)
        if (write(fd, buf, sizeof buf) != (ssize_t)sizeof buf) return 1;
    const long *p = mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return 1;
    volatile long s = 0;
    for (size_t i = 0; i < n / sizeof(long); i++) s += p[i];          /* every byte of the file, once */
    static volatile long a[8 * KB];
    for (int r = 0; r < 1000; r++)
        for (size_t i = 0; i < 8 * KB; i++) a[i] += (long)i;         /* 16 M references in 64 KB */
    munmap((void *)p, n);
    for (int r = 0; r < 100; r++)
        for (size_t i = 0; i < 8 * KB; i++) a[i] += (long)i;         /* after the unmap */
    return s == 0;
}

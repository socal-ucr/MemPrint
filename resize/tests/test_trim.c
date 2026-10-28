/* MPR_TRIM: 200 MB of 1 KB blocks, every one but each 100th freed (the heap cannot shrink from its top), then the
   anonymous memory after 100 ms. Prints "anon_mb_before anon_mb_after". */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static long anon_kb(void)
{
    char line[256];
    long kb = -1;
    FILE *f = fopen("/proc/self/status", "r");
    while (f && fgets(line, sizeof line, f))
        if (sscanf(line, "RssAnon: %ld", &kb) == 1) break;
    if (f) fclose(f);
    return kb;
}

int main(void)
{
    enum { N = 200000, SIZE = 1024 };
    static char *p[N];
    for (int i = 0; i < N; i++) memset(p[i] = malloc(SIZE), 1, SIZE);
    for (int i = 0; i < N; i++)
        if (i % 100) free(p[i]);
    long before = anon_kb();
    struct timespec d = {0, 100000000};
    nanosleep(&d, NULL);
    printf("%ld %ld\n", before / 1024, anon_kb() / 1024);
    return 0;
}

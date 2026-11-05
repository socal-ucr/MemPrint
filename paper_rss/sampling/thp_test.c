/* Which 4 KiB pages of a huge page were touched: a test of what the kernel reports without root (PROTOCOL.md).
   Maps 64 MiB aligned to 2 MiB, touches one 4 KiB page in each 2 MiB, and prints AnonHugePages and Rss of the range
   (/proc/self/smaps), the process's RssAnon, and how many of the range's 4 KiB pages mincore() reports resident:
     enabled    THP as the machine sets it
     nohuge     MADV_NOHUGEPAGE on the range before the touches
     disabled   prctl(PR_SET_THP_DISABLE) for the process
   /proc/self/pagemap shows no page frame numbers without CAP_SYS_ADMIN, and /sys/kernel/mm/page_idle needs root. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>

#define SZ (64ul << 20)
#define HUGE (2ul << 20)

static unsigned long field(const char *file, unsigned long lo, const char *key) {
    char line[512], k[64];
    unsigned long a, b, kb, v = 0;
    int in = !lo;
    FILE *f = fopen(file, "r");
    while (fgets(line, sizeof line, f)) {
        if (lo && sscanf(line, "%lx-%lx", &a, &b) == 2 && strchr(line, '-') < strchr(line, ' ')) in = a == lo;
        else if (in && sscanf(line, "%63[^:]: %lu", k, &kb) == 2 && !strcmp(k, key)) v = kb;
    }
    fclose(f);
    return v;
}

static void test(const char *name, int nohuge) {
    char *raw = mmap(NULL, SZ + HUGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    char *p = (char *)(((unsigned long)raw + HUGE - 1) & ~(HUGE - 1));
    if (nohuge) madvise(p, SZ, MADV_NOHUGEPAGE);
    unsigned long before = field("/proc/self/status", 0, "RssAnon");
    for (unsigned long o = 0; o < SZ; o += HUGE) p[o] = 1;
    unsigned char vec[SZ / 4096];
    mincore(p, SZ, vec);
    unsigned long res = 0;
    for (unsigned long i = 0; i < SZ / 4096; i++) res += vec[i] & 1;
    /* the range may be split from raw's mapping by madvise: look up the mapping that starts at p or at raw */
    unsigned long huge = field("/proc/self/smaps", (unsigned long)p, "AnonHugePages") +
                         field("/proc/self/smaps", (unsigned long)raw, "AnonHugePages");
    unsigned long rss = field("/proc/self/smaps", (unsigned long)p, "Rss") +
                        field("/proc/self/smaps", (unsigned long)raw, "Rss");
    printf("%s: touched %lu KiB, AnonHugePages %lu KiB, Rss %lu KiB, RssAnon +%lu KiB, mincore resident %lu KiB\n",
           name, SZ / HUGE * 4, huge, rss, field("/proc/self/status", 0, "RssAnon") - before, res * 4);
    munmap(raw, SZ + HUGE);
}

int main(void) {
    test("enabled", 0);
    test("nohuge", 1);
    prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0);
    test("disabled", 0);
    return 0;
}

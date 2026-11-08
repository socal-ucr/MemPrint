#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
int main(){ size_t n = (512UL<<20)/64; uint64_t *a = aligned_alloc(64, n*64), *perm = malloc(n*8);
 for (size_t i=0;i<n;i++) perm[i]=i; srand(1); for (size_t i=n-1;i>0;i--){ size_t j=((size_t)rand()<<16^rand())%(i+1); uint64_t t=perm[i]; perm[i]=perm[j]; perm[j]=t; }
 for (size_t i=0;i<n;i++) a[perm[i]*8]=perm[(i+1)%n]*8;
 struct timespec t0,t1; uint64_t p=0; size_t k=20000000; clock_gettime(CLOCK_MONOTONIC,&t0);
 for (size_t i=0;i<k;i++) p=a[p]; clock_gettime(CLOCK_MONOTONIC,&t1);
 printf("%.1f ns/load (%lu)\n", ((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/k, p&1); }

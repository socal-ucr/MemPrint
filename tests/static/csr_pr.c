/* PageRank on a random CSR graph, as a C program the static interpreter must run
 * by itself: edges from rand(), degree counting, a prefix sum, a counting-sort
 * fill and data-bounded neighbour loops. Build: cc -O0 -DSCALE=<s> csr_pr.c */
#include <stdio.h>
#include <stdlib.h>

#ifndef SCALE
#define SCALE 12
#endif
#define DEGREE 16
#define ITERATIONS 10

int main(void) {
  int n = 1 << SCALE;
  long m = (long)n * DEGREE;
  int *src = malloc(m * sizeof(int));
  int *dst = malloc(m * sizeof(int));
  srand(27491095);
  for (long e = 0; e < m; e++) {
    src[e] = rand() % n;
    dst[e] = rand() % n;
  }
  int *deg = calloc(n, sizeof(int));
  for (long e = 0; e < m; e++)
    deg[src[e]]++;
  long *off = malloc((n + 1) * sizeof(long));
  off[0] = 0;
  for (int u = 0; u < n; u++)
    off[u + 1] = off[u] + deg[u];
  long *pos = malloc(n * sizeof(long));
  for (int u = 0; u < n; u++)
    pos[u] = off[u];
  int *nbr = malloc(m * sizeof(int));
  for (long e = 0; e < m; e++)
    nbr[pos[src[e]]++] = dst[e];
  free(pos);
  free(dst);
  free(src);
  float *rank = malloc(n * sizeof(float));
  float *contrib = malloc(n * sizeof(float));
  for (int u = 0; u < n; u++) {
    rank[u] = 1.0f / n;
    contrib[u] = deg[u] ? rank[u] / deg[u] : 0.0f;
  }
  for (int it = 0; it < ITERATIONS; it++)
    for (int u = 0; u < n; u++) {
      float sum = 0.0f;
      for (long e = off[u]; e < off[u + 1]; e++)
        sum += contrib[nbr[e]];
      rank[u] = 0.15f / n + 0.85f * sum;
      contrib[u] = deg[u] ? rank[u] / deg[u] : 0.0f;
    }
  printf("%f\n", rank[0]);
  return 0;
}

/* Breadth-first search on a random CSR graph, as a C program the static interpreter
 * must run by itself: the graph as in csr_pr.c, then a queue-driven traversal whose
 * loops carry values through memory (dist, queue, head, tail). Build: cc -O0 -DSCALE=<s> */
#include <stdio.h>
#include <stdlib.h>

#ifndef SCALE
#define SCALE 12
#endif
#define DEGREE 16

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
  int *dist = malloc(n * sizeof(int));
  for (int u = 0; u < n; u++)
    dist[u] = -1;
  int *queue = malloc(n * sizeof(int));
  int source = rand() % n;
  dist[source] = 0;
  queue[0] = source;
  int head = 0, tail = 1;
  while (head < tail) {
    int u = queue[head++];
    for (long e = off[u]; e < off[u + 1]; e++) {
      int v = nbr[e];
      if (dist[v] < 0) {
        dist[v] = dist[u] + 1;
        queue[tail++] = v;
      }
    }
  }
  long reached = 0;
  for (int u = 0; u < n; u++)
    reached += dist[u] >= 0;
  printf("%ld\n", reached);
  return 0;
}

// compare_and_swap: a retry loop over a fresh read, and a claim with a constant old value
#include <cstdint>
void probe(long);
template <typename T> bool compare_and_swap(T &x, const T &old_val, const T &new_val) {
  if (x == old_val) { x = new_val; return true; } return false; }
uint64_t w[4];
int c[64];
int main() {
  long tries = 0;
  for (int pos = 0; pos < 64; pos++) {        // 64 bits into one word: one try each
    uint64_t o, n;
    do { tries++; o = w[pos / 64]; n = o | ((uint64_t)1 << (pos % 64)); } while (!compare_and_swap(w[pos / 64], o, n));
  }
  probe(tries);                               // 64
  probe((long)(w[0] == ~(uint64_t)0));        // 1
  for (int i = 0; i < 64; i++) c[i] = -1;
  long won = 0;
  for (int i = 0; i < 64; i++) { int h = -1; if (compare_and_swap(c[i / 8], h, i)) won++; }
  probe(won);                                 // 8: each slot claimed once
  long dist[4] = {100, 100, 100, 100};        // relax (sssp): a retry over a fresh read, not a bit update
  for (int i = 0; i < 64; i++) {
    long nd = 50 + (i * 7) % 13, od = dist[i % 4];
    while (nd < od) { if (compare_and_swap(dist[i % 4], od, nd)) break; od = dist[i % 4]; }
  }
  probe(dist[0] + dist[1] + dist[2] + dist[3]);  // 200: each the minimum, 50
  int parent[8]; for (int i = 0; i < 8; i++) parent[i] = -1;
  long claimed = 0;                           // claim through a fresh read (bfs)
  for (int i = 0; i < 64; i++) { int cur = parent[i % 8]; if (cur < 0 && compare_and_swap(parent[i % 8], cur, i)) claimed++; }
  probe(claimed); probe(parent[3]);           // 8, 3
  return 0;
}

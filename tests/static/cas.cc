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
  return 0;
}

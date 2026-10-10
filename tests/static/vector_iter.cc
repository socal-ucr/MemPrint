// vector iterators (__normal_iterator) in loops and algorithms (python tests/static/test_library.py)
#include <vector>
#include <algorithm>
#include <cstdint>
void probe(long);
int main() {
  std::vector<int> v;
  for (int i = 0; i < 6; i++) v.push_back((i * 5) % 7);      // 0 5 3 1 6 4
  long s = 0;
  for (auto it = v.begin(); it != v.end(); ++it) s += *it;
  probe(s);                                                  // 19
  auto b = v.begin();
  probe(*(b + 4)); probe(v.end() - v.begin()); probe(b[2]);  // 6 6 3
  probe(*std::max_element(v.begin(), v.end()));              // 6
  std::sort(v.begin(), v.end());
  probe(v[0] * 100 + v[5]);                                  // 6
  std::vector<int64_t> w(4, 9);
  std::vector<int64_t> x(4);
  std::copy(w.begin(), w.end(), x.begin());
  probe(x[3]);                                               // 9
  std::vector<std::vector<int>> bins(3);
  bins[2].push_back(77); bins[2].push_back(78);
  int frontier[4] = {5, 5, 5, 5};
  std::copy(bins[2].begin(), bins[2].end(), frontier + 1);
  probe(frontier[0] * 10000 + frontier[1] * 100 + frontier[2]);  // 57778
  auto it = v.begin(); it++; it += 2;
  probe(*it); probe(it < v.end());                           // 4 1 (sorted: 0 1 3 4 5 6)
  return 0;
}

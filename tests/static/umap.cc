// unordered_map iterators and the two compare-and-swap idioms (python tests/static/test_library.py)
#include <unordered_map>
#include <algorithm>
void probe(long);
int main() {
  std::unordered_map<int, int> m(32);
  using kvp = std::unordered_map<int, int>::value_type;
  for (int i = 0; i < 100; i++) m[i % 3 == 0 ? 7 : 4]++;
  probe(m.size());                                   // 2
  auto b = m.begin();
  probe(b->first + (*b).second);                     // 7+34 = 41 (insertion order here)
  auto mf = std::max_element(m.begin(), m.end(), [](const kvp& a, const kvp& c) { return a.second < c.second; });
  probe(mf->second); probe(mf->first);               // 66, 4
  long s = 0;
  for (auto it = m.begin(); it != m.end(); ++it) s += it->second;
  probe(s);                                          // 100
  for (auto& kv : m) s += kv.first;
  probe(s);                                          // 111
  probe(m.find(4) != m.end()); probe(m.find(5) == m.end()); probe(m.count(7));  // 1 1 1
  std::unordered_map<long, double> d; d[3] = 2.5;
  probe(d.begin()->first); probe((long)(d.begin()->second * 2));  // 3 5
  return 0;
}

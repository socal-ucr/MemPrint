// Small C++ programs for the static interpreter's C++ support: each probe(x) records the value the
// interpreter computes; tests/static/test_cpp.py checks them.
#include <algorithm>
#include <cstdint>
void probe(long);

template <typename T>
class Vec {                       // like GAP's pvector
 public:
  Vec() : start_(nullptr), end_(nullptr) {}
  explicit Vec(size_t n) { start_ = new T[n]; end_ = start_ + n; }
  Vec(size_t n, T init) : Vec(n) { fill(init); }
  Vec(Vec &&other) : start_(other.start_), end_(other.end_) { other.start_ = nullptr; other.end_ = nullptr; }
  ~Vec() { if (start_ != nullptr) delete[] start_; }
  void fill(T v) { for (size_t i = 0; i < size(); i++) start_[i] = v; }
  size_t size() const { return end_ - start_; }
  T &operator[](size_t i) { return start_[i]; }
  T *begin() const { return start_; }
  T *end() const { return end_; }
 private:
  T *start_;
  T *end_;
};

struct Pair {
  int u, v;
  Pair() {}
  Pair(int u, int v) : u(u), v(v) {}
};

struct Holder {
  int64_t count_ = -1;
  bool flag_ = true;
  int get() const { return count_ + 1; }
};

Vec<int> Make(long n) {
  Vec<int> out(n, 3);
  out[1] = 7;
  return out;
}

int main() {
  Holder h;
  probe(h.count_);                // -1: default member initialiser
  probe(h.get());                 // 0: method with this
  Vec<long> a(5, 2);
  probe(a.size());                // 5: delegating constructor, member function
  a[2] = 9;
  probe(a[2]);                    // 9: operator[] returns a reference
  Vec<int> b = Make(4);
  probe(b[1] + b[0]);             // 10: object returned by value
  long s = 0;
  for (int x : b) s += x;         // range-for over begin() / end()
  probe(s);                       // 16
  int &r = b[3];
  r = 5;
  probe(b[3]);                    // 5: reference variable
  Vec<Pair> p(3);
  p[0] = Pair(4, 6);
  probe(p[0].v - p[0].u);         // 2: temporary object assigned
  auto add = [&s](long d) { s += d; return s; };
  probe(add(4));                  // 20: lambda capturing by reference
  probe(std::min(s, 7L) + std::max(1L, 2L));  // 9: std::min / max
  Vec<int> c(4);
  for (int i = 0; i < 4; i++) c[i] = 4 - i;
  std::sort(c.begin(), c.end());
  probe(c[0] * 10 + c[3]);        // 14: std::sort on values
  int breaks();
  return breaks();
}
// (appended) break and continue
int breaks() {
  Vec<int> d(8);
  for (int i = 0; i < 8; i++) d[i] = i;
  long first_big = -1, odd = 0;
  for (int x : d) { if (x > 4) { first_big = x; break; } }
  for (int i = 0; i < 8; i++) { if (d[i] % 2 == 0) continue; odd += d[i]; }
  int k = 0;
  while (true) { k++; if (k == 6) break; }
  probe(first_big * 100 + odd * 10 + k);  // 5*100 + 16*10 + 6 = 666
  return 0;
}

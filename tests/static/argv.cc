#include <cstdlib>
#include <string>
void probe(long);
int main(int argc, char *argv[]) {
  int n = argc > 1 ? atoi(argv[1]) : 10;
  long m = std::stol(argv[2]);
  long x = strtol(argv[3], nullptr, 10);
  probe(argc); probe(n); probe(m); probe(x);
  int *a = new int[n];
  for (int i = 0; i < n; i++) a[i] = i;
  probe(a[n - 1]);
  return 0;
}

// Loops the C++ interpreter batches: prefix sums, output pointers advanced under conditions, nested
// reductions, conjunctions with a floating-point test, and pointer comparisons across objects.
// python tests/static/test_loops.py
void probe(long);

double norm(double *r, int n) {
  double s = 0;
  for (int i = 0; i < n; i++) s += r[i] * r[i];
  return s;
}

int main() {
  // 1. prefix sum (a scan): prefix[i] = total so far
  int n = 10;
  long *deg = new long[n];
  long *prefix = new long[n + 1];
  for (int i = 0; i < n; i++) deg[i] = i % 3 + 1;
  long total = 5;
  for (int i = 0; i < n; i++) {
    prefix[i] = total;
    total += deg[i];
  }
  prefix[n] = total;
  probe(prefix[4]);                 // 12
  probe(prefix[n]);                 // 24

  // 2. rows of a stencil filled through advancing pointers, under conditions
  int nx = 4, ny = 3;
  int *vals = new int[200];
  int **rows = new int *[nx * ny];
  int *cnt = new int[nx * ny];
  int *p = vals;
  long nnz = 0;
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      int row = iy * nx + ix, k = 0;
      rows[row] = p;
      for (int s = -1; s <= 1; s++) {
        if (ix + s >= 0 && ix + s < nx) { *p++ = row * 10 + s + 1; k++; }
      }
      cnt[row] = k;
      nnz += k;
    }
  }
  probe(nnz);                       // 30
  probe(p - vals);                  // 30
  probe(rows[5] - vals);            // 12
  probe(*rows[11]);                 // 110
  probe(cnt[3]);                    // 2

  // 3. nested sums, per outer iteration
  long m = 0;
  for (int j = 0; j < 4; j++) { for (int i = 0; i < j; i++) { m += i; } }
  probe(m);                         // 4

  // 4. a conjunction with an unknown floating-point test runs to its integer bound
  double *r = new double[n];
  for (int i = 0; i < n; i++) r[i] = 1.0 / (i + 1);
  double normr = norm(r, n), tol = 0.0;
  int iters = 0;
  for (int it = 1; it < 50 && normr > tol; it++) { normr = norm(r, n); iters++; }
  probe(iters);                     // 49

  // 5. pointers to different objects never compare equal
  probe(r == (double *)vals);       // 0
  probe(deg != prefix);             // 1
  return 0;
}

// a global scalar read and written through its address (NPB's randlc(&tran, amult)), and float globals
#include <cmath>
#define r23 (0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5*0.5)
#define r46 (r23*r23)
#define t23 (2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0*2.0)
#define t46 (t23*t23)
double randlc(double *x, double a){    
	double t1,t2,t3,t4,a1,a2,x1,x2,z;

	/*
	 * ---------------------------------------------------------------------
	 * break A into two parts such that A = 2^23 * A1 + A2.
	 * ---------------------------------------------------------------------
	 */
	t1 = r23 * a;
	a1 = (int)t1;
	a2 = a - t23 * a1;

	/*
	 * ---------------------------------------------------------------------
	 * break X into two parts such that X = 2^23 * X1 + X2, compute
	 * Z = A1 * X2 + A2 * X1  (mod 2^23), and then
	 * X = 2^23 * Z + A2 * X2  (mod 2^46).
	 * ---------------------------------------------------------------------
	 */
	t1 = r23 * (*x);
	x1 = (int)t1;
	x2 = (*x) - t23 * x1;
	t1 = a1 * x2 + a2 * x1;
	t2 = (int)(r23 * t1);
	z = t1 - t23 * t2;
	t3 = t23 * z + a2 * x2;
	t4 = (int)(r46 * t3);
	(*x) = t3 - t46 * t4;

	return (r46 * (*x));
}
void probe(long);
static double amult;
static double tran;
static void sp(int nz) { int nzv = 0; while (nzv < nz) { double v = randlc(&tran, amult); probe((long)(v * 1e6)); nzv++; } }
int main() {
  tran = 314159265.0; amult = 1220703125.0;
  probe((long)tran); probe((long)amult); double *px = &tran; probe((long)*px); probe((long)(r23 * 1e12)); probe((long)t23);
  double zeta = randlc(&tran, amult);
  probe((long)(zeta * 1e6));
  sp(2);
  int more();
  return more();
}
static double half = 0.5 * 3.0;
int more() { probe((long)(half * 10)); return 0; }

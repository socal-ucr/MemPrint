// probe(), the test programs' hook for values the interpreter checks (tests/static/cuda), for real runs.
#include <stdio.h>

void probe(long v) { printf("probe %ld\n", v); }
void probe(double v) { printf("probe %g\n", v); }

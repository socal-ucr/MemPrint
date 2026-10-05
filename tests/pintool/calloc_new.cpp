// calloc 1 MB and touch half of it; new[] 1 MB and touch it all; free both.
// Peak 1.5 MB; >= 1.5 MB freed.
#include <cstdlib>
#include "common.h"

int main()
{
    char *a = static_cast<char *>(calloc(1, 1 * MB));
    touch(a, 512 * KB);
    double *b = new double[MB / sizeof(double)];
    touch(b, 1 * MB);
    delete[] b;
    free(a);
    return 0;
}

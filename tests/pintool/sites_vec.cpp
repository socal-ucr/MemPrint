// -sites with operator new: a std::vector grown by push_back is one site, in the function that
// grows it (STL code is instantiated in the program), not in libstdc++.
#include <vector>

__attribute__((noinline)) static long grow_vector()
{
    std::vector< long > v;
    for (long i = 0; i < 100000; i++)
        v.push_back(i); // 800 KB of longs; capacity doubles to 131072 (1 MB)
    long s = 0;
    for (long x : v)
        s += x;
    return s;
}

int main() { return grow_vector() == 4999950000L ? 0 : 1; }

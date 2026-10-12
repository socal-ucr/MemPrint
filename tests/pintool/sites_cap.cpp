// Container growth per call site, for the MemPrintResize pass's capacities.
//   big():    one std::vector<long> grows to 128 Ki elements (1 MB) at one call site
//   small():  2000 vectors of the same type grow to 100 elements each at another call site
//   text():   a std::string grows to 256 KB by append
//   table():  an unordered_map<long, long> grows to 64 Ki entries (its bucket array is reallocated)
// Built with the pass, each growth goes through __mpr_capacity; with the call sites' capacities,
// the big vector, the string and the bucket array are allocated once, and the small vectors as before.
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

__attribute__((noinline)) long big()
{
    std::vector< long > v;
    for (long i = 0; i < 128 * 1024; i++) v.push_back(i);
    return v[v.size() - 1];
}

__attribute__((noinline)) long small()
{
    long sum = 0;
    for (int k = 0; k < 2000; k++)
    {
        std::vector< long > v;
        for (long i = 0; i < 100; i++) v.push_back(i);
        sum += v[99];
    }
    return sum;
}

__attribute__((noinline)) long text()
{
    std::string s;
    for (int i = 0; i < 256 * 1024 / 8; i++) s.append("abcdefgh");
    return (long)s.size();
}

__attribute__((noinline)) long table()
{
    std::unordered_map< long, long > m;
    for (long i = 0; i < 64 * 1024; i++) m[i] = i;
    return (long)m.size();
}

int main()
{
    printf("%ld %ld %ld %ld\n", big(), small(), text(), table());
    return 0;
}

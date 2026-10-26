# rlz-ref

This repository contains a C++ implementation of relative Lempel-Ziv (RLZ) compression [1] of differentially encoded
integer sequences, e.g. the differential suffix array [2, 3].

- `rlz<_diff_access>` stores the RLZ of the differential suffix array; its reference consists of segments of the
  input. It is built in parallel, optionally with the phrases on disk during the construction.
- A decode context decodes the values to the left or to the right of any position.

## Build

Requirements: a C++20 compiler (GCC 13 or later, Clang 18 or later), CMake 3.21 or later and OpenMP.

```shell
git clone https://github.com/LukasNalbach/rlz-ref.git
cd rlz-ref
cmake -B build
cmake --build build -j
```

`cmake` downloads the submodule [string-utils](https://github.com/LukasNalbach/string-utils). The tests are written to
`build/tests/` (`RLZ_REF_BUILD_TESTS`, `ON` if rlz-ref is the top-level project). The general options and environment
variables (`RLZ_REF_MARCH_NATIVE`, ...) are described in string-utils.

## C++ usage

Add rlz-ref to a CMake project and link the target `rlz_ref`:

```cmake
add_subdirectory(rlz-ref/ EXCLUDE_FROM_ALL)
target_link_libraries(<target> PRIVATE rlz_ref)
```

The construction reads the differential suffix array from a function and needs the disjoint interval sequence of
Phi^{-1}:

```cpp
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include <rlz_ref/construction/construction.hpp>

// the pairs (SA[i-1], SA[i]) of the run starts i of the BWT, sorted
struct phi_inv_intervals {
    std::vector<std::pair<uint64_t, uint64_t>> pairs;
    template <uint8_t c> uint64_t get(uint64_t i) const
    { return c == 0 ? pairs[i].first : pairs[i].second; }
    void clear() { pairs = {}; }
};

int main()
{
    // T$ = abaababaab$, SA = 10 7 2 8 5 0 3 9 6 1 4, BWT = bbbab$aaaaa
    std::vector<uint64_t> SA { 10, 7, 2, 8, 5, 0, 3, 9, 6, 1, 4 };
    std::string BWT = "bbbab$aaaaa";
    uint64_t n = SA.size();
    phi_inv_intervals intervals;

    for (uint64_t i = 0; i < n; i++)
        if (i == 0 || BWT[i] != BWT[i - 1])
            intervals.pairs.emplace_back(SA[(i + n - 1) % n], SA[i]);

    std::sort(intervals.pairs.begin(), intervals.pairs.end());
    uint64_t r = intervals.pairs.size();

    // SA^d[i] = SA[i] - SA[i-1] + n (the construction adds n to SA[0])
    auto SAd = [&](uint64_t, uint64_t i) {
        return i == 0 ? SA[0] : SA[i] + n - SA[i - 1];
    };

    rlz<_diff_access> sa;
    rlz<_diff_access>::construction<decltype(SAd), phi_inv_intervals>(
        sa, SAd, intervals, n, r, { .num_threads = 2 }).build();
    std::cout << sa.size() << std::endl; // 11

    // SA[1..5] with next(), starting from SA[0] = n - 1
    auto ctx = sa.decode_context();
    ctx.init_right(1);
    ctx.set_value(sa.size() - 1);
    for (int k = 0; k < 5; k++) std::cout << ctx.next() << " ";
    std::cout << std::endl; // 7 2 8 5 0

    // SA[9..6] from right to left, starting at position 10 with SA[10] = 4
    ctx = sa.decode_context();
    ctx.init_left(10);
    ctx.set_value(4);
    ctx.report_left(6, [](uint64_t i, uint64_t v) { std::cout << v << " "; });
    std::cout << std::endl; // 1 6 9 3
}
```

`serialize` and `load` store and load the data structure.

## Tests

```shell
build/tests/test-rlz-ref
```

The tests check random inputs for at most `RLZ_REF_TEST_SECONDS` seconds (default 10) or `RLZ_REF_TEST_ITERATIONS`
inputs (default 1000) per function. A failed test prints its seed; `RLZ_REF_TEST_SEED=<seed>` repeats its inputs.

## References

[1] Shanika Kuruppu, Simon J. Puglisi and Justin Zobel. Relative Lempel-Ziv compression of genomes for large-scale
storage and retrieval. In String Processing and Information Retrieval (SPIRE), 2010.

[2] Simon J. Puglisi and Bella Zhukova. Relative Lempel-Ziv compression of suffix arrays. In String Processing and
Information Retrieval (SPIRE), 2020.

[3] Patrick Dinklage, Johannes Fischer, Lukas Nalbach and Jan Zumbrink. RLZ-r and LZ-End-r: Enhancing Move-r. In String
Processing and Information Retrieval (SPIRE), 2025. ([paper](https://arxiv.org/abs/2507.17300))

## License

This project is licensed under the MIT License, see [LICENSE](LICENSE). The submodules in [external/](external/) have
their own licenses.

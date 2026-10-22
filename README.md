# rlz-ref

This repository contains an implementation of relative Lempel-Ziv (RLZ) compression [1] of differentially encoded integer sequences, e.g. the differential suffix array [2, 3], with sequential decoding to the left or to the right from any position. The reference consists of segments of the input itself.

## Build Instructions

```shell
git clone git@github.com:LukasNalbach/rlz-ref.git
cd rlz-ref
cmake -B build
cmake --build build -j
```

CMake fetches the required submodule [string-utils](https://github.com/LukasNalbach/string-utils). The tests are built into `build/tests/`; `RLZ_REF_BUILD_TESTS` turns them on or off (`ON` if rlz-ref is the top-level project). The general build options (`RLZ_REF_MARCH_NATIVE`, `RLZ_REF_ENABLE_LTO`, ...) and environment variables (`RLZ_REF_TMPDIR`, ...) are described in [string-utils](https://github.com/LukasNalbach/string-utils).

## Usage in C++

```cmake
add_subdirectory(external/rlz-ref EXCLUDE_FROM_ALL)
target_link_libraries(<target> PRIVATE rlz_ref)
```

The class `rlz<_diff_access>` (`#include <rlz_ref/rlz.hpp>`) stores the differential suffix array SA^d[i] = SA[i] - SA[i-1] parsed greedily into z phrases: a copy phrase is a longest prefix of the rest of SA^d that occurs in the reference R, a literal phrase stores one value. The base class `rlz_phrases<reference_t>` stores the phrases (R, the phrase types, the phrase starts, the sources and the literal phrases) for any type of reference.

`rlz<_diff_access>::construction<sad_func_t, intervals_t>` (`#include <rlz_ref/construction/construction.hpp>`) builds it from a function `value_at(thread, i)` returning SA^d[i] (SA[0] for i = 0) and from the disjoint interval sequence of Phi^{-1}: `get<0>(i)` returns the starting position p_i of the i-th input interval (ascending, p_0 = 0), `get<1>(i)` returns Phi^{-1}(p_i), and `clear()` frees the sequence after the frequencies of the values of SA^d have been counted. The parameters (`rlz_construction_params`) that matter are `num_threads`, `reference_target_length` (maximum length of R; 0: tuned on a sample of SA^d), `revise_reference` and `space` (`_on_disk` stores the phrases in temporary files during the construction).

A decode context stores a position i and the value s = SA[i]; `init_right(i)` with s = SA[i-1] prepares it to decode SA[i], SA[i+1], ..., and `init_left(i)` with s = SA[i] prepares it to decode SA[i-1], SA[i-2], ....

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

    // now pos() = 6 and value() = SA[5]; skip to position 8, then report
    // SA[8..10]
    ctx.skip_right(8);
    ctx.report_right(10, [](uint64_t i, uint64_t v) {
        std::cout << i << ":" << v << " ";
    });
    std::cout << std::endl; // 8:6 9:1 10:4

    // SA[9..6] from right to left, starting at position 10 with SA[10] = 4
    ctx = sa.decode_context();
    ctx.init_left(10);
    ctx.set_value(4);
    ctx.report_left(6, [](uint64_t i, uint64_t v) { std::cout << v << " "; });
    std::cout << std::endl; // 1 6 9 3
    std::cout << ctx.pos() << " " << ctx.value() << std::endl; // 6 3

    // SA[5] with prev() from position 6; then decode to the right from
    // position 5 (s has to be SA[4] = 5) and to the left from position 7
    // (s has to be SA[7] = 9)
    ctx = sa.decode_context();
    ctx.init_left(6);
    ctx.set_value(3);
    std::cout << ctx.prev() << std::endl; // 0
    ctx.turn_right();
    ctx.set_value(5);
    std::cout << ctx.next() << " " << ctx.next() << std::endl; // 0 3
    ctx.turn_left();
    ctx.set_value(9);
    std::cout << ctx.prev() << std::endl; // 3
}
```

`serialize` and `load` store the data structure as it is in memory.

## Tests

```shell
build/tests/test-rlz-ref
```

The tests check random inputs for at most `RLZ_REF_TEST_SECONDS` seconds (default 10) or `RLZ_REF_TEST_ITERATIONS` inputs (default 1000) per functionality; `RLZ_REF_TEST_SEED=<seed>` replays the inputs of a seed that a failed test printed.

## References

[1] Shanika Kuruppu, Simon J. Puglisi and Justin Zobel. Relative Lempel-Ziv compression of genomes for large-scale storage and retrieval.
In String Processing and Information Retrieval (SPIRE), 2010.

[2] Simon J. Puglisi and Bella Zhukova. Relative Lempel-Ziv compression of suffix arrays.
In String Processing and Information Retrieval (SPIRE), 2020.

[3] Patrick Dinklage, Johannes Fischer, Lukas Nalbach and Jan Zumbrink. RLZ-r and LZ-End-r: Enhancing Move-r.
In String Processing and Information Retrieval (SPIRE), 2025. ([paper](https://arxiv.org/abs/2507.17300))

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

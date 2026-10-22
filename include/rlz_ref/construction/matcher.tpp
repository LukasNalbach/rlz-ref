/**
 * part of LukasNalbach/rlz-ref
 *
 * MIT License
 *
 * Copyright (c) Lukas Nalbach
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef RLZ_REF_RLZ_REF_CONSTRUCTION_MATCHER_TPP
#define RLZ_REF_RLZ_REF_CONSTRUCTION_MATCHER_TPP

#include <rlz_ref/construction/matcher.hpp>

template <typename value_t>
void rlz_reference_matcher::rank_values(value_t&& value, uint64_t size, uint64_t threads)
{
    check_reference_length(size);
    *this = rlz_reference_matcher(always_wide);
    reference_length = size;
    wide = always_wide || reference_length >= wide_from;
    if (reference_length == 0) return;
    threads = std::max<uint64_t>(1, threads);

    no_init_vector<uint64_t> distinct;
    resize_no_init(distinct, reference_length);

    #pragma omp parallel for num_threads(threads)
    for (uint64_t i = 0; i < reference_length; i++) distinct[i] = uint64_t(value(i));

    radix_sort(distinct.begin(), distinct.end(), key_of_value {}, threads);

    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    distinct.shrink_to_fit();
    num_distinct = distinct.size();

    if (!wide) resize_no_init(ranks, reference_length);
    else resize_no_init(ranks_wide, reference_length);

    #pragma omp parallel for num_threads(threads)
    for (uint64_t i = 0; i < reference_length; i++) {
        const uint64_t rank = std::lower_bound(distinct.begin(), distinct.end(), uint64_t(value(i)))
            - distinct.begin();

        if (!wide) ranks[i] = int32_t(rank);
        else ranks_wide[i] = rank;
    }

    // number of occurrences in R of each rank; sa_beg_of_value stores their prefix sums
    std::vector<uint64_t> occurrences(num_distinct, 0);

    for (uint64_t i = 0; i < reference_length; i++)
        occurrences[wide ? uint64_t(ranks_wide[i]) : uint64_t(uint32_t(ranks[i]))]++;

    if (!wide) resize_no_init(sa_beg_of_value, num_distinct + 1);
    else resize_no_init(sa_beg_of_value_wide, num_distinct + 1);
    uint64_t before = 0;

    for (uint64_t rank = 0; rank <= num_distinct; rank++) {
        if (!wide) sa_beg_of_value[rank] = uint32_t(before);
        else sa_beg_of_value_wide[rank] = before;
        if (rank < num_distinct) before += occurrences[rank];
    }

    distinct_values = std::move(distinct);
}

inline void rlz_reference_matcher::build_over_ranks(uint64_t threads)
{
    if (reference_length == 0) return;
    threads = std::max<uint64_t>(1, threads);

    if (!wide) {
        resize_no_init(sa, reference_length);

        const int32_t failed = threads > 1
            ? libsais_int_omp(ranks.data(), sa.data(), int32_t(reference_length), int32_t(num_distinct),
                0, int32_t(threads))
            : libsais_int(ranks.data(), sa.data(), int32_t(reference_length), int32_t(num_distinct), 0);

        if (failed != 0) throw std::runtime_error("libsais failed for an RLZ reference");
        ranks = no_init_vector<int32_t>();
    } else {
        resize_no_init(sa_wide, reference_length);

        if (libsais40(ranks_wide.data(), sa_wide.data(), reference_length, num_distinct, 0, threads) != 0)
            throw std::runtime_error("libsais40 failed for an RLZ reference");

        ranks_wide = no_init_vector<uint40_t>();
    }
}

template <typename sequence_t>
inline std::pair<uint64_t, uint64_t> rlz_reference_matcher::longest_match(sequence_t&& sequence,
    uint64_t max_len) const
{
    if (reference_length == 0 || max_len == 0) return { 0, 0 };
    uint64_t first = 0;
    uint64_t last = reference_length - 1;
    uint64_t known = 0; // length of the common prefix of the sequence and all suffixes in SA[first..last]

    // restrict the search to the suffixes starting with the first value of the sequence
    if (!distinct_values.empty()) {
        const uint64_t value = sequence.first();
        const uint64_t rank = std::lower_bound(distinct_values.begin(), distinct_values.end(), value)
            - distinct_values.begin();
        if (rank == distinct_values.size() || distinct_values[rank] != value) return { 0, 0 };
        first = wide ? uint64_t(sa_beg_of_value_wide[rank]) : uint64_t(sa_beg_of_value[rank]);
        last = (wide ? uint64_t(sa_beg_of_value_wide[rank + 1]) : uint64_t(sa_beg_of_value[rank + 1])) - 1;
        known = 1;
    }

    uint64_t longest = 0;
    uint64_t longest_source = 0;
    bool is_smaller = false; // whether the suffix compared last is lexicographically smaller than the sequence

    // compares the suffix SA[i] with the sequence, skipping their first from symbols; returns the
    // length of their longest common prefix
    auto compare = [&](uint64_t i, uint64_t from) {
        const uint64_t source = source_at(i);
        const uint64_t bound = std::min<uint64_t>(max_len, reference_length - source);
        const uint64_t lcp = sequence.lcp(source, from, bound);

        if (lcp > longest || (lcp == longest && source < longest_source)) {
            longest = lcp;
            longest_source = source;
        }

        // a suffix that is a proper prefix of the sequence is smaller than the sequence
        is_smaller = lcp < max_len && (source + lcp == reference_length || sequence.is_smaller_at(source, lcp));
        return lcp;
    };

    uint64_t lcp_first = compare(first, known);
    if (lcp_first == max_len || !is_smaller) return { longest, longest_source };
    if (last == first) return { longest, longest_source };
    uint64_t lcp_last = compare(last, known);
    if (lcp_last == max_len || is_smaller) return { longest, longest_source };

    // binary search; the suffix SA[first] is smaller and the suffix SA[last] is larger than the sequence
    while (last - first > 1) {
        const uint64_t middle = first + (last - first) / 2;
        const uint64_t lcp = compare(middle, std::min<uint64_t>(lcp_first, lcp_last));
        if (lcp == max_len) break;

        if (is_smaller) {
            first = middle;
            lcp_first = lcp;
        } else {
            last = middle;
            lcp_last = lcp;
        }
    }

    return { longest, longest_source };
}

#endif // RLZ_REF_RLZ_REF_CONSTRUCTION_MATCHER_TPP

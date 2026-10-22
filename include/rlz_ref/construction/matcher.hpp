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

#ifndef RLZ_REF_RLZ_REF_CONSTRUCTION_MATCHER_HPP
#define RLZ_REF_RLZ_REF_CONSTRUCTION_MATCHER_HPP

#include <algorithm>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include <misc/sort.hpp>
#include <libsais.h>
#include <libsais40.hpp>

#include <misc/utils.hpp>

/**
 * @brief a sequence of integers to be matched in a reference of integers (see
 *        rlz_reference_matcher::longest_match)
 * @tparam reference_t type of the reference; reference[i] returns R[i]
 * @tparam read_t type of the function returning the values of the sequence; read(k) returns its k-th value
 */
template <typename reference_t, typename read_t>
struct rlz_value_sequence {
    const reference_t& reference;
    read_t& read;

    // returns the length (at most bound) of the longest common prefix of the sequence and R[source..|R|-1],
    // given that their first from symbols are equal
    inline uint64_t lcp(uint64_t source, uint64_t from, uint64_t bound)
    {
        uint64_t len = from;
        while (len < bound && uint64_t(reference[source + len]) == uint64_t(read(len))) len++;
        return len;
    }

    // returns whether R[source + pos] is smaller than the value at position pos of the sequence
    inline bool is_smaller_at(uint64_t source, uint64_t pos)
    {
        return uint64_t(reference[source + pos]) < uint64_t(read(pos));
    }

    // returns the first value of the sequence
    inline uint64_t first() { return uint64_t(read(0)); }
};

/**
 * @brief finds the longest prefix of a sequence that occurs in the reference R
 *
 * Stores the suffix array of R (4 bytes per entry if |R| < 2^31, else 5 bytes; |R| is at most
 * max_reference_length = 2^38, the limit of libsais40). A match is found by a
 * binary search for the sequence in the suffix array, which compares the sequence with O(log |R|) suffixes
 * of R. For a reference of integers, the values are replaced by their ranks among the distinct values of R,
 * and a table stores for each distinct value the range of the suffix array containing the suffixes starting
 * with it; the binary search starts in this range.
 */
class rlz_reference_matcher {
protected:
    // minimum |R| for which the suffix array entries are stored in 5 bytes (else in 4 bytes)
    static constexpr uint64_t wide_from = uint64_t(1) << 31;

    bool always_wide = false; // true <=> the suffix array entries are stored in 5 bytes regardless of |R|
    bool wide = false; // true <=> the suffix array entries are stored in 5 bytes
    uint64_t reference_length = 0; // |R|
    uint64_t num_distinct = 0; // number of distinct values in R (only for a reference of integers)
    // SA, the suffix array of R: [0..|R|-1] SA[i] is the starting position in R of the i-th smallest suffix
    // of R; only one of sa (if !wide) and sa_wide (if wide) is non-empty
    no_init_vector<int32_t> sa;
    no_init_vector<uint40_t> sa_wide;
    // the ranks of the values of R: [0..|R|-1] ranks[x] is the rank of R[x] among the distinct values of R;
    // only one of ranks (if !wide) and ranks_wide (if wide) is non-empty; freed by build_over_ranks()
    no_init_vector<int32_t> ranks;
    no_init_vector<uint40_t> ranks_wide;
    // [0..num_distinct-1] the distinct values of R, ascending (only for a reference of integers)
    no_init_vector<uint64_t> distinct_values;
    /* the suffix array intervals of the distinct values: [0..num_distinct] sa_beg_of_value[x] is the index in
       SA of the first suffix starting with distinct_values[x]; sa_beg_of_value[num_distinct] = |R|; only one of
       sa_beg_of_value (if !wide) and sa_beg_of_value_wide (if wide) is non-empty */
    no_init_vector<uint32_t> sa_beg_of_value;
    no_init_vector<uint64_t> sa_beg_of_value_wide;

    // returns SA[i]
    inline uint64_t source_at(uint64_t i) const
    {
        return wide ? uint64_t(sa_wide[i]) : uint64_t(uint32_t(sa[i]));
    }

public:
    // largest |R| (the suffix array of a longer R cannot be built with libsais40)
    static constexpr uint64_t max_reference_length = uint64_t(SA_INT40_MAX_N);

    // throws std::length_error if size is larger than max_reference_length
    static void check_reference_length(uint64_t size)
    {
        if (size > max_reference_length) [[unlikely]] {
            throw std::length_error("the RLZ reference is longer than 2^"
                + std::to_string(std::bit_width(max_reference_length) - 1) + " symbols");
        }
    }

    rlz_reference_matcher() = default;

    // always_wide = true <=> the suffix array entries are stored in 5 bytes regardless of |R|
    explicit rlz_reference_matcher(bool always_wide) : always_wide(always_wide) { }

    // returns |R|
    inline uint64_t size() const { return reference_length; }

    /**
     * @brief computes the ranks of the values of a reference of integers among its distinct values and
     *        the table of the distinct values; the suffix array is built from the ranks by build_over_ranks()
     * @tparam value_t type of the function returning the values of R
     * @param value value(i) returns R[i]; called by several threads at the same time
     * @param size |R| (at most max_reference_length)
     * @param threads number of threads to use
     */
    template <typename value_t>
    void rank_values(value_t&& value, uint64_t size, uint64_t threads);

    /**
     * @brief builds the suffix array of the ranks computed by rank_values() (R is not accessed)
     * @param threads number of threads to use
     */
    void build_over_ranks(uint64_t threads);

    /**
     * @brief builds the suffix array of a reference of integers (rank_values() and build_over_ranks())
     * @tparam value_t type of the function returning the values of R
     * @param value value(i) returns R[i]; called by several threads at the same time
     * @param size |R|
     * @param threads number of threads to use
     */
    template <typename value_t>
    void build_over_values(value_t&& value, uint64_t size, uint64_t threads)
    {
        rank_values(value, size, threads);
        build_over_ranks(threads);
    }

    /**
     * @brief returns the longest prefix of a sequence that occurs in R
     * @tparam sequence_t type of the sequence (see rlz_value_sequence); it has to
     *         provide
     *         uint64_t lcp(uint64_t source, uint64_t from, uint64_t bound): returns the length (at most
     *         bound) of the longest common prefix of the sequence and R[source..|R|-1], given that their
     *         first from symbols are equal
     *         bool is_smaller_at(uint64_t source, uint64_t pos): returns whether R[source + pos] is smaller
     *         than the symbol at position pos of the sequence
     *         uint64_t first(): returns the first symbol of the sequence
     * @param sequence the sequence; it is accessed at most up to the first symbol after the match
     * @param max_len upper bound for the length of the match
     * @return (length of the match, starting position of the match in R); the length is 0 if the first
     *         symbol of the sequence does not occur in R
     */
    template <typename sequence_t>
    inline std::pair<uint64_t, uint64_t> longest_match(sequence_t&& sequence, uint64_t max_len) const;
};

#include <rlz_ref/construction/matcher.tpp>

#endif // RLZ_REF_RLZ_REF_CONSTRUCTION_MATCHER_HPP

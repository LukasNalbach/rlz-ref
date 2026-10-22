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

#include <algorithm>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <misc/utils.hpp>
#include <rlz_ref/construction/construction.hpp>
#include <rlz_ref/rlz.hpp>
#include "test-fuzz.hpp"
#include "test-inputs.hpp"
#include "test-references.hpp"

std::mt19937 gen(test_seed());
uint64_t max_num_threads = omp_get_max_threads();

// maximum input size of verify_decode_context; adjusted by run_fuzz, down to min_input_size
uint64_t max_input_size = 165000;
constexpr uint64_t min_input_size = 2000;

// smallest and largest symbol of the random inputs (the two largest char values do not occur)
constexpr char min_input_symbol = std::numeric_limits<char>::min();
constexpr char max_input_symbol = std::numeric_limits<char>::max() - 2;

/**
 * @brief the disjoint interval sequence of Phi^{-1} of a suffix array SA[0..n-1]: the pairs (SA[i-1], SA[i]) of the
 *        starting positions i of the runs in the BWT (SA[n-1] for i = 0), ascending by their first value (see
 *        rlz::construction)
 */
struct phi_inv_intervals {
    std::vector<std::pair<uint64_t, uint64_t>> pairs;

    // returns the starting position of the i-th input interval (column 0) or its image under Phi^{-1} (column 1)
    template <uint8_t column>
    uint64_t get(uint64_t i) const { return column == 0 ? pairs[i].first : pairs[i].second; }

    void clear() { pairs = {}; }
};

/**
 * @brief builds the rlz of the differential suffix array of an input with a terminator
 * @param input the input
 * @param SA set to the suffix array of the input with the terminator (see compute_reference_sa)
 */
static rlz<_diff_access> rlz_of_suffix_array(const std::string& input, std::vector<int64_t>& SA)
{
    SA = compute_reference_sa<char, std::string>(input, input.size(), max_num_threads);
    const uint64_t n = SA.size();

    // returns BWT[i] = T[SA[i] - 1] (the terminator for SA[i] = 0), the terminator as -1
    auto bwt = [&](uint64_t i) -> int64_t { return SA[i] == 0 ? -1 : int64_t(uint8_t(input[SA[i] - 1])); };

    phi_inv_intervals intervals;

    for (uint64_t i = 0; i < n; i++) {
        if (i == 0 || bwt(i) != bwt(i - 1)) intervals.pairs.emplace_back(SA[i == 0 ? n - 1 : i - 1], SA[i]);
    }

    std::sort(intervals.pairs.begin(), intervals.pairs.end());
    const uint64_t r = intervals.pairs.size();

    // returns SA^d[i] = SA[i] - SA[i-1] + n (SA[0] = n - 1 for i = 0)
    auto SAd = [&](uint64_t, uint64_t i) -> uint64_t {
        return i == 0 ? n - 1 : (uint64_t(SA[i]) + n) - uint64_t(SA[i - 1]);
    };

    rlz<_diff_access> parsing;
    rlz<_diff_access>::construction<decltype(SAd), phi_inv_intervals> construction(parsing, SAd, intervals, n, r,
        { .num_threads = max_num_threads });
    construction.build();
    return parsing;
}

// each of the following four functions verifies one group of operations of the rlz decode context for
// one query (1 <= l < c <= r <= n-1) against the suffix array SA

// range reporting: report_right decodes SA[l..r] left-to-right, report_left decodes SA[c-1..l] right-to-left
static void verify_decode_report(const rlz<_diff_access>& parsing, const std::vector<int64_t>& SA,
    uint64_t l, uint64_t c, uint64_t r)
{
    {
        auto context = parsing.decode_context();
        context.init_right(l);
        context.set_value(SA[l - 1]);
        EXPECT_EQ(context.pos(), l);

        uint64_t count = 0;
        context.report_right(r, [&](uint64_t p, uint64_t v) {
            EXPECT_EQ(v, SA[p]);
            count++;
        });
        EXPECT_EQ(count, r - l + 1);
    }

    {
        auto context = parsing.decode_context();
        context.init_left(c);
        context.set_value(SA[c]);
        EXPECT_EQ(context.pos(), c);
        EXPECT_EQ(context.value(), SA[c]);

        std::vector<uint64_t> reported;
        context.report_left(l, [&](uint64_t, uint64_t v) { reported.push_back(v); });
        EXPECT_EQ(reported.size(), c - l);
        for (uint64_t k = 0; k < reported.size() && k < c - l; k++)
            EXPECT_EQ(reported[k], SA[c - 1 - k]);
    }
}

// single-stepping: next() decodes SA[l..r] one value forward, prev() decodes SA[c-1..l] one value backward,
// plus the set_pos()/set_value() accessor round-trip
static void verify_decode_step(const rlz<_diff_access>& parsing, const std::vector<int64_t>& SA,
    uint64_t l, uint64_t c, uint64_t r)
{
    {
        auto context = parsing.decode_context();
        context.init_right(l);
        context.set_value(SA[l - 1]);
        for (uint64_t p = l; p <= r; p++)
            EXPECT_EQ(context.next(), SA[p]);
    }

    {
        auto context = parsing.decode_context();
        context.init_left(c);
        context.set_value(SA[c]);
        for (uint64_t p = c; p > l; p--)
            EXPECT_EQ(context.prev(), SA[p - 1]);
    }

    {
        auto context = parsing.decode_context();
        context.set_pos(c);
        EXPECT_EQ(context.pos(), c);
        context.set_value(SA[c]);
        EXPECT_EQ(context.value(), SA[c]);
    }
}

// skipping: skip_right jumps l -> c then decodes SA[c..r]; skip_left jumps r -> c then decodes SA[c-1..l]
static void verify_decode_skip(const rlz<_diff_access>& parsing, const std::vector<int64_t>& SA,
    uint64_t l, uint64_t c, uint64_t r)
{
    {
        auto context = parsing.decode_context();
        context.init_right(l);
        context.set_value(SA[l - 1]);
        context.skip_right(c);
        EXPECT_EQ(context.pos(), c);
        EXPECT_EQ(context.value(), SA[c - 1]);

        uint64_t count = 0;
        context.report_right(r, [&](uint64_t, uint64_t v) {
            EXPECT_EQ(v, SA[c + count]);
            count++;
        });
        EXPECT_EQ(count, r - c + 1);
    }

    {
        auto context = parsing.decode_context();
        context.init_left(r);
        context.set_value(SA[r]);
        context.skip_left(c);
        EXPECT_EQ(context.pos(), c);
        EXPECT_EQ(context.value(), SA[c]);

        std::vector<uint64_t> reported;
        context.report_left(l, [&](uint64_t, uint64_t v) { reported.push_back(v); });
        EXPECT_EQ(reported.size(), c - l);
        for (uint64_t k = 0; k < reported.size() && k < c - l; k++)
            EXPECT_EQ(reported[k], SA[c - 1 - k]);
    }
}

// changing the direction: turn_left() makes a context prepared to decode to the right decode to the left (then
// prev()), turn_right() the converse (then next()), and turn_left() followed by turn_right() changes nothing
static void verify_decode_turn(const rlz<_diff_access>& parsing, const std::vector<int64_t>& SA,
    uint64_t l, uint64_t c, uint64_t r)
{
    {
        auto context = parsing.decode_context();
        context.init_right(c);
        context.turn_left();
        EXPECT_EQ(context.pos(), c);
        context.set_value(SA[c]);
        for (uint64_t p = c; p > l; p--)
            EXPECT_EQ(context.prev(), SA[p - 1]);
    }

    {
        auto context = parsing.decode_context();
        context.init_left(c);
        context.turn_right();
        EXPECT_EQ(context.pos(), c);
        context.set_value(SA[c - 1]);
        for (uint64_t p = c; p <= r; p++)
            EXPECT_EQ(context.next(), SA[p]);
    }

    {
        auto context = parsing.decode_context();
        context.init_right(c);
        context.turn_left();
        context.turn_right();
        context.set_value(SA[c - 1]);
        for (uint64_t p = c; p <= r; p++)
            EXPECT_EQ(context.next(), SA[p]);
    }
}

/**
 * @brief builds the rlz of the differential suffix array of a random repetitive input and verifies the operations of
 * its rlz decode context (rlz::decode_context_t) in parallel against the suffix array: init_left()/
 * init_right(), turn_left()/turn_right(), next()/prev(), skip_left()/skip_right(), report_left()/
 * report_right(), pos()/value() and set_pos()/set_value()
 */
static void verify_decode_context()
{
    // the input is at least min(min_length, max_input_size / 4) long
    constexpr uint64_t min_length = 500;
    // number of queries: n / values_per_query, in [min_queries..max_queries]
    constexpr uint64_t values_per_query = 20, min_queries = 100, max_queries = 2000;

    std::string input = random_repetitive_input<std::string>(std::min<uint64_t>(min_length, max_input_size / 4),
        max_input_size, min_input_symbol, max_input_symbol, gen());
    uint64_t input_size = input.size();

    std::vector<int64_t> SA;
    const rlz<_diff_access> parsing = rlz_of_suffix_array(input, SA);
    ASSERT_FALSE(parsing.empty());

    uint64_t n = parsing.size();
    ASSERT_EQ(n, SA.size());

    uint64_t total_queries = std::clamp<uint64_t>(n / values_per_query, min_queries, max_queries);
    uint64_t queries_per_thread = std::max<uint64_t>(1, total_queries / max_num_threads);

    // [0..max_num_threads-1] seed of the random number generator of each thread
    std::vector<uint64_t> thread_seeds(max_num_threads);
    for (uint64_t& seed : thread_seeds) seed = gen();

    #pragma omp parallel num_threads(max_num_threads)
    {
        std::mt19937 gen_thr(thread_seeds[omp_get_thread_num()]);

        for (uint64_t q = 0; q < queries_per_thread; q++) {
            uint64_t l = std::uniform_int_distribution<uint64_t>(1, n - 2)(gen_thr);
            uint64_t r = std::uniform_int_distribution<uint64_t>(l + 1, n - 1)(gen_thr);
            uint64_t c = std::uniform_int_distribution<uint64_t>(l + 1, r)(gen_thr);

            verify_decode_report(parsing, SA, l, c, r);
            verify_decode_step(parsing, SA, l, c, r);
            verify_decode_skip(parsing, SA, l, c, r);
            verify_decode_turn(parsing, SA, l, c, r);
        }
    }
}

TEST(test_rlz_ref, decode)
{
    run_fuzz("rlz-ref", { { "decode", [](uint64_t) { verify_decode_context(); } } },
        fuzz_iterations(), &max_input_size, min_input_size);
}

/**
 * @brief an rlz decode context together with the position p and the direction it is expected to have; every
 *        operation verifies the decoded values, pos() and value() against the suffix array. The operations
 *        keep 1 <= p <= n-1, so that SA[p-1] (decoding to the right) and SA[p] (decoding to the left) exist.
 */
struct verified_decode_context {
    rlz<_diff_access>::decode_context_t context;
    const std::vector<int64_t>& SA;
    uint64_t n; // length of SA
    uint64_t p; // expected value of context.pos()
    bool to_left; // true <=> the context is prepared to decode to the left

    /**
     * @brief prepares a decode context at position p with init_left() or init_right()
     * @param p [1..n-1] position in the suffix array
     * @param to_left true <=> the context decodes to the left
     */
    verified_decode_context(const rlz<_diff_access>& parsing, const std::vector<int64_t>& SA, uint64_t p,
        bool to_left)
        : context(parsing.decode_context()), SA(SA), n(SA.size()), p(p), to_left(to_left)
    {
        if (to_left) context.init_left(p);
        else context.init_right(p);
        context.set_value(to_left ? SA[p] : SA[p - 1]);
    }

    // checks pos() and value() (SA[p] when decoding to the left, SA[p-1] when decoding to the right)
    void check_state() const
    {
        EXPECT_EQ(context.pos(), p);
        EXPECT_EQ(context.value(), to_left ? SA[p] : SA[p - 1]);
    }

    // returns whether step() and report() keep 1 <= p <= n-1
    bool can_decode() const { return to_left ? p >= 2 : p + 2 <= n; }

    // the smallest and largest target position of report() ([1..p-1] or [p..n-2]), if can_decode()
    uint64_t min_report_target() const { return to_left ? 1 : p; }
    uint64_t max_report_target() const { return to_left ? p - 1 : n - 2; }

    // the smallest and largest target position of skip() ([1..p] or [p..n-1])
    uint64_t min_skip_target() const { return to_left ? 1 : p; }
    uint64_t max_skip_target() const { return to_left ? p : n - 1; }

    // decodes one value with prev() or next()
    void step()
    {
        if (to_left) {
            EXPECT_EQ(context.prev(), SA[p - 1]);
            p--;
        } else {
            EXPECT_EQ(context.next(), SA[p]);
            p++;
        }

        check_state();
    }

    // moves the context to the target position with skip_left() or skip_right()
    void skip(uint64_t target)
    {
        if (to_left) context.skip_left(target);
        else context.skip_right(target);
        p = target;
        check_state();
    }

    // decodes SA[target..p-1] with report_left() or SA[p..target] with report_right()
    void report(uint64_t target)
    {
        // position of the next value that has to be reported
        uint64_t expected_pos = to_left ? p - 1 : p;

        auto check = [&](uint64_t j, uint64_t v) {
            EXPECT_EQ(j, expected_pos);
            EXPECT_EQ(v, SA[j]);
            if (to_left) expected_pos--;
            else expected_pos++;
        };

        if (to_left) {
            context.report_left(target, check);
            EXPECT_EQ(expected_pos, target - 1);
            p = target;
        } else {
            context.report_right(target, check);
            EXPECT_EQ(expected_pos, target + 1);
            p = target + 1;
        }

        check_state();
    }

    // changes the decoding direction with turn_left() or turn_right() and sets the value it requires
    void turn()
    {
        if (to_left) context.turn_right();
        else context.turn_left();
        to_left = !to_left;
        EXPECT_EQ(context.pos(), p);
        context.set_value(to_left ? SA[p] : SA[p - 1]);
    }
};

/**
 * @brief builds the rlz of the differential suffix array of a random repetitive input and verifies, for random start
 *        positions, a report_left() or report_right() followed by a random sequence of prev()/next(),
 *        skip_left()/skip_right(), report_left()/report_right() and turn_left()/turn_right() against the
 *        suffix array (each operation has to leave the context in a state from which decoding can continue)
 */
static void verify_decode_continuation()
{
    // the input is at least min(min_length, max_input_size / 4) long
    constexpr uint64_t min_length = 500;
    // number of operation sequences: n / values_per_sequence, in [min_sequences..max_sequences]
    constexpr uint64_t values_per_sequence = 20, min_sequences = 100, max_sequences = 2000;
    // number of operations of a sequence, and the maximum distance between the position of the context and the
    // target position of a skip or a report
    constexpr uint64_t operations_per_sequence = 16, max_target_distance = 64;

    // the operations of a sequence
    enum operation_t : uint64_t { STEP, SKIP, REPORT, TURN, NUM_OPERATIONS };

    std::string input = random_repetitive_input<std::string>(std::min<uint64_t>(min_length, max_input_size / 4),
        max_input_size, min_input_symbol, max_input_symbol, gen());

    std::vector<int64_t> SA;
    const rlz<_diff_access> parsing = rlz_of_suffix_array(input, SA);
    ASSERT_FALSE(parsing.empty());

    uint64_t n = parsing.size();
    ASSERT_EQ(n, SA.size());

    uint64_t total_sequences = std::clamp<uint64_t>(n / values_per_sequence, min_sequences, max_sequences);
    uint64_t sequences_per_thread = std::max<uint64_t>(1, total_sequences / max_num_threads);

    // [0..max_num_threads-1] seed of the random number generator of each thread
    std::vector<uint64_t> thread_seeds(max_num_threads);
    for (uint64_t& seed : thread_seeds) seed = gen();

    #pragma omp parallel num_threads(max_num_threads)
    {
        std::mt19937 gen_thr(thread_seeds[omp_get_thread_num()]);
        auto uniform = [&](uint64_t min, uint64_t max) {
            return std::uniform_int_distribution<uint64_t>(min, max)(gen_thr);
        };

        for (uint64_t q = 0; q < sequences_per_thread; q++) {
            // the start position allows a report in both directions
            verified_decode_context walk(parsing, SA, uniform(2, n - 2), uniform(0, 1) == 1);

            for (uint64_t k = 0; k < operations_per_sequence; k++) {
                operation_t operation = k == 0 ? REPORT : operation_t(uniform(0, NUM_OPERATIONS - 1));
                if ((operation == STEP || operation == REPORT) && !walk.can_decode()) operation = TURN;

                // the target positions of a skip or a report are at most max_target_distance away from p
                uint64_t min_target = walk.to_left ? walk.p - std::min(walk.p, max_target_distance) : walk.p;
                uint64_t max_target = walk.to_left ? walk.p : walk.p + max_target_distance;

                switch (operation) {
                    case STEP: walk.step(); break;
                    case SKIP:
                        walk.skip(uniform(std::max(min_target, walk.min_skip_target()),
                            std::min(max_target, walk.max_skip_target())));
                        break;
                    case REPORT:
                        walk.report(uniform(std::max(min_target, walk.min_report_target()),
                            std::min(max_target, walk.max_report_target())));
                        break;
                    default: walk.turn(); break;
                }
            }
        }
    }
}

TEST(test_rlz_ref, decode_continuation)
{
    run_fuzz("rlz-ref", { { "decode continuation", [](uint64_t) { verify_decode_continuation(); } } },
        fuzz_iterations(), &max_input_size, min_input_size);
}

/**
 * @brief verifies on a small text, for every start position, direction and report target, a report_left() or
 *        report_right() followed by prev()/next() up to the end, by turn_left()/turn_right() and prev()/next(),
 *        by turn_left()/turn_right() and a report up to the end, and by a skip up to the end, turn_left()/
 *        turn_right() and prev()/next()
 */
TEST(test_rlz_ref, decode_continuation_small)
{
    std::string input = "abaababaab";
    std::vector<int64_t> SA;
    const rlz<_diff_access> parsing = rlz_of_suffix_array(input, SA);
    ASSERT_FALSE(parsing.empty());
    uint64_t n = parsing.size();
    ASSERT_EQ(n, SA.size());

    // the reported failure: init_left(n-1) and report_left(n-5), then turn_right() and next() decoded a wrong
    // SA[n-5]
    {
        constexpr uint64_t start_offset = 1, report_offset = 5;
        verified_decode_context walk(parsing, SA, n - start_offset, true);
        walk.report(n - report_offset);
        walk.turn();
        walk.step();
    }

    // the continuations after the report
    enum continuation_t : uint64_t { STEPS, TURN_STEPS, TURN_REPORT, SKIP_TURN_STEPS, NUM_CONTINUATIONS };

    for (uint64_t start = 2; start <= n - 2; start++) {
        for (bool to_left : { false, true }) {
            verified_decode_context initial(parsing, SA, start, to_left);

            for (uint64_t target = initial.min_report_target(); target <= initial.max_report_target(); target++) {
                for (uint64_t continuation = 0; continuation < NUM_CONTINUATIONS; continuation++) {
                    SCOPED_TRACE("start " + std::to_string(start) + (to_left ? ", left" : ", right") +
                        ", report target " + std::to_string(target) + ", continuation " +
                        std::to_string(continuation));

                    verified_decode_context walk(parsing, SA, start, to_left);
                    walk.report(target);

                    if (continuation == SKIP_TURN_STEPS) {
                        walk.skip(walk.to_left ? walk.min_skip_target() : walk.max_skip_target());
                    }

                    if (continuation != STEPS) walk.turn();

                    if (continuation == TURN_REPORT) {
                        if (walk.can_decode()) {
                            walk.report(walk.to_left ? walk.min_report_target() : walk.max_report_target());
                        }
                    } else {
                        while (walk.can_decode()) walk.step();
                    }
                }
            }
        }
    }
}

/**
 * @brief verifies rlz_reference_matcher::longest_match against a comparison of the sequence with every suffix
 *        of the reference
 * @param sigma number of distinct symbols in the reference
 * @param wide whether the suffix array entries are stored in 5 bytes
 */
static void verify_reference_matcher(uint64_t sigma, bool wide)
{
    // largest length of the reference
    constexpr uint64_t max_size = 3000;
    /* at each position, a copy (if the position is > min_copy_position) and a run start with probability
       1 / piece_period each; maximum length of a copy and of a run */
    constexpr uint64_t piece_period = 60, min_copy_position = 20, max_copy_length = 300, max_run_length = 200;
    // number of queries; maximum length of a sequence and of the copied part of a sequence
    constexpr uint64_t num_queries = 300, max_sequence_length = 500, max_copied_length = 500;

    const uint64_t size = std::uniform_int_distribution<uint64_t>(1, max_size)(gen);
    // maps a random number to one of sigma large integers
    auto symbol = [&](uint64_t drawn) { return (drawn % sigma) * 1000003 + 12345; };

    std::vector<uint64_t> reference(size);

    for (uint64_t i = 0; i < size;) {
        const uint64_t kind = gen() % piece_period;
        uint64_t length = 1;

        if (i > min_copy_position && kind == 0) {
            // copy a substring from an earlier position
            const uint64_t from = std::uniform_int_distribution<uint64_t>(0, i - 1)(gen);
            length = std::min<uint64_t>(size - i, std::uniform_int_distribution<uint64_t>(1, max_copy_length)(gen));
            for (uint64_t k = 0; k < length; k++) reference[i + k] = reference[from + k];
        } else if (kind == 1) {
            // a run of one symbol
            length = std::min<uint64_t>(size - i, std::uniform_int_distribution<uint64_t>(1, max_run_length)(gen));
            const uint64_t run_symbol = symbol(gen());
            for (uint64_t k = 0; k < length; k++) reference[i + k] = run_symbol;
        } else {
            reference[i] = symbol(gen());
        }

        i += length;
    }

    rlz_reference_matcher matcher(wide);
    matcher.build_over_values([&](uint64_t i) { return reference[i]; }, size, max_num_threads);

    ASSERT_EQ(matcher.size(), size);

    for (uint64_t query = 0; query < num_queries; query++) {
        /* the sequence: a substring of the reference (empty in every fifth query) followed by random symbols; in every
           third query, one symbol is replaced */
        const uint64_t length = std::uniform_int_distribution<uint64_t>(1, max_sequence_length)(gen);
        const uint64_t from = std::uniform_int_distribution<uint64_t>(0, size - 1)(gen);
        const uint64_t copied = query % 5 == 0 ? 0 : std::min<uint64_t>({ length, size - from,
            std::uniform_int_distribution<uint64_t>(0, max_copied_length)(gen) });
        std::vector<uint64_t> sequence(length);
        for (uint64_t k = 0; k < length; k++) sequence[k] = k < copied ? reference[from + k] : symbol(gen());
        if (query % 3 == 0) sequence[gen() % length] = symbol(gen());
        const uint64_t max_len = std::uniform_int_distribution<uint64_t>(1, length)(gen);

        uint64_t longest = 0;

        for (uint64_t source = 0; source < size; source++) {
            uint64_t lcp = 0;
            while (lcp < max_len && source + lcp < size && reference[source + lcp] == sequence[lcp]) lcp++;
            longest = std::max<uint64_t>(longest, lcp);
        }

        auto read = [&](uint64_t k) { return sequence[k]; };

        const auto [found, source] = matcher.longest_match(
            rlz_value_sequence<std::vector<uint64_t>, decltype(read)> { reference, read }, max_len);

        ASSERT_EQ(found, longest) << "sigma " << sigma << ", wide " << wide;
        ASSERT_LE(source + found, size);

        for (uint64_t k = 0; k < found; k++) {
            ASSERT_EQ(reference[source + k], sequence[k]) << "at " << k << " of a match of " << found;
        }
    }
}

TEST(test_rlz_ref, reference_matcher)
{
    // number of random references per alphabet size
    constexpr uint64_t num_rounds = 8;

    for (uint64_t round = 0; round < num_rounds; round++)
        for (uint64_t sigma : { 1, 2, 4, 20, 256 })
            for (bool wide : { false, true }) verify_reference_matcher(sigma, wide);
}

// checks that a revision drops the additions with the smallest gains if R would become longer than the given maximum
TEST(test_rlz_ref, revision_respects_max_reference_length)
{
    // the input consists of num_blocks blocks; R is one block in the middle; the ends of R are lengthened by up to
    // max_lengthening blocks
    constexpr uint64_t block_length = 64, num_blocks = 16, max_lengthening = 4;
    // phrases ending at the end of R and starting at its start (each next to a phrase of max_lengthening blocks)
    constexpr uint32_t num_ended = 1000, num_began = 500;
    const uint64_t n = num_blocks * block_length, mid = num_blocks / 2;

    rlz_revision_params params;
    params.block_length = block_length;
    params.max_lengthening = max_lengthening;
    rlz_revision revision(n, 1, params, test_seed());

    rlz_reference_segments ref_segments;
    ref_segments.segments = { rlz_segment_t { mid * block_length, (mid + 1) * block_length } };
    ref_segments.compute_begins();
    const uint64_t reference_length = ref_segments.length();

    rlz_parse_stats<uint32_t> stats = revision.new_stats<uint32_t>({});
    stats.ended_with.assign(max_lengthening, 0);
    stats.began_with.assign(max_lengthening, 0);
    stats.ended_with[max_lengthening - 1] = num_ended;
    stats.began_with[max_lengthening - 1] = num_began;
    stats.z = num_ended + num_began;

    // one byte per symbol of R; every phrase is worth about as much as the input divided by the number of phrases
    const rlz_size_estimate_t estimate { double(n + reference_length), double(reference_length), double(stats.z) };
    auto reference_symbol = [](uint64_t) { return uint64_t(0); };
    auto block_symbols = [](uint64_t, uint64_t, uint64_t) { return [](uint64_t) { return uint64_t(0); }; };

    auto revise = [&](uint64_t max_reference_length) {
        return revision.revise(ref_segments, reference_length, stats, estimate, reference_symbol, block_symbols,
            max_reference_length);
    };

    // without a tight maximum, both ends are lengthened
    const rlz_revised_reference both = revise(rlz_max_reference_length);
    ASSERT_EQ(both.ref_segments.length(), reference_length + 2 * max_lengthening * block_length);
    ASSERT_EQ(both.num_lengthened_ends, 2u);

    // room for one lengthening: the end (the larger gain) is lengthened
    const rlz_revised_reference one = revise(reference_length + max_lengthening * block_length);
    ASSERT_EQ(one.ref_segments.length(), reference_length + max_lengthening * block_length);
    ASSERT_EQ(one.num_lengthened_ends, 1u);
    ASSERT_EQ(one.ref_segments.segments.front().beg, mid * block_length);
    ASSERT_LT(one.gain, both.gain);

    // no room: R stays as it is
    const rlz_revised_reference none = revise(reference_length);
    ASSERT_EQ(none.ref_segments.length(), reference_length);
    ASSERT_FALSE(none.changes());
}

// checks the limits of the reference matcher and of the counters of a revision
TEST(test_rlz_ref, construction_limits)
{
    ASSERT_EQ(rlz_reference_matcher::max_reference_length, rlz_max_reference_length);
    const uint64_t too_long = rlz_reference_matcher::max_reference_length + 1;
    rlz_reference_matcher matcher;
    EXPECT_THROW(matcher.build_over_values([](uint64_t) { return uint64_t(0); }, too_long, 1), std::length_error);

    constexpr uint32_t largest = std::numeric_limits<uint32_t>::max();
    uint32_t counter = largest - 1;
    rlz_revision::count(counter);
    ASSERT_EQ(counter, largest);
    rlz_revision::count(counter);
    ASSERT_EQ(counter, largest);
    uint32_t sum = largest - 1;
    rlz_revision::add_count(sum, 2);
    ASSERT_EQ(sum, largest);
}

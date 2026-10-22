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

#ifndef RLZ_REF_RLZ_REF_CONSTRUCTION_REVISION_HPP
#define RLZ_REF_RLZ_REF_CONSTRUCTION_REVISION_HPP

#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtl/phmap.hpp>

#include <misc/log.hpp>
#include <rlz_ref/construction/karp_rabin.hpp>
#include <rlz_ref/construction/reference_choice.hpp>
#include <rlz_ref/construction/tuning.hpp>

/**
 * @brief parameters for revising R (see rlz_revision)
 *
 * The input is divided into blocks of block_length symbols. A revision adds blocks of the input to R,
 * lengthens segments of R by blocks and removes blocks from R.
 */
struct rlz_revision_params {
    uint64_t block_length = 4096; // length of a block
    double sample_share = 0.04; // share of the blocks of the input that is parsed in a round on a sample
    uint64_t max_sampled_blocks = 32768; // maximum number of blocks that are parsed in a round on a sample
    uint64_t sampled_rounds = 2; // number of rounds on a sample, before the whole input is parsed
    uint64_t whole_rounds = 2; // maximum number of parses of the whole input
    // minimum share of the size of the rlz that a round has to be expected to save for R to be revised
    double min_gain = 0.01;
    // minimum number of phrases starting in a block for the block to be considered for R
    uint64_t min_phrases_in_dense_block = 3;
    // maximum number of such blocks considered in a round (the blocks with the most phrases)
    uint64_t max_dense_blocks = 16384;
    /* minimum share of the k-mers of a block that occur neither in R nor in another group for the block to
       start a group of its own; else it is counted to the group that contains most of its k-mers */
    double min_new_kmer_share = 0.25;
    // number of phrases that are assumed to still start in a block after it has been added to R
    double phrases_left_after_adding = 2.0;
    uint64_t max_lengthening = 16; // maximum number of blocks an end of a segment is lengthened by in one round
    uint64_t min_kmer_length = 16; // minimum length of the k-mers the dense blocks are grouped by
    uint64_t max_kmer_length = 1024; // maximum length of the k-mers the dense blocks are grouped by
};

/**
 * @brief statistics on the phrases of a parse of the input (or of a sample of its blocks) with R; the counters
 *        ended_with, began_with and copied_from stop at the largest uint32_t (see rlz_revision::count)
 * @tparam phrase_count_t type of the number of phrases starting in a block
 */
template <typename phrase_count_t>
struct rlz_parse_stats {
    // the parsed blocks, ascending; empty <=> all blocks were parsed
    std::vector<uint64_t> blocks;
    /* the number of phrases starting in each parsed block: [0..number of parsed blocks - 1] phrases[b] =
       number of phrases starting in the b-th parsed block */
    std::vector<phrase_count_t> phrases;
    /* the phrases ending at the ends of the segments of R: [0..|segments| * max_lengthening - 1]
       ended_with[s * max_lengthening + e - 1] = number of phrases ending at the end of segment s of R that
       are followed by a phrase of length e blocks (rounded up, at most max_lengthening) */
    std::vector<uint32_t> ended_with;
    /* the phrases starting at the starts of the segments of R: [0..|segments| * max_lengthening - 1]
       began_with[s * max_lengthening + e - 1] = number of phrases starting at the start of segment s of R
       that are preceded by a phrase of length e blocks (rounded up, at most max_lengthening) */
    std::vector<uint32_t> began_with;
    /* the phrases copying from each block of R: [0..ceil(|R| / block_length) - 1] copied_from[b] = number
       of phrases copying from block b of R; empty <=> not counted */
    std::vector<uint32_t> copied_from;
    uint64_t z = 0; // number of phrases starting in the parsed blocks
    uint64_t z_l = 0; // number of literal phrases starting in the parsed blocks
    double scale = 1.0; // number of blocks of the input divided by the number of parsed blocks

    // returns whether all blocks were parsed
    bool whole() const { return blocks.empty(); }
};

// a revised reference (see rlz_revision::revise)
struct rlz_revised_reference {
    rlz_reference_segments ref_segments; // the segments of the input that the revised reference consists of
    uint64_t num_added_blocks = 0; // number of added blocks of the input
    uint64_t num_lengthened_ends = 0; // number of lengthened ends of segments
    uint64_t num_lengthening_blocks = 0; // total number of blocks the ends were lengthened by
    double estimated_bytes = 0; // estimated size in bytes of the rlz before the revision
    double gain = 0; // number of bytes the added and lengthened segments are expected to save

    // returns whether blocks were added or segments lengthened
    bool changes() const { return num_added_blocks != 0 || num_lengthened_ends != 0; }
};

/**
 * @brief revises R using the statistics of a parse of the input with R
 *
 * The input is divided into blocks of block_length symbols. A revision (see revise)
 * 1. removes the blocks of R that no phrase copies from (only after a parse of the whole input that counted
 *    them, see rlz_parse_stats::copied_from),
 * 2. lengthens the segments of R at the ends at which phrases end (and at the starts at which phrases
 *    start) by the number of blocks with the largest expected gain,
 * 3. groups the blocks in which at least min_phrases_in_dense_block phrases start by their common k-mers
 *    and adds the block with the most phrases of each group to R, if the expected gain is positive; k is
 *    half the average phrase length (between min_kmer_length and max_kmer_length),
 * 4. if R becomes longer than the largest supported |R|, drops the additions of 2. and 3. with the smallest
 *    expected gains until it fits.
 * The expected gain is the estimated size of the saved phrases minus the size of the added blocks.
 */
class rlz_revision {
public:
    // value of prev_phrase_t::end_segment, if the previous phrase does not end at the end of a segment
    static constexpr uint64_t none = std::numeric_limits<uint64_t>::max();

    // the previous phrase of a parse (see count_phrase)
    struct prev_phrase_t {
        uint64_t length = 0; // length of the previous phrase (0 if there is none)
        uint64_t end_segment = none; // index of the segment of R at whose end the previous phrase ends
        bool counted = false; // whether the previous phrase was counted
    };

    // the counters of a parse of one thread (see rlz_parse_stats)
    struct counters_t {
        std::vector<uint32_t> ended_with;
        std::vector<uint32_t> began_with;
        std::vector<uint32_t> copied_from;
    };

    using thread_counters_t = std::vector<cache_line_padded<counters_t>>;

    // increments a counter of counters_t, unless it has the largest value of its type
    static inline void count(uint32_t& counter) { counter += counter != std::numeric_limits<uint32_t>::max(); }

    // adds a counter of counters_t to another one; the sum is at most the largest value of their type
    static inline void add_count(uint32_t& sum, uint32_t counter)
    { sum = uint32_t(std::min<uint64_t>(uint64_t(sum) + counter, std::numeric_limits<uint32_t>::max())); }

protected:
    uint64_t n; // length of the input
    uint64_t threads; // number of threads to use
    rlz_revision_params params;
    uint64_t block_length; // length of a block
    uint64_t num_blocks; // number of blocks of the input (the last one may be shorter)
    uint64_t max_lengthening; // maximum number of blocks an end of a segment is lengthened by
    std::mt19937_64 random; // random number generator the sampled blocks are drawn with

    // returns the starting position of a block in the input
    inline uint64_t block_beg(uint64_t block) const { return block * block_length; }

    // returns the end position (exclusive) of a block in the input
    inline uint64_t block_end(uint64_t block) const { return std::min<uint64_t>(n, (block + 1) * block_length); }

    /**
     * @brief returns the number of blocks a phrase needs to be lengthened by (rounded up, at most
     *        max_lengthening)
     * @param length length of the phrase
     */
    inline uint64_t blocks_needed(uint64_t length) const
    {
        return std::min<uint64_t>(max_lengthening, (length + block_length - 1) / block_length);
    }

public:
    /**
     * @param n length of the input
     * @param threads number of threads to use
     * @param params revision parameters
     * @param seed seed of the random number generator the sampled blocks are drawn with
     */
    rlz_revision(uint64_t n, uint64_t threads, const rlz_revision_params& params, uint64_t seed)
        : n(n), threads(threads), params(params)
        , block_length(std::max<uint64_t>(1, params.block_length))
        , num_blocks((n + block_length - 1) / block_length)
        , max_lengthening(std::max<uint64_t>(1, params.max_lengthening))
        , random(seed)
    { }

    // returns the revision parameters
    inline const rlz_revision_params& parameters() const { return params; }

    // returns the number of blocks of the input
    inline uint64_t num_input_blocks() const { return num_blocks; }

    // draws the blocks of the input to parse in a round on a sample; returns them in ascending order
    std::vector<uint64_t> draw_blocks();

    /**
     * @brief returns the range of the input a sampled block is parsed in: the block and
     *        rlz_sample_window_length symbols (rounded up to blocks) before it
     * @param block a block of the input
     * @return [first, second) the range
     */
    std::pair<uint64_t, uint64_t> sampled_range(uint64_t block) const;

    /**
     * @brief returns empty statistics of a parse
     * @param blocks the blocks to parse, in ascending order; empty <=> the whole input is parsed
     */
    template <typename phrase_count_t>
    rlz_parse_stats<phrase_count_t> new_stats(std::vector<uint64_t> blocks) const;

    /**
     * @brief returns zeroed counters for each thread
     * @param num_segments number of segments of R
     * @param num_reference_blocks number of blocks of R whose phrases are counted (0 <=> none)
     */
    thread_counters_t new_counters(uint64_t num_segments, uint64_t num_reference_blocks) const;

    // adds the counters of the threads to stats
    template <typename phrase_count_t>
    void add_counters(rlz_parse_stats<phrase_count_t>& stats, const thread_counters_t& counters) const;

    /**
     * @brief counts the next phrase of a parse in ended_with and began_with (see rlz_parse_stats)
     * @param prev the previous phrase of the parse; replaced by the next phrase
     * @param counters the counters of the calling thread
     * @param ref_segments the segments of the input that R consists of
     * @param counted whether the phrase starts in a block whose phrases are counted
     * @param literal whether the phrase is literal
     * @param length length of the phrase
     * @param source starting position in R of the phrase (if it is a copy phrase)
     */
    void count_phrase(prev_phrase_t& prev, counters_t& counters, const rlz_reference_segments& ref_segments,
        bool counted, bool literal, uint64_t length, uint64_t source) const;

    /**
     * @brief computes a revised reference from the statistics of a parse with R
     * @param ref_segments the segments of the input that R consists of
     * @param reference_length |R|
     * @param stats statistics of the parse
     * @param estimate estimated size of the rlz built with R, from the statistics
     * @param reference_symbol reference_symbol(i) returns R[i] as a symbol of a k-mer (see rlz_karp_rabin);
     *        called by several threads at the same time
     * @param block_symbols block_symbols(thread, beg, end) returns a function that returns for j in
     *        [0..end-beg-1] the input symbol at position beg + j as a symbol of a k-mer; the function is valid
     *        until the next call by the same thread
     * @param max_reference_length largest |R| after the revision (at least reference_length); the additions with
     *        the smallest gains are dropped until the revised R fits
     * @return the segments of the revised reference and the number of bytes it is expected to save
     */
    template <typename phrase_count_t, typename reference_symbol_t, typename block_symbols_t>
    rlz_revised_reference revise(const rlz_reference_segments& ref_segments, uint64_t reference_length,
        const rlz_parse_stats<phrase_count_t>& stats, const rlz_size_estimate_t& estimate,
        reference_symbol_t&& reference_symbol, block_symbols_t&& block_symbols,
        uint64_t max_reference_length = rlz_max_reference_length);

    /**
     * @brief returns the log message describing a revision: |R| and below it, indented, z, the size, the numbers of
     *        added blocks, of lengthened ends of segments and of blocks added at them, and the expected saving, one per
     *        line (without a final line break)
     * @param reference_length |R| before the revision
     * @param stats statistics of the parse the revision is computed from
     * @param revised the revision
     */
    template <typename phrase_count_t>
    static std::string describe(uint64_t reference_length, const rlz_parse_stats<phrase_count_t>& stats,
        const rlz_revised_reference& revised);
};

#include <rlz_ref/construction/revision.tpp>

#endif // RLZ_REF_RLZ_REF_CONSTRUCTION_REVISION_HPP

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

#ifndef RLZ_REF_RLZ_REF_CONSTRUCTION_CONSTRUCTION_HPP
#define RLZ_REF_RLZ_REF_CONSTRUCTION_CONSTRUCTION_HPP

#include <libsais40_types.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>

#include <gtl/phmap.hpp>
#include <hash_table5.hpp>

#include <misc/temp_files.hpp>
#include <rlz_ref/construction/matcher.hpp>
#include <rlz_ref/construction/reference_choice.hpp>
#include <rlz_ref/construction/revision.hpp>
#include <rlz_ref/construction/spill_vector.hpp>
#include <rlz_ref/construction/tuning.hpp>
#include <rlz_ref/rlz.hpp>

// whether the construction keeps the phrases in memory or stores them in temporary files
enum rlz_space_t : uint8_t { _in_memory = 0, _on_disk = 1 };

// optional parameters for the construction of the rlz of SA^d
struct rlz_construction_params {
    // maximum number of threads to use during the construction (see rlz::construction::min_values_per_thread)
    uint64_t num_threads = omp_get_max_threads();
    rlz_space_t space = _in_memory; // whether the phrases are kept in memory or stored in temporary files
    bool log = false; // controls whether to print log messages
    std::ostream* measurement_file = nullptr; // measurement file the durations of the construction phases are written to
    std::string measurement_key_suffix = ""; // suffix appended to each measurement key
    // file the temporary files are placed next to; empty <=> they are placed in the temporary directory
    std::string tmp_files_beside = "";
    // maximum length of R; 0 <=> chosen by the construction (see rlz::construction::tune_reference)
    uint64_t reference_target_length = 0;
    // controls whether the length of R is tuned; if not, min(n / 3, 5.2 * r) is used
    bool tune_reference_length = true;
    // controls whether a tuned reference is revised (see rlz::construction::revise_reference)
    bool revise_reference = true;
};

/**
 * @brief scores segments of SA^d by the frequencies of their values
 *
 * The score of a segment is the sum of sqrt(frequency) over its distinct values that R does not contain yet,
 * divided by the length of the segment.
 * @tparam value_func_t type of the function returning SA^d[i]
 */
template <typename value_func_t>
class rlz_value_frequency_scorer {
public:
    // type of the hash map storing the frequency of each value in SA^d; frequency 0 <=> R contains the value
    using frequencies_t = emhash5::HashMap<uint64_t, uint64_t, std::identity>;

protected:
    value_func_t& value_at; // value_at(thread, i) returns SA^d[i]
    frequencies_t& frequency; // the frequency of each value in SA^d; 0 <=> R contains the value
    // [0..num_threads-1] the values a thread has already counted in the segment it is scoring
    std::vector<cache_line_padded<gtl::flat_hash_set<uint64_t, std::identity>>> seen;

public:
    rlz_value_frequency_scorer(value_func_t& value_at, frequencies_t& frequency, uint64_t num_threads)
        : value_at(value_at), frequency(frequency), seen(num_threads)
    { }

    // returns the score of the segment SA^d[beg..end-1]
    float score(uint64_t thread, uint64_t beg, uint64_t end);

    // sets the frequencies of the values in SA^d[beg..end-1] to 0
    void cover(uint64_t thread, uint64_t beg, uint64_t end)
    {
        for (uint64_t i = beg; i < end; i++) frequency.find(value_at(thread, i))->second = 0;
    }
};

/**
 * @brief constructs the rlz of SA^d, the differential suffix array: SA^d[i] = SA[i] - SA[i-1] + n for i in
 *        [1..n-1] and SA^d[0] = SA[0] (stored as SA[0] + n, see first_value)
 *
 * 1. computes the frequency of each value in SA^d from I_Phi^{-1}
 * 2. chooses R (see rlz_reference_choice and rlz_value_frequency_scorer); unless a length is given, several
 *    lengths of R are tried on a sample of SA^d (see tune_reference) and R is revised (see revise_reference)
 * 3. builds the suffix array of R (see rlz_reference_matcher)
 * 4. parses SA^d greedily into longest matches in R; each thread parses one range of SA^d; R can be revised
 *    by the parse and SA^d parsed again (see build_parsing)
 * @tparam value_func_t type of the function returning SA^d[i]
 * @tparam intervals_t type of the disjoint interval sequence of Phi^{-1}: for i in [0..r-1], get<0>(i) returns the
 *         starting position p_i of the i-th input interval in ascending order (p_0 = 0) and get<1>(i) returns
 *         Phi^{-1}(p_i); clear() frees it
 */
template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
class rlz<kind>::construction {
protected:
    // ############################# CONSTANTS AND TYPES #############################

    // maximum length of a copy phrase (the lengths of the copy phrases are stored in 16 bits)
    static constexpr uint64_t max_copy_length = std::numeric_limits<uint16_t>::max();
    // |R| is at most n / max_reference_divisor
    static constexpr uint64_t max_reference_divisor = 3;
    // number of values in a block of SA^d (see rlz_revision)
    static constexpr uint64_t revision_block_length = 1024;
    // parameters of the revision of R
    static constexpr rlz_revision_params revision_params {
        .block_length = revision_block_length,
        .min_kmer_length = 2,
        .max_kmer_length = revision_block_length / 2 };
    // minimum number of values of SA^d per thread; a shorter input is built with fewer threads
    static constexpr uint64_t min_values_per_thread = 1000;
    // initial number of entries of the buffer of the values read ahead (see parse_range)
    static constexpr uint64_t initial_lookahead_length = 1024;
    // buffer size in bytes of the file buffers of the phrases (for _on_disk)
    static constexpr uint64_t file_buffer_size = 128 * 1024;

    // a reference tried by tune_reference()
    using tried_t = rlz_tried_reference_t<interleaved_bit_aligned_vectors>;

    // a window of SA^d in the sample used by tune_reference() (positions in the sample)
    struct sampled_window_t {
        uint64_t beg;          // starting position of the values read for the window
        uint64_t counted_from; // starting position of the window; the phrases starting from here on are counted
        uint64_t end;          // end position of the window (exclusive)
    };

    // ############################# MEMBER VARIABLES #############################

    rlz& built; // the rlz to build
    value_func_t value_at; // value_at(i_p, i) returns SA^d[i]; evaluated by thread i_p
    intervals_t& I_Phi_inv; // the disjoint interval sequence of Phi^{-1}; cleared after computing the frequencies

    uint64_t n; // the length of T
    uint64_t r; // r, the number of runs in L
    uint64_t p; // the number of threads to use
    rlz_space_t space; // whether the phrases are kept in memory or stored in temporary files
    bool log; // controls whether to print log messages
    std::ostream* measurement_file; // measurement file the durations of the phases are written to (or nullptr)
    std::string measurement_key_suffix; // suffix appended to each measurement key
    std::string tmp_files_beside; // file the temporary files are placed next to
    uint64_t given_reference_target_length; // maximum length of R given by the caller (0 if none)
    bool tune_reference_length; // controls whether the length of R is tuned
    bool revise_tuned_reference; // controls whether a tuned reference is revised
    rlz_revision revision; // revises R (see revise_reference and build_parsing)

    /* the ranges of SA^d parsed by the threads: [0..p] range_bounds[i_p] is the starting position of the range
       of thread i_p; range_bounds[p] = n; strictly increasing */
    std::vector<uint64_t> range_bounds;
    std::string prefix_tmp_files; // prefix of the temporary files
    std::chrono::steady_clock::time_point time; // time of the start of the last build phase
    uint64_t baseline_mem_usage = 0; // memory allocation at the start of the construction

    uint64_t reference_length = 0;        // |R|
    uint64_t reference_target_length = 0; // maximum length of R
    uint64_t segment_length = 0;          // maximum length of a segment of R
    uint64_t num_candidates = 0;          // number of candidate segments scored per round
    bool reference_tuned = false; // true <=> tune_reference() has built reference, reference_segments and matcher

    // the frequency of each value in SA^d; frequency 0 <=> R contains the value
    typename rlz_value_frequency_scorer<value_func_t>::frequencies_t value_frequency;

    /* R, the chosen segments of SA^d, concatenated in ascending order of their starting positions:
       [0..|R|-1] R[x] is a value of SA^d, value_width(n) bits each */
    interleaved_bit_aligned_vectors reference;

    // the segments of SA^d that a tuned reference consists of, ascending (see revise_reference)
    std::vector<rlz_segment_t> reference_segments;

    // suffix array of R
    rlz_reference_matcher matcher;

    /* the lengths of the copy phrases of each thread (CPL): [0..p-1] CPL_of_thread[i_p][x] is the length of
       the x-th copy phrase in the range of thread i_p (at most max_copy_length) */
    std::vector<std::vector<uint16_t>> CPL_of_thread;
    /* the starting positions in R of the copy phrases of each thread (SR): [0..p-1] SR_of_thread[i_p][x] is
       the starting position in R of the x-th copy phrase in the range of thread i_p; bit_width(|R|) bits each */
    std::vector<interleaved_bit_aligned_vectors> SR_of_thread;
    /* the literal phrases of each thread (LP): [0..p-1] LP_of_thread[i_p][x] is the value of SA^d of the x-th
       literal phrase in the range of thread i_p (see first_value); value_width(n) bits each */
    std::vector<interleaved_bit_aligned_vectors> LP_of_thread;
    /* the phrase types of each thread (PT): [0..p-1] PT_of_thread[i_p][x] = 1 <=> the x-th phrase in the range
       of thread i_p is literal; larger than the number of phrases during the parse, trimmed after it */
    std::vector<sdsl::bit_vector> PT_of_thread;

    // [0..p-1] temporary files with the same content as CPL_of_thread, SR_of_thread, LP_of_thread and
    // PT_of_thread (for _on_disk)
    std::vector<rlz_spill_vector> CPL_files;
    std::vector<rlz_spill_vector> SR_files;
    std::vector<rlz_spill_vector> LP_files;
    std::vector<rlz_spill_vector> PT_files;

    // [0..p] phrases_before[i_p] = number of phrases starting before range_bounds[i_p]
    std::vector<uint64_t> phrases_before;
    // [0..p] literals_before[i_p] = number of literal phrases starting before range_bounds[i_p]
    std::vector<uint64_t> literals_before;
    // [0..p] copies_before[i_p] = number of copy phrases starting before range_bounds[i_p]
    std::vector<uint64_t> copies_before;

    // ############################# INTERNAL METHODS #############################

    /**
     * @brief returns the value stored for SA^d[0] = SA[0]; like the differences, it is stored with an
     *        offset of n (see decode_context_t::add_difference)
     * @param value SA[0]
     */
    inline uint64_t first_value(uint64_t value) const { return value + n; }

    // logs the peak memory usage until now
    void log_peak_mem_usage()
    {
        log_peak_memory_usage("peak memory usage until now", overall_peak() - baseline_mem_usage, n);
    }

    // logs the end of the current build phase and writes its duration with the key key to the measurement file
    void finish_phase(const std::string& key)
    {
        log_phase_end(log, time, measurement_file, key + measurement_key_suffix);
    }

    /**
     * @brief returns the length of the x-th copy phrase in a thread's range of SA^d
     * @tparam space _on_disk <=> the phrases are read from the temporary files
     * @param thread [0..p-1] thread index
     * @param x index of the copy phrase in the thread's range
     */
    template <rlz_space_t space>
    inline uint64_t copy_length(uint64_t thread, uint64_t x)
    {
        if constexpr (space == _on_disk) return CPL_files[thread][x];
        else return CPL_of_thread[thread][x];
    }

    // returns the starting position in R of the x-th copy phrase in a thread's range of SA^d (see copy_length)
    template <rlz_space_t space>
    inline uint64_t copy_source(uint64_t thread, uint64_t x)
    {
        if constexpr (space == _on_disk) return SR_files[thread][x];
        else return SR_of_thread[thread][x];
    }

    // returns the x-th literal phrase in a thread's range of SA^d (see copy_length)
    template <rlz_space_t space>
    inline uint64_t literal_value(uint64_t thread, uint64_t x)
    {
        if constexpr (space == _on_disk) return LP_files[thread][x];
        else return LP_of_thread[thread][x];
    }

    // returns whether the x-th phrase in a thread's range of SA^d is literal (see copy_length)
    template <rlz_space_t space>
    inline bool is_literal(uint64_t thread, uint64_t x)
    {
        if constexpr (space == _on_disk) return PT_files[thread][x];
        else return PT_of_thread[thread][x];
    }

    /**
     * @brief reads the values of the segments of a reference from SA^d
     * @param segments the segments of SA^d that the reference consists of, ascending
     * @param prev_segments the segments of SA^d of a reference that has already been read, ascending; the
     *        values they contain are copied from prev_values
     * @param prev_values the values of the segments in prev_segments, concatenated
     * @return the values of the segments, concatenated in the order of the segments
     */
    interleaved_bit_aligned_vectors read_segments(const std::vector<rlz_segment_t>& segments,
        const std::vector<rlz_segment_t>& prev_segments, const interleaved_bit_aligned_vectors& prev_values);

    /**
     * @brief estimates the size in bytes of an rlz of SA^d
     * @param n length of the input
     * @param reference_length |R|
     * @param z number of phrases
     * @param z_l number of literal phrases
     */
    static double estimate_bytes(uint64_t n, uint64_t reference_length, uint64_t z, uint64_t z_l);

    /**
     * @brief builds the suffix array of a tried reference and estimates the size of the rlz built with it by
     *        parsing the sample
     * @param tried the reference; its suffix array and the estimate are stored in it
     * @param sample values of SA^d
     * @param windows the windows of SA^d that the sample consists of
     */
    void estimate_size(tried_t& tried, const no_init_vector<uint64_t>& sample,
        const std::vector<sampled_window_t>& windows);

    /**
     * @brief parses SA^d[beg..end-1] greedily into longest matches in R and reports each phrase, except the
     *        first one: SA^d[beg] is a literal phrase
     * @param thread [0..p-1] index of the calling thread
     * @param beg starting position of the range
     * @param end end position of the range (exclusive)
     * @param lookahead buffer of the values read after the start of the current phrase
     * @param report function called as report(i, literal, length, value) for each phrase starting at
     *        position i > beg, where value is SA^d[i] for a literal phrase and the starting position in R of a
     *        copy phrase
     * @param prog progress of the parse (nullptr if none is shown)
     */
    template <typename lookahead_t, typename report_t>
    void parse_range(uint64_t thread, uint64_t beg, uint64_t end, std::vector<lookahead_t>& lookahead,
        report_t&& report, phase_progress* prog = nullptr);

    /**
     * @brief parses blocks of SA^d with R
     * @param ref_segments the segments of SA^d that R consists of
     * @param blocks the blocks to parse, ascending
     * @return statistics on the phrases starting in the blocks
     */
    rlz_parse_stats<uint16_t> parse_sample(const rlz_reference_segments& ref_segments,
        std::vector<uint64_t> blocks);

    /**
     * @brief computes a revised reference from the statistics of a parse with R (see rlz_revision::revise)
     * @param ref_segments the segments of SA^d that R consists of
     * @param stats statistics of the parse
     */
    rlz_revised_reference revise(const rlz_reference_segments& ref_segments,
        const rlz_parse_stats<uint16_t>& stats);

    /**
     * @brief replaces R with a revised reference and builds its suffix array
     * @param ref_segments the segments of SA^d that R consists of; replaced with revised
     * @param revised the segments of SA^d that the revised reference consists of
     */
    void apply_revision(rlz_reference_segments& ref_segments, rlz_reference_segments&& revised);

    // ############################# CONSTRUCTION PHASES #############################

    // computes the frequency of each value in SA^d
    void build_value_frequencies();

    /**
     * @brief chooses the length of R
     *
     * For each tried length, a sample of SA^d is parsed with the reference of that length (the segments
     * chosen first, see rlz_reference_choice::segments_for_target) and the size of the resulting rlz is
     * estimated (see rlz_search_target_length). Afterwards, reference, reference_segments and matcher store
     * the reference of the chosen length.
     */
    void tune_reference();

    /**
     * @brief revises the tuned reference in up to revision_params.sampled_rounds rounds
     *
     * In each round, a random sample of the blocks of SA^d is parsed with R (see parse_sample) and R is
     * revised (see rlz_revision). The rounds end if a round is expected to save less than
     * revision_params.min_gain of the size of the rlz.
     */
    void revise_reference();

    // chooses the segments of R, reads R and ranks its values (see rlz_reference_matcher::rank_values)
    void build_reference();

    // stores R to disk
    void store_reference();

    // builds the suffix array of R
    void build_reference_index();

    // loads R from disk
    void load_reference();

    /**
     * @brief parses SA^d and builds the rlz from the phrases; if R is tuned and revised, R is revised by the
     *        parse and SA^d is parsed again (at most revision_params.whole_rounds parses)
     * @tparam space _on_disk <=> the phrases of each thread are stored in files
     */
    template <rlz_space_t space>
    void build_parsing();

public:
    /**
     * @param built the rlz to build
     * @param value_at function returning SA^d[i] (callable as value_at(thread, i))
     * @param I_Phi_inv the disjoint interval sequence of Phi^{-1}
     * @param n the length of T
     * @param r the number of runs in L
     * @param params optional construction parameters
     */
    construction(
        rlz& built,
        value_func_t value_at,
        intervals_t& I_Phi_inv,
        uint64_t n,
        uint64_t r,
        rlz_construction_params params = {})
        : built(built)
        , value_at(std::move(value_at))
        , I_Phi_inv(I_Phi_inv)
        , n(n)
        , r(r)
        , p(std::clamp<uint64_t>(n / min_values_per_thread, 1, std::max<uint64_t>(1, params.num_threads)))
        , space(params.space)
        , log(params.log)
        , measurement_file(params.measurement_file)
        , measurement_key_suffix(params.measurement_key_suffix)
        , tmp_files_beside(params.tmp_files_beside)
        , given_reference_target_length(params.reference_target_length)
        , tune_reference_length(params.tune_reference_length)
        , revise_tuned_reference(params.revise_reference)
        , revision(n, p, revision_params, rlz_default_seed)
    {
        prefix_tmp_files = temporary_file_path("rlz_", tmp_files_beside);
        baseline_mem_usage = malloc_count_current();
    }

    // builds the rlz
    void build();
};

#include <rlz_ref/construction/construction.tpp>

#endif // RLZ_REF_RLZ_REF_CONSTRUCTION_CONSTRUCTION_HPP

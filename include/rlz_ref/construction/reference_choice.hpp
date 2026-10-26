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

#ifndef RLZ_REF_RLZ_REF_CONSTRUCTION_REFERENCE_CHOICE_HPP
#define RLZ_REF_RLZ_REF_CONSTRUCTION_REFERENCE_CHOICE_HPP

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <gtl/btree.hpp>
#include <libsais40_types.hpp>
#include <misc/log.hpp>
#include <misc/memory.hpp>

// seed of the random number generators of the construction (candidate segments and samples)
inline constexpr uint64_t rlz_default_seed = 918273645;

// returns a value in [lo, hi] computed from the next output of random (independent of the standard library)
inline uint64_t rlz_draw(std::mt19937_64& random, uint64_t lo, uint64_t hi)
{
    return lo + uint64_t(((unsigned __int128) random() * (hi - lo + 1)) >> 64);
}

// the largest length of R; its suffix array is built with libsais40 (see rlz_reference_matcher)
inline constexpr uint64_t rlz_max_reference_length = SA_INT40_MAX_N;

/**
 * @brief returns the default number of candidate segments drawn per round, 5 * (n/r)^0.45
 * @param n_over_r n/r, the average length of a BWT run
 */
inline uint64_t rlz_default_num_candidates(double n_over_r)
{
    return std::max<uint64_t>(1, uint64_t(5 * std::pow(n_over_r, 0.45)));
}

// a segment [beg, end) of the input
struct rlz_segment_t {
    uint64_t beg;
    uint64_t end;

    uint64_t length() const { return end - beg; }
};

// the segments of the input that R consists of, with their starting positions in R
struct rlz_reference_segments {
    // the segments, ascending by their starting positions; no two segments overlap or are adjacent
    std::vector<rlz_segment_t> segments;
    /* the starting positions of the segments in R: [0..segments.size()] begins[i] is the starting position
       of segments[i] in R; begins[segments.size()] = |R|; empty until compute_begins() is called */
    std::vector<uint64_t> begins;

    // inserts segments of the input and merges overlapping and adjacent segments
    void insert(std::vector<rlz_segment_t> added);

    // computes begins
    void compute_begins()
    {
        begins.assign(segments.size() + 1, 0);
        for (uint64_t i = 0; i < segments.size(); i++) begins[i + 1] = begins[i] + segments[i].length();
    }

    // returns |R| (0 if compute_begins() has not been called)
    uint64_t length() const { return begins.empty() ? 0 : begins.back(); }

    // returns the index of the segment containing position pos of R
    uint64_t segment_of(uint64_t pos) const
    {
        return std::upper_bound(begins.begin(), begins.end(), pos) - begins.begin() - 1;
    }

    // returns whether position pos of the input lies in one of the segments
    bool contains(uint64_t pos) const
    {
        auto it = std::upper_bound(segments.begin(), segments.end(), pos,
            [](uint64_t position, const rlz_segment_t& segment) { return position < segment.end; });
        return it != segments.end() && it->beg <= pos;
    }
};

/**
 * @brief marks the positions of R that a copy phrase copies; after removing all other positions from R, a
 *        copied position pos is at position num_copied_before(pos)
 */
class rlz_copied_positions {
protected:
    /* marks the copied positions of R: [0..ceil(|R|/64)] bit pos % 64 of copied[pos / 64] is set <=>
       position pos of R is copied; the last word is padding */
    std::vector<uint64_t> copied;
    /* the number of copied positions before each word: [0..copied.size()] copied_before[w] = number of
       copied positions of R before position 64 * w; filled by build_rank() */
    std::vector<uint64_t> copied_before;

public:
    // reference_length is |R|
    explicit rlz_copied_positions(uint64_t reference_length)
        : copied((reference_length + 63) / 64 + 1, 0)
    { }

    /**
     * @brief marks R[source..source+length-1] as copied; must not be called by several threads at the
     *        same time
     * @param source [0..|R|-1] starting position in R of a copy phrase
     * @param length length of the copy phrase (at least 1)
     */
    inline void mark_copied(uint64_t source, uint64_t length);

    /**
     * @brief builds copied_before; has to be called after the last call of mark_copied()
     * @return number of copied positions of R
     */
    uint64_t build_rank();

    // returns whether position pos of R is copied
    inline bool is_copied(uint64_t pos) const { return (copied[pos / 64] >> (pos % 64)) & 1; }

    // returns the number of copied positions of R before position pos; build_rank() has to be called first
    inline uint64_t num_copied_before(uint64_t pos) const
    {
        return copied_before[pos / 64] + std::popcount(copied[pos / 64] & ((uint64_t(1) << (pos % 64)) - 1));
    }
};

// parameters for choosing the segments of R (see rlz_reference_choice)
struct rlz_reference_choice_params {
    uint64_t target_length = 0; // maximum length of R
    uint64_t segment_length = 3072; // maximum length of a segment
    uint64_t num_candidates = 1; // maximum number of candidate segments drawn and scored per round
    /* number of candidates per batch; a round ends after the first batch that does not improve the best
       score of the round; 0 <=> each round is one batch of num_candidates candidates */
    uint64_t candidates_per_batch = 0;
    /* the choice ends with the first round whose best score is below min_score_share times the best
       score of the first round; 0 <=> no such limit */
    double min_score_share = 0.0;
    uint64_t num_threads = 1; // number of threads that score the candidates of a batch
    uint64_t seed = rlz_default_seed; // seed of the random number generator the candidates are drawn with
    /* true <=> if no candidate of a round has a positive score, a random candidate of its last batch is
       chosen; false <=> the choice ends there */
    bool choose_random_if_unscored = false;
};

/**
 * @brief chooses the segments of the input that R consists of
 *
 * 1. In each round, up to num_candidates segments of length segment_length are drawn at random, clipped
 *    to the positions that are not chosen yet, and scored; the candidate with the highest score is chosen.
 *    The rounds end when the total length of the chosen segments reaches choose_share * target_length.
 * 2. The gaps between consecutive chosen segments are closed in descending order of their score (see
 *    gap_t), as long as the total length of the segments does not exceed target_length.
 *
 * @tparam scorer_t type of the scorer; it has to provide
 *         float score(uint64_t thread, uint64_t beg, uint64_t end): returns the score per position of the
 *         segment [beg, end), counting only content that R does not contain yet; called by several threads
 *         at the same time
 *         void cover(uint64_t thread, uint64_t beg, uint64_t end): marks the content of the segment
 *         [beg, end) as contained in R; called by one thread at a time
 */
template <typename scorer_t>
class rlz_reference_choice {
public:
    // share of target_length that is chosen in rounds; the rest is used to close gaps
    static constexpr double choose_share = 0.95;
    // maximum number of draws for one candidate whose draws lie within chosen segments
    static constexpr uint64_t max_draws_per_candidate = 8;

protected:
    // orders segments ascending by their starting positions
    struct by_beg {
        bool operator()(const rlz_segment_t& a, const rlz_segment_t& b) const { return a.beg < b.beg; }
    };

    using segments_t = gtl::btree_set<rlz_segment_t, by_beg>;
    using segment_it_t = typename segments_t::iterator;

    // gap between two consecutive chosen segments
    struct gap_t {
        uint64_t beg_before; // starting position of the segment before the gap
        float score; // length of the segment resulting from closing the gap divided by the length of the gap
    };

    // orders gaps descending by their scores
    struct by_score {
        bool operator()(const gap_t& a, const gap_t& b) const
        {
            return a.score > b.score || (a.score == b.score && a.beg_before < b.beg_before);
        }
    };

    uint64_t n;                    // length of the input
    scorer_t& scorer;              // scorer of the segments
    uint64_t target_length;        // maximum length of R
    uint64_t segment_length;       // maximum length of a segment
    uint64_t num_candidates;       // maximum number of candidates drawn per round
    uint64_t candidates_per_batch; // see rlz_reference_choice_params
    uint64_t num_threads;          // number of threads that score the candidates of a batch
    double min_score_share;        // see rlz_reference_choice_params
    bool choose_random_if_unscored; // see rlz_reference_choice_params
    std::mt19937_64 random;        // random number generator the candidates are drawn with

    // the chosen segments, ascending by their starting positions
    segments_t chosen_by_beg;
    // the segments chosen by choose_segments(), in the order they were chosen (see segments_for_target)
    std::vector<rlz_segment_t> chosen_in_order;
    // the gaps between the chosen segments, descending by their scores (see close_gaps)
    gtl::btree_set<gap_t, by_score> gaps;
    uint64_t total_chosen_length = 0; // total length of the chosen segments
    float first_score = 0; // best score of the first round (see rlz_reference_choice_params::min_score_share)

    /**
     * @brief clips the candidate segment [beg, end) such that it does not overlap a chosen segment
     * @param beg starting position of the candidate (adjusted)
     * @param end end position of the candidate (adjusted); beg == end <=> the candidate lies within a
     *        chosen segment
     * @return iterator to the first chosen segment starting at or after beg
     */
    segment_it_t clip_to_free_space(uint64_t& beg, uint64_t& end);

    // inserts the segment [beg, end) into chosen_by_beg and merges it with adjacent segments; next is the
    // first chosen segment starting at or after beg
    void insert_chosen(uint64_t beg, uint64_t end, segment_it_t next);

    // returns the score (end_after - beg_before) / gap_length of the gap between two segments, where
    // beg_before is the starting position of the first and end_after the end position of the second segment
    static float gap_score(uint64_t beg_before, uint64_t end_after, uint64_t gap_length);

public:
    /**
     * @param n length of the input
     * @param scorer scorer of the segments (see scorer_t)
     * @param params parameters
     */
    rlz_reference_choice(uint64_t n, scorer_t& scorer, rlz_reference_choice_params params)
        : n(n)
        , scorer(scorer)
        , target_length(std::min<uint64_t>(params.target_length, rlz_max_reference_length))
        , segment_length(std::max<uint64_t>(1, params.segment_length))
        , num_candidates(std::max<uint64_t>(1, params.num_candidates))
        , candidates_per_batch(params.candidates_per_batch)
        , num_threads(std::max<uint64_t>(1, params.num_threads))
        , min_score_share(params.min_score_share)
        , choose_random_if_unscored(params.choose_random_if_unscored)
        , random(params.seed)
    { }

    // returns the chosen segments, ascending by their starting positions
    std::vector<rlz_segment_t> segments() const
    {
        return std::vector<rlz_segment_t>(chosen_by_beg.begin(), chosen_by_beg.end());
    }

    // returns the number of chosen segments
    uint64_t num_segments() const { return chosen_by_beg.size(); }

    // returns the total length of the chosen segments
    uint64_t chosen_length() const { return total_chosen_length; }

    /**
     * @brief chooses the segments (choose_segments()) and closes the gaps between them (close_gaps())
     * @param log whether the progress of choosing the segments is printed
     * @return the chosen segments, ascending by their starting positions
     */
    std::vector<rlz_segment_t> choose(bool log = false)
    {
        choose_segments(log);
        close_gaps();
        return segments();
    }

    /**
     * @brief draws and scores candidate segments and chooses the best one of each round, until the total
     *        length of the chosen segments reaches choose_share * target_length; continues after the rounds
     *        of the previous calls
     * @param log whether the progress (the total length of the chosen segments) is printed
     */
    void choose_segments(bool log = false);

    // closes the gaps between the chosen segments in descending order of their score
    void close_gaps();

    /**
     * @brief raises target_length to target and continues choosing segments (see choose_segments)
     * @param target maximum length of R
     */
    void extend_to(uint64_t target)
    {
        target = std::min(target, rlz_max_reference_length);
        if (target > target_length) target_length = target;
        choose_segments();
    }

    /**
     * @brief returns the segments of a reference of maximum length target: the segments chosen first, up
     *        to a total length of choose_share * target, with the gaps between them closed; does not change
     *        the chosen segments
     * @param target maximum length of R
     * @return the segments, ascending by their starting positions
     */
    std::vector<rlz_segment_t> segments_for_target(uint64_t target);
};

#include <rlz_ref/construction/reference_choice.tpp>

#endif // RLZ_REF_RLZ_REF_CONSTRUCTION_REFERENCE_CHOICE_HPP

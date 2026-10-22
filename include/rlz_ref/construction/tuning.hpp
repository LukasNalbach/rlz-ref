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

#ifndef RLZ_REF_RLZ_REF_CONSTRUCTION_TUNING_HPP
#define RLZ_REF_RLZ_REF_CONSTRUCTION_TUNING_HPP

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <rlz_ref/construction/matcher.hpp>
#include <rlz_ref/construction/reference_choice.hpp>

// length of a window of the sample of the input the tuning parses; a sampled window or block is parsed from
// rlz_sample_window_length symbols before its start, and only the phrases starting in it are counted
inline constexpr uint64_t rlz_sample_window_length = 4096;

// estimated size of an rlz built with a reference of a certain length
struct rlz_size_estimate_t {
    double bytes = std::numeric_limits<double>::max(); // size of the rlz in bytes
    double reference_bytes = 0; // size of R in bytes
    double phrases = 0; // number of phrases
};

// returns the estimated size in bytes of the phrase starts (Elias-Fano coded) of z phrases of an input of
// length n (see rlz_phrase_starts)
inline double rlz_estimate_phrase_starts_bytes(uint64_t n, uint64_t z)
{
    return z * (2.0 + std::log2(std::max<double>(2.0, n / double(z)))) / 8.0;
}

// returns the estimated size in bytes of PT with rank support (1.25 bits per phrase) for z phrases
inline double rlz_estimate_PT_bytes(uint64_t z) { return z * 1.25 / 8.0; }

// returns the size in bytes of SR for z_c copy phrases and a reference of length reference_length
inline double rlz_estimate_SR_bytes(uint64_t z_c, uint64_t reference_length)
{
    return z_c * double(std::bit_width(std::max<uint64_t>(1, reference_length))) / 8.0;
}

// returns whether R occupies less than 5 % of the size of the rlz
inline bool rlz_reference_is_small(const rlz_size_estimate_t& estimate)
{
    return estimate.reference_bytes < 0.05 * estimate.bytes;
}

/**
 * @brief returns whether an rlz built with a reference of twice the length is expected to be smaller
 *
 * Returns false if 1.8 times the size of R is at least min(smallest, last.bytes). Else, the number of
 * phrases is assumed to decrease by the same factor as with the last doubling; returns true if the phrases
 * saved this way occupy more than 1.1 times the size of R.
 * @param before estimate for the length before the last one (nullptr if there is none)
 * @param last estimate for the last length
 * @param smallest size in bytes of the smallest rlz estimated so far
 */
inline bool rlz_should_double_reference(const rlz_size_estimate_t* before, const rlz_size_estimate_t& last,
    double smallest)
{
    static constexpr double reference_growth = 1.8;
    static constexpr double min_margin = 0.1;

    if (reference_growth * last.reference_bytes >= std::min<double>(smallest, last.bytes)) return false;
    if (before == nullptr || before->phrases <= 0) return true;

    const double factor = std::min<double>(1.0, last.phrases / before->phrases);
    const double saved = (last.bytes - last.reference_bytes) * (1.0 - factor);
    return saved > (1.0 + min_margin) * last.reference_bytes;
}

// returns the first maximum length of R the tuning tries, max(4096, n / 4096), at most cap
inline uint64_t rlz_first_target_length(uint64_t n, uint64_t cap)
{
    return std::min<uint64_t>(cap, std::max<uint64_t>(4096, n / 4096));
}

/**
 * @brief draws non-overlapping windows of the input at random positions
 * @param n length of the input
 * @param num_windows number of drawn windows; a window overlapping the previous one is skipped
 * @return the windows [first, second) of length min(rlz_sample_window_length, n), ascending
 */
inline std::vector<std::pair<uint64_t, uint64_t>> rlz_draw_sample_windows(uint64_t n, uint64_t num_windows)
{
    const uint64_t window_length = std::min<uint64_t>(rlz_sample_window_length, n);
    std::mt19937_64 random(rlz_default_seed);
    std::vector<uint64_t> begs;
    begs.reserve(num_windows);
    for (uint64_t i = 0; i < num_windows; i++) begs.emplace_back(rlz_draw(random, 0, n - window_length));
    std::sort(begs.begin(), begs.end());
    std::vector<std::pair<uint64_t, uint64_t>> windows;

    for (uint64_t beg : begs) {
        if (!windows.empty() && beg < windows.back().second) continue;
        windows.emplace_back(beg, beg + window_length);
    }

    return windows;
}

/**
 * @brief a reference of one maximum length tried by the tuning
 * @tparam reference_t type of R
 * @tparam matcher_t type of the suffix array of R (rlz_reference_matcher or a class derived from it)
 */
template <typename reference_t, typename matcher_t = rlz_reference_matcher>
struct rlz_tried_reference_t {
    uint64_t target_length = 0; // maximum length of R
    uint64_t segment_length = 0; // maximum length of a segment
    rlz_reference_segments ref_segments; // the segments of the input that R consists of
    reference_t reference; // R
    matcher_t matcher; // suffix array of R
    rlz_size_estimate_t estimate; // estimated size of the rlz built with R

    // returns |R|
    uint64_t length() const { return reference.size(); }
};

/**
 * @brief searches the maximum length of R with the smallest estimated rlz
 *
 * 1. Tries first, 2 * first, 4 * first, ... (at most cap): the doubling continues while the estimate
 *    decreases (or R is small, see rlz_reference_is_small) and rlz_should_double_reference() returns true.
 * 2. If first yields the smallest estimate, tries first / 2, first / 4, ... (at least min_length), while the
 *    estimate decreases.
 * @param best the tried reference with the smallest estimate (initially empty)
 * @param next_smaller if not nullptr, the tried reference of the next smaller length before best (empty if
 *        there is none)
 * @param first first maximum length of R to try
 * @param cap largest maximum length of R to try
 * @param min_length smallest maximum length of R to try in step 2
 * @param search false <=> only first is tried
 * @param smallest_before size in bytes of the smallest rlz estimated by earlier searches
 * @param try_target try_target(target_length) returns the tried reference of maximum length target_length;
 *        its length is 0 if no segment is chosen
 */
template <typename tried_t, typename try_target_t>
void rlz_search_target_length(tried_t& best, tried_t* next_smaller, uint64_t first, uint64_t cap,
    uint64_t min_length, bool search, double smallest_before, try_target_t&& try_target)
{
    // a length with a small R is accepted, if its estimate exceeds the smallest one by at most this factor
    static constexpr double small_reference_tolerance = 1.01;

    rlz_size_estimate_t before; // estimate for the previously tried length
    bool has_before = false;
    uint64_t length_before = 0; // |R| for the previously tried length

    // 1. double the length of R
    for (uint64_t target_length = first;; target_length = std::min<uint64_t>(2 * target_length, cap)) {
        tried_t tried = try_target(target_length);

        // stop if the choice did not add a segment
        if (tried.length() == 0 || tried.length() == length_before) break;
        length_before = tried.length();
        const rlz_size_estimate_t estimate = tried.estimate;
        const bool small = rlz_reference_is_small(estimate);

        if (estimate.bytes < best.estimate.bytes) {
            if (next_smaller != nullptr) *next_smaller = std::move(best);
            best = std::move(tried);
        } else if (!small || estimate.bytes > small_reference_tolerance * best.estimate.bytes) {
            break; // the estimated size of the rlz did not decrease
        }

        if (!search || target_length >= cap) break;

        if (!small && !rlz_should_double_reference(has_before ? &before : nullptr, estimate,
            std::min<double>(best.estimate.bytes, smallest_before))) break;

        before = estimate;
        has_before = true;
    }

    // 2. if the first length yields the smallest rlz, halve the length while the rlz gets smaller
    if (search && best.target_length == first && best.length() != 0 && !rlz_reference_is_small(best.estimate)) {
        for (uint64_t target_length = first / 2; target_length >= min_length; target_length /= 2) {
            tried_t tried = try_target(target_length);
            if (tried.length() == 0 || tried.length() == best.length()) break;
            if (tried.estimate.bytes >= best.estimate.bytes) break;
            if (next_smaller != nullptr) *next_smaller = tried_t {};
            best = std::move(tried);
        }
    }
}

#endif // RLZ_REF_RLZ_REF_CONSTRUCTION_TUNING_HPP

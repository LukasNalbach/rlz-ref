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

#ifndef RLZ_REF_RLZ_REF_CONSTRUCTION_CONSTRUCTION_TPP
#define RLZ_REF_RLZ_REF_CONSTRUCTION_CONSTRUCTION_TPP

#include <rlz_ref/construction/construction.hpp>

template <typename value_func_t>
float rlz_value_frequency_scorer<value_func_t>::score(uint64_t thread, uint64_t beg, uint64_t end)
{
    auto& counted = seen[thread].data;
    float total = 0;

    for (uint64_t i = beg; i < end; i++) {
        uint64_t value = value_at(thread, i);
        uint64_t occurrences = (*frequency.find(value)).second;

        // count each distinct value that R does not contain once
        if (occurrences != 0 && counted.emplace(value).second) total += std::sqrt(occurrences);
    }

    counted.clear();
    return total / (end - beg);
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
void rlz<kind>::construction<value_func_t, intervals_t>::build()
{
    // the maximum length of R is min(n / 3, 5.2 * r), if it is not given or tuned
    reference_target_length = given_reference_target_length != 0 ? given_reference_target_length
        : std::min<uint64_t>(std::max<uint64_t>(1, n / max_reference_divisor), 5.2 * r);
    segment_length = std::min<uint64_t>(rlz_reference_choice_params {}.segment_length, reference_target_length);
    num_candidates = rlz_default_num_candidates(n / (float) r);

    if (rlz::value_width(n) > 64)
        throw std::runtime_error("the input is too long to encode its differential suffix array");

    const bool spill = space == _on_disk;
    build_value_frequencies();
    if (given_reference_target_length == 0 && tune_reference_length) tune_reference();
    if (reference_tuned && revise_tuned_reference) revise_reference();

    if (!reference_tuned) {
        build_reference();
        if (spill) store_reference();
        build_reference_index();
        if (spill) load_reference();
    }

    if (spill) build_parsing<_on_disk>();
    else build_parsing<_in_memory>();
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
void rlz<kind>::construction<value_func_t, intervals_t>::build_value_frequencies()
{
    log_message(log, "building the RLZ encoding:\n");
    log_phase_start(log, time, "computing value frequencies");

    value_frequency.reserve(r);

    // the i-th pair of I_Phi^{-1} yields a value of SA^d and its number of occurrences in the i-th interval
    for (uint64_t i = 0; i < r; i++) {
        uint64_t value = (I_Phi_inv.template get<1>(i) + n) - I_Phi_inv.template get<0>(i);
        uint64_t occurrences = (i == r - 1 ? n : I_Phi_inv.template get<0>(i + 1)) - I_Phi_inv.template get<0>(i);
        auto it = value_frequency.find(value);

        if (it == value_frequency.end()) {
            value_frequency.insert_unique(value, occurrences);
        } else {
            (*it).second += occurrences;
        }
    }

    I_Phi_inv.clear();

    finish_phase("time_build_rlz_value_frequencies");
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
interleaved_bit_aligned_vectors rlz<kind>::construction<value_func_t, intervals_t>::read_segments(
    const std::vector<rlz_segment_t>& segments, const std::vector<rlz_segment_t>& prev_segments,
    const interleaved_bit_aligned_vectors& prev_values)
{
    // starting positions of the segments in the reference (the last entry is its length)
    std::vector<uint64_t> begins(segments.size() + 1, 0);
    std::vector<uint64_t> prev_begins(prev_segments.size() + 1, 0);

    for (uint64_t i = 0; i < segments.size(); i++) begins[i + 1] = begins[i] + segments[i].length();
    for (uint64_t i = 0; i < prev_segments.size(); i++) prev_begins[i + 1] = prev_begins[i] + prev_segments[i].length();

    interleaved_bit_aligned_vectors values({ rlz::value_width(n) });
    values.resize_no_init(begins.back());

    // each segment is read sequentially by one thread
    #pragma omp parallel for num_threads(p) schedule(dynamic, 1)
    for (uint64_t s = 0; s < segments.size(); s++) {
        const uint64_t thread = omp_get_thread_num();
        uint64_t pos = segments[s].beg;

        // the first previous segment ending after pos
        uint64_t h = std::upper_bound(prev_segments.begin(), prev_segments.end(), pos,
            [](uint64_t position, const rlz_segment_t& segment) { return position < segment.end; })
            - prev_segments.begin();

        while (pos < segments[s].end) {
            const bool in_prev = h < prev_segments.size() && prev_segments[h].beg <= pos;
            // [pos, until) lies within the previous segment h (copied from it) or before it (read from SA^d)
            const uint64_t until = std::min<uint64_t>(segments[s].end, in_prev ? prev_segments[h].end
                : (h < prev_segments.size() ? prev_segments[h].beg : segments[s].end));

            for (uint64_t i = pos; i < until; i++) {
                values.template set_parallel<0, uint64_t>(begins[s] + (i - segments[s].beg), in_prev
                    ? uint64_t(prev_values[prev_begins[h] + (i - prev_segments[h].beg)])
                    : value_at(thread, i));
            }

            pos = until;
            if (in_prev) h++;
        }
    }

    return values;
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
double rlz<kind>::construction<value_func_t, intervals_t>::estimate_bytes(uint64_t n, uint64_t reference_length,
    uint64_t z, uint64_t z_l)
{
    const double reference_bytes = double(reference_length) * rlz::value_width(n) / 8.0;
    if (z == 0) return reference_bytes;
    const double literals_bytes = z_l * double(rlz::value_width(n)) / 8.0;

    return reference_bytes + rlz_estimate_PT_bytes(z) + rlz_estimate_phrase_starts_bytes(n, z)
        + rlz_estimate_SR_bytes(z - z_l, reference_length) + literals_bytes;
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
void rlz<kind>::construction<value_func_t, intervals_t>::estimate_size(tried_t& tried,
    const no_init_vector<uint64_t>& sample, const std::vector<sampled_window_t>& windows)
{
    tried.matcher.build_over_values([&](uint64_t i) { return uint64_t(tried.reference[i]); },
        tried.length(), p);

    // parse the sample (see build_parsing)
    uint64_t z = 0;
    uint64_t z_l = 0;

    #pragma omp parallel for num_threads(p) reduction(+: z, z_l) schedule(dynamic)
    for (uint64_t w = 0; w < windows.size(); w++) {
        const auto [beg, counted_from, end] = windows[w];

        // the first value is a literal phrase
        if (counted_from == beg) {
            z++;
            z_l++;
        }

        for (uint64_t i = beg + 1; i < end;) {
            const uint64_t max_length = std::min<uint64_t>(end - i, max_copy_length);
            auto read = [&](uint64_t k) { return sample[i + k]; };

            const uint64_t length = tried.matcher.longest_match(
                rlz_value_sequence<interleaved_bit_aligned_vectors, decltype(read)> { tried.reference, read },
                max_length).first;

            // only the phrases starting in the window are counted
            const bool counted = i >= counted_from;
            z += counted ? 1 : 0;

            if (length <= 1) {
                z_l += counted ? 1 : 0;
                i++;
            } else {
                i += length;
            }
        }
    }

    uint64_t counted_values = 0;
    for (const sampled_window_t& window : windows) counted_values += window.end - window.counted_from;
    const double scale = n / double(std::max<uint64_t>(1, counted_values));
    tried.estimate.phrases = z * scale;
    tried.estimate.reference_bytes = double(tried.length()) * rlz::value_width(n) / 8.0;
    tried.estimate.bytes = estimate_bytes(n, tried.length(), uint64_t(z * scale), uint64_t(z_l * scale));

    if (log) {
        log_out() << "  tuning: s = " << format_number(tried.target_length) << " ("
                  << format_ratio(tried.length() / double(r)) << " r), |R| = " << format_number(tried.length())
                  << ", ~z = " << format_number(uint64_t(z * scale)) << ", ~size = "
                  << format_size(uint64_t(tried.estimate.bytes)) << std::endl;
    }
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
void rlz<kind>::construction<value_func_t, intervals_t>::tune_reference()
{
    log_phase_start(log, time, "tuning R\n");

    // the sample: windows of SA^d at random positions (about n / 100 values in total, at least 1024 and at
    // most 2^22); each window is read and parsed from one window length before its start
    const uint64_t sampled = std::clamp<uint64_t>(n / 100, 1024, uint64_t(4) << 20);
    const uint64_t window_length = std::min<uint64_t>(rlz_sample_window_length, n);
    const uint64_t num_windows = std::max<uint64_t>(1, sampled / (2 * window_length));
    std::vector<uint64_t> read_begs; // starting positions in SA^d of the values read for each window
    std::vector<sampled_window_t> windows;
    uint64_t sample_length = 0;

    for (const auto& [beg, end] : rlz_draw_sample_windows(n, num_windows)) {
        const uint64_t read_beg = beg - std::min<uint64_t>(beg, window_length);
        read_begs.emplace_back(read_beg);
        windows.emplace_back(sampled_window_t { sample_length, sample_length + (beg - read_beg),
            sample_length + (end - read_beg) });
        sample_length += end - read_beg;
    }

    no_init_vector<uint64_t> sample;
    resize_no_init(sample, sample_length);

    #pragma omp parallel for num_threads(p) schedule(dynamic, 1)
    for (uint64_t w = 0; w < windows.size(); w++) {
        const uint64_t thread = omp_get_thread_num();

        for (uint64_t i = windows[w].beg; i < windows[w].end; i++)
            sample[i] = value_at(thread, read_begs[w] + (i - windows[w].beg));
    }

    rlz_value_frequency_scorer<value_func_t> scorer(value_at, value_frequency, p);
    rlz_reference_choice choice(n, scorer, rlz_reference_choice_params {
        .target_length = 0,
        .segment_length = segment_length,
        .num_candidates = num_candidates,
        .num_threads = p,
        .choose_random_if_unscored = true });

    tried_t best; // the reference with the smallest estimated rlz
    uint64_t max_tried_target = 0; // the largest maximum length of R tried so far

    // builds the reference of maximum length target_length and estimates the size of the rlz built with
    // it; the values that best contains are copied from it
    auto try_target = [&](uint64_t target_length) {
        // continue the choice up to the largest tried maximum length of R
        if (target_length > max_tried_target) {
            choice.extend_to(target_length);
            max_tried_target = target_length;
        }

        tried_t tried;
        tried.target_length = target_length;
        tried.ref_segments.segments = choice.segments_for_target(target_length);
        if (tried.ref_segments.segments.empty()) return tried;
        tried.reference = read_segments(tried.ref_segments.segments, best.ref_segments.segments, best.reference);
        estimate_size(tried, sample, windows);
        return tried;
    };

    // the tuning tries maximum lengths of R up to min(n / 3, 64 * r)
    const uint64_t cap = std::min<uint64_t>(std::max<uint64_t>(1, n / max_reference_divisor), 64 * r);
    rlz_search_target_length(best, (tried_t*) nullptr, rlz_first_target_length(n, cap), cap, segment_length,
        true, std::numeric_limits<double>::max(), try_target);

    value_frequency.clear();
    value_frequency.shrink_to_fit();

    if (best.length() != 0) {
        reference_target_length = best.target_length;
        reference_length = best.length();
        reference = std::move(best.reference);
        reference_segments = std::move(best.ref_segments.segments);
        matcher = std::move(best.matcher);
        reference_tuned = true;
    }

    log_message(log, "chose a reference of " + format_number(reference_target_length) + " values ("
        + format_ratio(reference_target_length / double(r)) + " r)");
    finish_phase("time_tune_rlz_reference");
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
template <typename lookahead_t, typename report_t>
void rlz<kind>::construction<value_func_t, intervals_t>::parse_range(uint64_t thread, uint64_t beg, uint64_t end,
    std::vector<lookahead_t>& lookahead, report_t&& report, phase_progress* prog)
{
    // lookahead[0..num_lookahead-1] stores the values SA^d[i..i+num_lookahead-1] that have been read; each
    // value of SA^d is read once, in ascending order of its position
    uint64_t num_lookahead = 0;
    uint64_t counted = beg; // the positions [beg, counted) are counted by prog
    const uint64_t grain = prog != nullptr ? prog->grain() : end - beg; // prog counts at least grain positions at once

    for (uint64_t i = beg + 1; i < end;) {
        if (i - counted >= grain) [[unlikely]] {
            if (prog != nullptr) prog->advance(i - counted);
            counted = i;
        }

        const uint64_t max_length = std::min<uint64_t>(end - i, max_copy_length);

        // returns SA^d[i + k]
        auto read = [&](uint64_t k) {
            while (num_lookahead <= k) {
                if (num_lookahead == lookahead.size()) lookahead.resize(2 * lookahead.size());
                lookahead[num_lookahead] = value_at(thread, i + num_lookahead);
                num_lookahead++;
            }

            return uint64_t(lookahead[k]);
        };

        // find the longest prefix of SA^d[i..i+max_length-1] that occurs in R
        auto [length, source] = matcher.longest_match(
            rlz_value_sequence<interleaved_bit_aligned_vectors, decltype(read)> { reference, read }, max_length);

        if (length <= 1) {
            length = 1;
            report(i, true, length, read(0));
        } else {
            report(i, false, length, source);
        }

        // move the values read after the phrase to the front of lookahead
        std::copy(lookahead.begin() + length, lookahead.begin() + num_lookahead, lookahead.begin());
        num_lookahead -= length;
        i += length;
    }

    if (prog != nullptr) prog->advance(end - counted);
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
auto rlz<kind>::construction<value_func_t, intervals_t>::parse_sample(const rlz_reference_segments& ref_segments,
    std::vector<uint64_t> blocks) -> rlz_parse_stats<uint16_t>
{
    rlz_parse_stats<uint16_t> stats = revision.new_stats<uint16_t>(std::move(blocks));
    rlz_revision::thread_counters_t counters = revision.new_counters(ref_segments.segments.size(), 0);
    uint64_t z = 0;
    uint64_t z_l = 0;

    #pragma omp parallel for num_threads(p) schedule(dynamic, 1) reduction(+: z, z_l)
    for (uint64_t b = 0; b < stats.blocks.size(); b++) {
        const uint64_t thread = omp_get_thread_num();
        const uint64_t counted_from = stats.blocks[b] * revision_block_length;
        const auto [beg, end] = revision.sampled_range(stats.blocks[b]);
        rlz_revision::prev_phrase_t prev;
        std::vector<uint64_t> lookahead(initial_lookahead_length);

        auto count = [&](uint64_t pos, bool literal, uint64_t length, uint64_t value) {
            const bool counted = pos >= counted_from;

            if (counted) {
                stats.phrases[b]++;
                z++;
                z_l += literal ? 1 : 0;
            }

            revision.count_phrase(prev, counters[thread].data, ref_segments, counted, literal, length, value);
        };

        // the first value is a literal phrase
        count(beg, true, 1, 0);
        parse_range(thread, beg, end, lookahead, count);
    }

    stats.z = z;
    stats.z_l = z_l;
    revision.add_counters(stats, counters);
    return stats;
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
rlz_revised_reference rlz<kind>::construction<value_func_t, intervals_t>::revise(const rlz_reference_segments& ref_segments,
    const rlz_parse_stats<uint16_t>& stats)
{
    const double z = stats.z * stats.scale;
    rlz_size_estimate_t estimate;
    estimate.phrases = z;
    estimate.reference_bytes = double(reference_length) * rlz::value_width(n) / 8.0;
    estimate.bytes = estimate_bytes(n, reference_length, uint64_t(z), uint64_t(stats.z_l * stats.scale));

    // [0..p-1] the values of the block a thread computes the fingerprints of
    std::vector<cache_line_padded<std::vector<uint64_t>>> block_values(p);

    // the symbols of the k-mers are hashed values of SA^d (see rlz_karp_rabin)
    return revision.revise(ref_segments, reference_length, stats, estimate,
        [&](uint64_t i) { return hash_64(uint64_t(reference[i])) % mersenne_61::prime; },
        [&](uint64_t thread, uint64_t beg, uint64_t end) {
            std::vector<uint64_t>& values = block_values[thread].data;
            values.resize(end - beg);
            for (uint64_t pos = beg; pos < end; pos++) values[pos - beg] = value_at(thread, pos);
            return [&values](uint64_t j) { return hash_64(values[j]) % mersenne_61::prime; };
        });
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
void rlz<kind>::construction<value_func_t, intervals_t>::apply_revision(rlz_reference_segments& ref_segments,
    rlz_reference_segments&& revised)
{
    // the values that R already contains are copied from it (see read_segments)
    reference = read_segments(revised.segments, ref_segments.segments, reference);
    reference_length = revised.length();
    matcher.build_over_values([&](uint64_t i) { return uint64_t(reference[i]); }, reference_length, p);
    ref_segments = std::move(revised);
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
void rlz<kind>::construction<value_func_t, intervals_t>::revise_reference()
{
    log_phase_start(log, time, "revising R on a sample\n");

    rlz_reference_segments ref_segments;
    ref_segments.segments = std::move(reference_segments);
    ref_segments.compute_begins();

    for (uint64_t round = 0; round < revision_params.sampled_rounds; round++) {
        const rlz_parse_stats<uint16_t> stats = parse_sample(ref_segments, revision.draw_blocks());
        rlz_revised_reference revised = revise(ref_segments, stats);
        log_message(log, rlz_revision::describe(reference_length, stats, revised) + "\n");
        // stop if the round is expected to save less than min_gain of the size of the rlz
        if (revised.gain < revision_params.min_gain * revised.estimated_bytes) break;
        apply_revision(ref_segments, std::move(revised.ref_segments));
    }

    reference_segments = std::move(ref_segments.segments);
    log_message(log, "revised the reference");
    finish_phase("time_revise_rlz_reference");
    if (log) log_out() << "|R| = " << format_number(reference_length) << std::endl;
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
void rlz<kind>::construction<value_func_t, intervals_t>::build_reference()
{
    rlz_value_frequency_scorer<value_func_t> scorer(value_at, value_frequency, p);

    rlz_reference_choice choice(n, scorer, rlz_reference_choice_params {
        .target_length = reference_target_length,
        .segment_length = segment_length,
        .num_candidates = num_candidates,
        .num_threads = p });

    log_phase_start(log, time, "choosing the segments of R");
    choice.choose_segments(log);
    finish_phase("time_choose_rlz_reference_segments");
    log_message(log, "number of segments: " + format_number(choice.num_segments()) + "\n");

    log_phase_start(log, time, "closing the gaps between the segments");
    choice.close_gaps();
    finish_phase("time_close_rlz_reference_gaps");
    log_message(log, "number of segments: " + format_number(choice.num_segments()) + "\n");

    log_phase_start(log, time, "building R");

    value_frequency.clear();
    value_frequency.shrink_to_fit();

    const std::vector<rlz_segment_t> segments = choice.segments();
    reference = read_segments(segments, {}, interleaved_bit_aligned_vectors());
    reference_length = reference.size();

    finish_phase("time_build_rlz_reference");
    log_phase_start(log, time, "ranking the values of R");

    // the suffix array is built from the ranks (see build_reference_index); R can be stored to disk before
    matcher.rank_values([&](uint64_t i) { return uint64_t(reference[i]); }, reference_length, p);

    finish_phase("time_rank_rlz_reference");
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
void rlz<kind>::construction<value_func_t, intervals_t>::store_reference()
{
    log_phase_start(log, time, "storing R");

    direct_ofstream tmp_file(prefix_tmp_files + "_R");
    reference.serialize(tmp_file);
    reference.clear();
    reference.shrink_to_fit();
    tmp_file.close();
    log_spilled_size(log, prefix_tmp_files + "_R");

    finish_phase("time_store_rlz_reference");
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
void rlz<kind>::construction<value_func_t, intervals_t>::build_reference_index()
{
    log_phase_start(log, time, "building SA of R");
    matcher.build_over_ranks(p);
    finish_phase("time_build_rlz_reference_index");
    if (log) log_peak_mem_usage();
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
void rlz<kind>::construction<value_func_t, intervals_t>::load_reference()
{
    log_phase_start(log, time, "reading R");

    direct_ifstream tmp_file(prefix_tmp_files + "_R");
    reference.load(tmp_file);
    tmp_file.close();
    std::filesystem::remove(prefix_tmp_files + "_R");

    finish_phase("time_load_rlz_reference");
}

template <array_kind_t kind>
template <typename value_func_t, typename intervals_t>
template <rlz_space_t space>
void rlz<kind>::construction<value_func_t, intervals_t>::build_parsing()
{
    log_phase_start(log, time, "computing the RLZ encoding");

    uint64_t num_phrases = 0, num_literals = 0, num_copies = 0;
    std::string revision_log_msg; // log message of a revision by the parse that was not applied
    uint8_t source_width = std::bit_width(reference_length); // SR stores positions in R
    uint8_t literal_width = rlz::value_width(n); // LP stores values in [0..2n]

    // collect_stats <=> the statistics for revising R are collected during the parse (see rlz_parse_stats);
    // then each thread's range starts at the start of a block
    const uint64_t num_blocks = revision.num_input_blocks();
    const bool collect_stats = reference_tuned && revise_tuned_reference && num_blocks >= p;
    range_bounds.clear();
    range_bounds.reserve(p + 1);

    for (uint64_t i = 0; i < p; i++)
        range_bounds.emplace_back(collect_stats ? (num_blocks / p) * i * revision_block_length : i * (n / p));

    range_bounds.emplace_back(n);
    rlz_reference_segments ref_segments;

    if (collect_stats) {
        ref_segments.segments = std::move(reference_segments);
        ref_segments.compute_begins();
    }

    // 1. parse SA^d; if R is revised by the parse, SA^d is parsed again
    for (uint64_t round = 1;; round++) {
        source_width = std::bit_width(reference_length);

        if constexpr (space == _on_disk) {
            for (uint64_t i = 0; i < p; i++) {
                CPL_files.emplace_back(prefix_tmp_files + ".cpl_" + std::to_string(i),
                    std::numeric_limits<uint16_t>::digits, file_buffer_size);
                SR_files.emplace_back(prefix_tmp_files + ".sr_" + std::to_string(i), source_width, file_buffer_size);
                LP_files.emplace_back(prefix_tmp_files + ".lp_" + std::to_string(i), literal_width, file_buffer_size);
                PT_files.emplace_back(prefix_tmp_files + ".pt_" + std::to_string(i), 1, file_buffer_size);
            }
        } else {
            CPL_of_thread.assign(p, std::vector<uint16_t>());
            SR_of_thread.assign(p, interleaved_bit_aligned_vectors({ source_width }));
            LP_of_thread.assign(p, interleaved_bit_aligned_vectors({ literal_width }));
            PT_of_thread.assign(p, sdsl::bit_vector());
        }

        // statistics of the parse (if collect_stats)
        rlz_parse_stats<uint16_t> stats;
        if (collect_stats) stats = revision.new_stats<uint16_t>({});
        rlz_revision::thread_counters_t counters = revision.new_counters(
            collect_stats ? ref_segments.segments.size() : 0, 0);
        phase_progress prog(log, n);

        #pragma omp parallel num_threads(p)
        {
            uint64_t thread = omp_get_thread_num();
            uint64_t beg = range_bounds[thread];
            uint64_t end = range_bounds[thread + 1];
            uint64_t num_written = 1; // number of phrases written in the range
            uint64_t PT_size = std::max<uint64_t>(2, r / p); // size of PT_of_thread[thread]
            rlz_revision::prev_phrase_t prev;

            // writes a phrase starting at position i (see parse_range) and counts it
            auto write_phrase = [&](uint64_t i, bool literal, uint64_t length, uint64_t value) {
                if (literal) {
                    if constexpr (space == _on_disk) {
                        PT_files[thread].push_back(1);
                        LP_files[thread].push_back(value);
                    } else {
                        PT_of_thread[thread][num_written] = 1;
                        LP_of_thread[thread].emplace_back(value);
                    }
                } else {
                    if constexpr (space == _on_disk) {
                        PT_files[thread].push_back(0);
                        CPL_files[thread].push_back(length);
                        SR_files[thread].push_back(value);
                    } else {
                        PT_of_thread[thread][num_written] = 0;
                        CPL_of_thread[thread].emplace_back(length);
                        SR_of_thread[thread].emplace_back(value);
                    }
                }

                if (collect_stats) {
                    stats.phrases[i / revision_block_length]++;
                    revision.count_phrase(prev, counters[thread].data, ref_segments, true, literal, length, value);
                }

                if constexpr (space == _in_memory) {
                    num_written++;

                    if (num_written == PT_size) {
                        PT_size *= 2;
                        PT_of_thread[thread].resize(PT_size);
                    }
                }
            };

            // the first value of a range is a literal phrase
            if constexpr (space == _on_disk) {
                PT_files[thread].push_back(1);
                LP_files[thread].push_back(thread == 0 ? first_value(value_at(thread, beg)) : value_at(thread, beg));
            } else {
                PT_of_thread[thread].resize(PT_size);
                PT_of_thread[thread][0] = 1;
                LP_of_thread[thread].emplace_back(
                    thread == 0 ? first_value(value_at(thread, beg)) : value_at(thread, beg));
            }

            if (collect_stats) {
                stats.phrases[beg / revision_block_length]++;
                revision.count_phrase(prev, counters[thread].data, ref_segments, true, true, 1, 0);
            }

            // the values are buffered in 4 bytes each, if they fit
            if (literal_width <= 32) {
                std::vector<uint32_t> lookahead(initial_lookahead_length);
                parse_range(thread, beg, end, lookahead, write_phrase, &prog);
            } else {
                std::vector<uint64_t> lookahead(initial_lookahead_length);
                parse_range(thread, beg, end, lookahead, write_phrase, &prog);
            }

            if constexpr (space == _in_memory) PT_of_thread[thread].resize(num_written);
        }

        // revise R by the parse and parse SA^d again, if the revision is expected to save at least
        // min_gain of the size of the rlz
        if (!collect_stats || round >= revision_params.whole_rounds) break;

        for (uint64_t thread = 0; thread < p; thread++) {
            if constexpr (space == _on_disk) {
                stats.z += PT_files[thread].size();
                stats.z_l += LP_files[thread].size();
            } else {
                stats.z += PT_of_thread[thread].size();
                stats.z_l += LP_of_thread[thread].size();
            }
        }

        revision.add_counters(stats, counters);
        rlz_revised_reference revised = revise(ref_segments, stats);
        revision_log_msg = rlz_revision::describe(reference_length, stats, revised);
        if (!revised.changes() || revised.gain < revision_params.min_gain * revised.estimated_bytes) break;

        finish_phase("time_build_rlz_first_factorization");
        log_message(log, revision_log_msg + "\n");
        revision_log_msg.clear();
        log_phase_start(log, time, "revising R");

        if constexpr (space == _on_disk) {
            CPL_files.clear();
            SR_files.clear();
            LP_files.clear();
            PT_files.clear();
        }

        apply_revision(ref_segments, std::move(revised.ref_segments));
        finish_phase("time_revise_rlz_reference_by_parse");
        log_phase_start(log, time, "computing the RLZ encoding");
    }

    matcher = rlz_reference_matcher();

    // 2. compute phrases_before, literals_before and copies_before
    phrases_before.emplace_back(0);
    literals_before.emplace_back(0);
    copies_before.emplace_back(0);

    for (uint64_t i = 0; i < p; i++) {
        if constexpr (space == _on_disk) {
            num_phrases += PT_files[i].size();
            num_literals += LP_files[i].size();
        } else {
            num_phrases += PT_of_thread[i].size();
            num_literals += LP_of_thread[i].size();
        }

        phrases_before.emplace_back(num_phrases);
        literals_before.emplace_back(num_literals);
        copies_before.emplace_back(num_phrases - num_literals);
    }

    num_copies = copies_before[p];

    // 3. remove the positions from R that no phrase copies (see rlz_copied_positions)
    rlz_copied_positions copied(reference_length);

    for (uint64_t thread = 0; thread < p; thread++) {
        const uint64_t num = copies_before[thread + 1] - copies_before[thread];

        for (uint64_t i = 0; i < num; i++)
            copied.mark_copied(copy_source<space>(thread, i), copy_length<space>(thread, i));
    }

    const uint64_t num_copied = copied.build_rank();
    const uint64_t num_removed = reference_length - num_copied;

    if (num_copied != reference_length) {
        interleaved_bit_aligned_vectors kept({ literal_width });
        kept.resize_no_init(num_copied);

        #pragma omp parallel for num_threads(p)
        for (uint64_t i = 0; i < reference_length; i++) {
            if (copied.is_copied(i))
                kept.template set_parallel<0, uint64_t>(copied.num_copied_before(i), uint64_t(reference[i]));
        }

        reference = std::move(kept);
        reference_length = num_copied;
        source_width = std::bit_width(reference_length);
    }

    // 4. build CPL, SR, LP and PT of the rlz from the phrases of the threads
    no_init_vector<uint16_t> CPL; // [0..z_c-1] CPL[x] is the length of the x-th copy phrase
    resize_no_init(CPL, num_copies);

    #pragma omp parallel num_threads(p)
    {
        uint64_t thread = omp_get_thread_num();
        uint64_t beg = copies_before[thread];
        uint64_t num = copies_before[thread + 1] - beg;

        for (uint64_t i = 0; i < num; i++) CPL[beg + i] = copy_length<space>(thread, i);
    }

    CPL_of_thread.clear();
    CPL_of_thread.shrink_to_fit();

    // [0..z_c] SR[x] is the starting position in R of the x-th copy phrase; SR[z_c] = |R|
    interleaved_bit_aligned_vectors SR({ source_width });
    SR.resize_no_init(num_copies + 1);

    #pragma omp parallel num_threads(p)
    {
        uint64_t thread = omp_get_thread_num();
        uint64_t beg = copies_before[thread];
        uint64_t num = copies_before[thread + 1] - beg;

        for (uint64_t i = 0; i < num; i++)
            SR.template set_parallel<0, uint64_t>(beg + i, copied.num_copied_before(copy_source<space>(thread, i)));
    }

    SR.template set_parallel<0, uint64_t>(num_copies, reference_length);

    // [0..z_l-1] LP[x] is the value of the x-th literal phrase
    interleaved_bit_aligned_vectors LP({ literal_width });
    LP.resize_no_init(num_literals, p);

    #pragma omp parallel num_threads(p)
    {
        uint64_t thread = omp_get_thread_num();
        uint64_t beg = literals_before[thread];
        uint64_t num = literals_before[thread + 1] - beg;

        for (uint64_t i = 0; i < num; i++)
            LP.template set_parallel<0, uint64_t>(beg + i, literal_value<space>(thread, i));
    }

    // [0..z+1] PT[x] = 1 <=> the x-th phrase is literal; PT[z] = PT[z+1] = 0
    sdsl::bit_vector PT_bits;
    PT_bits.resize(num_phrases + 2);

    for (uint64_t thread = 0; thread < p; thread++) {
        uint64_t beg = phrases_before[thread];
        uint64_t num = phrases_before[thread + 1] - beg;

        for (uint64_t i = 0; i < num; i++) PT_bits[beg + i] = is_literal<space>(thread, i);
    }

    PT_bits[num_phrases] = 0;
    PT_bits[num_phrases + 1] = 0;
    plain_bit_vector PT(std::move(PT_bits), RANK_1 | SELECT_0 | SELECT_1, p);

    if constexpr (space == _on_disk) {
        CPL_files.clear();
        SR_files.clear();
        LP_files.clear();
        PT_files.clear();
    }

    // 5. build the phrase starts (a literal phrase has length 1)
    elias_fano_bit_vector phrase_starts;

    {
        std::vector<uint64_t> phrase_lengths;
        phrase_lengths.reserve(num_phrases);
        uint64_t copy = 0;

        for (uint64_t phrase = 0; phrase < num_phrases; phrase++) {
            if (PT[phrase]) phrase_lengths.emplace_back(1);
            else phrase_lengths.emplace_back(CPL[copy++]);
        }

        phrase_starts = rlz_phrase_starts(phrase_lengths, n, p);
    }

    built = rlz(
        n, num_phrases, num_literals, num_copies,
        std::move(reference), std::move(PT), std::move(phrase_starts),
        std::move(SR), std::move(LP));

    finish_phase("time_build_rlz_factorization");

    if (log) {
        if (!revision_log_msg.empty()) log_out() << revision_log_msg << std::endl;
        log_out() << "z = " << format_number(num_phrases) << ", z_l/z = "
                  << format_ratio(num_literals / (double) num_phrases) << ", z/r = "
                  << format_ratio(num_phrases / (double) r) << ", values removed from R = " << format_number(num_removed)
                  << ", |R| = " << format_number(reference_length) << std::endl;
    }
}

#endif // RLZ_REF_RLZ_REF_CONSTRUCTION_CONSTRUCTION_TPP

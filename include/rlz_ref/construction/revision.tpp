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

#ifndef RLZ_REF_RLZ_REF_CONSTRUCTION_REVISION_TPP
#define RLZ_REF_RLZ_REF_CONSTRUCTION_REVISION_TPP

#include <bit>

#include <omp.h>

#include <rlz_ref/construction/revision.hpp>

inline std::vector<uint64_t> rlz_revision::draw_blocks()
{
    const uint64_t num_drawn = std::clamp<uint64_t>(uint64_t(params.sample_share * num_blocks),
        1, std::min<uint64_t>(num_blocks, params.max_sampled_blocks));
    std::vector<uint64_t> blocks;
    blocks.reserve(num_drawn);
    for (uint64_t i = 0; i < num_drawn; i++) blocks.emplace_back(rlz_draw(random, 0, num_blocks - 1));
    std::sort(blocks.begin(), blocks.end());
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
    return blocks;
}

inline std::pair<uint64_t, uint64_t> rlz_revision::sampled_range(uint64_t block) const
{
    const uint64_t num_lead_blocks = (rlz_sample_window_length + block_length - 1) / block_length;
    return { block_beg(block - std::min<uint64_t>(block, num_lead_blocks)), block_end(block) };
}

template <typename phrase_count_t>
rlz_parse_stats<phrase_count_t> rlz_revision::new_stats(std::vector<uint64_t> blocks) const
{
    rlz_parse_stats<phrase_count_t> stats;
    stats.blocks = std::move(blocks);
    const uint64_t num_parsed_blocks = stats.whole() ? num_blocks : stats.blocks.size();
    stats.phrases.assign(num_parsed_blocks, 0);
    stats.scale = num_blocks / double(std::max<uint64_t>(1, num_parsed_blocks));
    return stats;
}

inline rlz_revision::thread_counters_t rlz_revision::new_counters(uint64_t num_segments,
    uint64_t num_reference_blocks) const
{
    thread_counters_t counters(threads);

    for (cache_line_padded<counters_t>& thread_counters : counters) {
        thread_counters.data.ended_with.assign(num_segments * max_lengthening, 0);
        thread_counters.data.began_with.assign(num_segments * max_lengthening, 0);
        thread_counters.data.copied_from.assign(num_reference_blocks, 0);
    }

    return counters;
}

template <typename phrase_count_t>
void rlz_revision::add_counters(rlz_parse_stats<phrase_count_t>& stats, const thread_counters_t& counters) const
{
    stats.ended_with.assign(counters[0].data.ended_with.size(), 0);
    stats.began_with.assign(counters[0].data.began_with.size(), 0);
    stats.copied_from.assign(counters[0].data.copied_from.size(), 0);

    for (const cache_line_padded<counters_t>& thread_counters : counters) {
        for (uint64_t i = 0; i < stats.ended_with.size(); i++) {
            add_count(stats.ended_with[i], thread_counters.data.ended_with[i]);
            add_count(stats.began_with[i], thread_counters.data.began_with[i]);
        }

        for (uint64_t i = 0; i < stats.copied_from.size(); i++)
            add_count(stats.copied_from[i], thread_counters.data.copied_from[i]);
    }
}

inline void rlz_revision::count_phrase(prev_phrase_t& prev, counters_t& counters,
    const rlz_reference_segments& ref_segments, bool counted, bool literal, uint64_t length, uint64_t source) const
{
    const uint64_t segment = literal ? none : ref_segments.segment_of(source);

    if (counted && prev.counted) {
        // the previous phrase ends at the end of a segment
        if (prev.end_segment != none)
            count(counters.ended_with[prev.end_segment * max_lengthening + blocks_needed(length) - 1]);

        // the phrase starts at the start of a segment
        if (!literal && source == ref_segments.begins[segment] && prev.length != 0)
            count(counters.began_with[segment * max_lengthening + blocks_needed(prev.length) - 1]);
    }

    prev.end_segment = !literal && source + length == ref_segments.begins[segment + 1] ? segment : none;
    prev.length = length;
    prev.counted = counted;
}

template <typename phrase_count_t, typename reference_symbol_t, typename block_symbols_t>
rlz_revised_reference rlz_revision::revise(const rlz_reference_segments& ref_segments,
    uint64_t reference_length, const rlz_parse_stats<phrase_count_t>& stats, const rlz_size_estimate_t& estimate,
    reference_symbol_t&& reference_symbol, block_symbols_t&& block_symbols, uint64_t max_reference_length)
{
    rlz_revised_reference revised;
    revised.ref_segments = ref_segments;
    if (reference_length == 0) return revised;

    // estimated size of a phrase and of a block of R
    const double z = estimate.phrases;
    const double phrase_bytes = (estimate.bytes - estimate.reference_bytes) / std::max<double>(1.0, z);
    const double block_bytes = block_length * (estimate.reference_bytes / reference_length);
    revised.estimated_bytes = estimate.bytes;
    // returns the index in the input of the b-th parsed block
    auto block_of = [&](uint64_t b) { return stats.whole() ? b : stats.blocks[b]; };

    // 1. remove the blocks of R that no phrase copies from
    if (stats.whole() && !stats.copied_from.empty()) {
        revised.ref_segments.segments.clear();

        for (uint64_t s = 0; s < ref_segments.segments.size(); s++) {
            uint64_t beg = none; // starting position in the input of the current kept part of the segment

            for (uint64_t pos = ref_segments.begins[s]; pos < ref_segments.begins[s + 1];) {
                const uint64_t block = pos / block_length;
                const uint64_t until = std::min<uint64_t>(ref_segments.begins[s + 1],
                    (block + 1) * block_length);
                const uint64_t in_input = ref_segments.segments[s].beg + (pos - ref_segments.begins[s]);

                if (stats.copied_from[block] != 0) {
                    if (beg == none) beg = in_input;
                } else {
                    if (beg != none) revised.ref_segments.segments.emplace_back(rlz_segment_t { beg, in_input });
                    beg = none;
                }

                pos = until;
            }

            if (beg != none)
                revised.ref_segments.segments.emplace_back(rlz_segment_t { beg, ref_segments.segments[s].end });
        }
    }

    // a segment of the input added to R
    struct addition_t {
        rlz_segment_t segment;
        double gain; // number of bytes it is expected to save
        bool lengthens; // true <=> it lengthens an end of a segment of R, false <=> it is a block added to R
    };

    std::vector<addition_t> additions; // the segments of the input added to R

    // 2. lengthen each end and start of a segment by the number of blocks with the largest expected gain
    for (uint64_t s = 0; s < ref_segments.segments.size(); s++) {
        for (const std::vector<uint32_t>* counted : { &stats.ended_with, &stats.began_with }) {
            double num_saved = 0; // number of phrases saved by lengthening by e blocks
            double best_gain = 0;
            uint64_t best_e = 0;

            for (uint64_t e = 1; e <= max_lengthening; e++) {
                num_saved += (*counted)[s * max_lengthening + e - 1];
                const double gain = num_saved * stats.scale * phrase_bytes - e * block_bytes;

                if (gain > best_gain) {
                    best_gain = gain;
                    best_e = e;
                }
            }

            if (best_e == 0) continue;
            const bool at_end = counted == &stats.ended_with;
            const rlz_segment_t& segment = ref_segments.segments[s];
            const uint64_t beg = at_end ? segment.end
                : segment.beg - std::min<uint64_t>(segment.beg, best_e * block_length);
            const uint64_t end = at_end ? std::min<uint64_t>(n, segment.end + best_e * block_length)
                : segment.beg;
            if (end <= beg) continue;

            additions.emplace_back(addition_t { rlz_segment_t { beg, end }, best_gain, true });
        }
    }

    // 3. add blocks in which at least min_phrases_in_dense_block phrases start; dense stores their indices
    // in stats.phrases in descending order of their numbers of phrases
    std::vector<uint64_t> dense;

    for (uint64_t b = 0; b < stats.phrases.size(); b++) {
        if (stats.phrases[b] >= params.min_phrases_in_dense_block
            && !ref_segments.contains(block_beg(block_of(b)))) dense.emplace_back(b);
    }

    auto denser = [&](uint64_t a, uint64_t b) {
        return stats.phrases[a] > stats.phrases[b] || (stats.phrases[a] == stats.phrases[b] && a < b);
    };

    if (dense.size() > params.max_dense_blocks) {
        std::nth_element(dense.begin(), dense.begin() + params.max_dense_blocks, dense.end(), denser);
        dense.resize(params.max_dense_blocks);
    }

    std::sort(dense.begin(), dense.end(), denser);

    if (!dense.empty()) {
        // half the average phrase length, rounded down to a power of two
        const uint64_t k = std::clamp<uint64_t>(
            std::bit_floor(std::max<uint64_t>(1, uint64_t(n / std::max<double>(1.0, z)) / 2)),
            params.min_kmer_length, params.max_kmer_length);
        // only the hashed fingerprints divisible by divisor are used (about 2^20 fingerprints of R)
        const uint64_t divisor = std::max<uint64_t>(4, std::bit_ceil(reference_length >> 20));

        // the fingerprints of the k-mers in R; each thread computes those of one part of R
        gtl::flat_hash_set<uint64_t, std::identity> reference_fingerprints;
        {
            std::vector<std::vector<uint64_t>> part_fingerprints(threads);
            const uint64_t part_length = (reference_length + threads - 1) / threads;

            #pragma omp parallel for num_threads(threads)
            for (uint64_t part = 0; part < threads; part++) {
                const uint64_t beg = std::min<uint64_t>(reference_length, part * part_length);
                const uint64_t end = std::min<uint64_t>(reference_length, beg + part_length + k - 1);

                rlz_karp_rabin::for_each_sampled_kmer(end - beg, k, divisor,
                    [&](uint64_t i) { return reference_symbol(beg + i); },
                    [&](uint64_t fingerprint) { part_fingerprints[part].emplace_back(fingerprint); });
            }

            for (const std::vector<uint64_t>& fingerprints : part_fingerprints)
                for (uint64_t fingerprint : fingerprints) reference_fingerprints.emplace(fingerprint);
        }

        // a group of blocks with common k-mers
        struct group_t {
            uint64_t founder; // the first block of the group (the one with the most phrases)
            double saved;     // number of phrases in the parsed blocks that adding the founder to R saves
        };

        std::vector<group_t> groups;
        // maps the fingerprint of a k-mer to the group whose founder contains it
        gtl::flat_hash_map<uint64_t, uint32_t, std::identity> fingerprint_group;
        // shared[g] = number of fingerprints the current block shares with group g
        std::vector<uint32_t> shared;
        static constexpr uint64_t batch_length = 256;
        // [0..batch_length-1] the fingerprints of the blocks of the current batch
        std::vector<std::vector<uint64_t>> fingerprints(batch_length);

        // compute the fingerprints of a batch of blocks in parallel, then assign its blocks to groups: a
        // block starts a group, if at least min_new_kmer_share of its k-mers occur neither in R nor in a
        // founder; else it is counted to the group it shares the most k-mers with
        for (uint64_t batch = 0; batch < dense.size(); batch += batch_length) {
            const uint64_t in_batch = std::min<uint64_t>(batch_length, dense.size() - batch);

            #pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
            for (uint64_t i = 0; i < in_batch; i++) {
                const uint64_t block = block_of(dense[batch + i]);
                auto symbol = block_symbols(uint64_t(omp_get_thread_num()), block_beg(block), block_end(block));
                fingerprints[i].clear();

                rlz_karp_rabin::for_each_sampled_kmer(block_end(block) - block_beg(block), k, divisor, symbol,
                    [&](uint64_t fingerprint) { fingerprints[i].emplace_back(fingerprint); });
            }

            for (uint64_t i = 0; i < in_batch; i++) {
                if (fingerprints[i].empty()) continue;
                const double saved = std::max<double>(0.0,
                    stats.phrases[dense[batch + i]] - params.phrases_left_after_adding);
                uint64_t num_new = 0;
                shared.assign(groups.size(), 0);

                for (uint64_t fingerprint : fingerprints[i]) {
                    if (reference_fingerprints.contains(fingerprint)) continue;
                    auto it = fingerprint_group.find(fingerprint);

                    if (it == fingerprint_group.end()) num_new++;
                    else shared[it->second]++;
                }

                if (num_new >= params.min_new_kmer_share * fingerprints[i].size()) {
                    const uint32_t group = uint32_t(groups.size());
                    groups.emplace_back(group_t { dense[batch + i], saved });

                    for (uint64_t fingerprint : fingerprints[i]) {
                        if (!reference_fingerprints.contains(fingerprint))
                            fingerprint_group.try_emplace(fingerprint, group);
                    }
                } else if (!groups.empty()) {
                    const uint64_t best = std::max_element(shared.begin(), shared.end()) - shared.begin();

                    if (shared[best] >= params.min_new_kmer_share * fingerprints[i].size())
                        groups[best].saved += saved;
                }
            }
        }

        // add the founders with a positive expected gain
        for (const group_t& group : groups) {
            const double gain = group.saved * stats.scale * phrase_bytes - block_bytes;
            if (gain <= 0) continue;
            const uint64_t block = block_of(group.founder);
            additions.emplace_back(addition_t { rlz_segment_t { block_beg(block), block_end(block) }, gain, false });
        }
    }

    // returns R (after step 1) with the first count additions
    auto with_additions = [&](uint64_t count) {
        rlz_reference_segments result = revised.ref_segments;
        std::vector<rlz_segment_t> added;
        added.reserve(count);
        for (uint64_t i = 0; i < count; i++) added.emplace_back(additions[i].segment);
        result.insert(std::move(added));
        result.compute_begins();
        return result;
    };

    rlz_reference_segments result = with_additions(additions.size());

    // 4. if R becomes longer than max_reference_length, keep only the additions with the largest gains that fit
    if (result.length() > max_reference_length) {
        std::stable_sort(additions.begin(), additions.end(),
            [](const addition_t& a, const addition_t& b) { return a.gain > b.gain; });
        // R with the first fitting additions has at most max_reference_length positions, with the first too_many not
        uint64_t fitting = 0, too_many = additions.size();

        while (too_many - fitting > 1) {
            const uint64_t middle = fitting + (too_many - fitting) / 2;
            if (with_additions(middle).length() <= max_reference_length) fitting = middle;
            else too_many = middle;
        }

        additions.resize(fitting);
        result = with_additions(fitting);
    }

    for (const addition_t& addition : additions) {
        if (addition.lengthens) {
            revised.num_lengthened_ends++;
            revised.num_lengthening_blocks += (addition.segment.length() + block_length - 1) / block_length;
        } else {
            revised.num_added_blocks++;
        }

        revised.gain += addition.gain;
    }

    revised.ref_segments = std::move(result);
    return revised;
}

template <typename phrase_count_t>
std::string rlz_revision::describe(uint64_t reference_length, const rlz_parse_stats<phrase_count_t>& stats,
    const rlz_revised_reference& revised)
{
    const std::string estimated = stats.whole() ? "" : "~";
    return "  revising: |R| = " + format_number(reference_length) + "\n"
        + "    " + estimated + "z = " + format_number(uint64_t(stats.z * stats.scale)) + "\n"
        + "    " + estimated + "size = " + format_size(uint64_t(revised.estimated_bytes)) + "\n"
        + "    blocks added = " + format_number(revised.num_added_blocks) + "\n"
        + "    ends of segments lengthened = " + format_number(revised.num_lengthened_ends) + "\n"
        + "    blocks added at the ends of segments = " + format_number(revised.num_lengthening_blocks) + "\n"
        + "    expected saving = " + format_size(uint64_t(revised.gain));
}

#endif // RLZ_REF_RLZ_REF_CONSTRUCTION_REVISION_TPP

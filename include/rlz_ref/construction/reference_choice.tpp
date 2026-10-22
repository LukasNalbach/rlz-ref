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

#ifndef RLZ_REF_RLZ_REF_CONSTRUCTION_REFERENCE_CHOICE_TPP
#define RLZ_REF_RLZ_REF_CONSTRUCTION_REFERENCE_CHOICE_TPP

#include <omp.h>

#include <rlz_ref/construction/reference_choice.hpp>

inline void rlz_reference_segments::insert(std::vector<rlz_segment_t> added)
{
    added.insert(added.end(), segments.begin(), segments.end());
    std::sort(added.begin(), added.end(),
        [](const rlz_segment_t& a, const rlz_segment_t& b) { return a.beg < b.beg; });
    segments.clear();

    for (const rlz_segment_t& segment : added) {
        if (segment.end <= segment.beg) continue;

        if (!segments.empty() && segment.beg <= segments.back().end)
            segments.back().end = std::max<uint64_t>(segments.back().end, segment.end);
        else segments.emplace_back(segment);
    }
}

inline void rlz_copied_positions::mark_copied(uint64_t source, uint64_t length)
{
    const uint64_t last = source + length - 1;
    // the bits of the first and of the last word that are copied
    const uint64_t first_word_bits = ~uint64_t(0) << (source % 64);
    const uint64_t last_word_bits = ~uint64_t(0) >> (63 - last % 64);

    if (source / 64 == last / 64) {
        copied[source / 64] |= first_word_bits & last_word_bits;
        return;
    }

    copied[source / 64] |= first_word_bits;
    for (uint64_t word = source / 64 + 1; word < last / 64; word++) copied[word] = ~uint64_t(0);
    copied[last / 64] |= last_word_bits;
}

inline uint64_t rlz_copied_positions::build_rank()
{
    copied_before.assign(copied.size() + 1, 0);

    for (uint64_t word = 0; word < copied.size(); word++)
        copied_before[word + 1] = copied_before[word] + std::popcount(copied[word]);

    return copied_before.back();
}

template <typename scorer_t>
auto rlz_reference_choice<scorer_t>::clip_to_free_space(uint64_t& beg, uint64_t& end) -> segment_it_t
{
    // the first chosen segment starting at or after beg; the candidate must end before it
    segment_it_t next = chosen_by_beg.lower_bound(rlz_segment_t { beg, 0 });

    if (next != chosen_by_beg.end() && (*next).beg < end) {
        end = (*next).beg;
    }

    // the candidate must start after the previous chosen segment; if it lies within that segment, it is
    // set to the empty range [beg, beg)
    if (next != chosen_by_beg.begin()) {
        segment_it_t before = next;
        --before;
        uint64_t end_before = (*before).end;

        if (beg < end_before) {
            if (end < end_before) end = beg;
            else beg = end_before;
        }
    }

    return next;
}

template <typename scorer_t>
void rlz_reference_choice<scorer_t>::insert_chosen(uint64_t beg, uint64_t end, segment_it_t next)
{
    // the segment ends where the next chosen segment starts
    if (next != chosen_by_beg.end() && end == (*next).beg) {
        if (next != chosen_by_beg.begin()) {
            segment_it_t before = next;
            --before;

            // it also starts where the previous chosen segment ends: merge the three segments
            if ((*before).end == beg) {
                (*before).end = (*next).end;
                chosen_by_beg.erase(next);
                return;
            }
        }

        (*next).beg = beg;
        return;
    }

    // the segment starts where the previous chosen segment ends
    if (!chosen_by_beg.empty() && next != chosen_by_beg.begin()) {
        segment_it_t before = next;
        --before;

        if ((*before).end == beg) {
            (*before).end = end;
            return;
        }
    }

    chosen_by_beg.emplace_hint(next, rlz_segment_t { beg, end });
}

template <typename scorer_t>
void rlz_reference_choice<scorer_t>::choose_segments()
{
    if (target_length == 0 || n == 0) return;

    const uint64_t segment_length = std::min<uint64_t>(this->segment_length,
        std::max<uint64_t>(1, target_length));
    // number of candidates per batch
    const uint64_t batch_length = candidates_per_batch == 0 ? num_candidates
        : std::min<uint64_t>(candidates_per_batch, num_candidates);

    // the input is divided into batch_length sections; the i-th candidate of a batch starts in the
    // i-th section
    // [0..batch_length-1] the smallest and the largest start of a candidate of each section
    std::vector<std::pair<uint64_t, uint64_t>> draw_start;
    draw_start.reserve(batch_length);

    for (uint64_t section = 0; section < batch_length; section++) {
        const uint64_t section_beg = section * (n / batch_length);
        const uint64_t section_end = section + 1 == batch_length ? n : (section + 1) * (n / batch_length);
        draw_start.emplace_back(section_beg,
            std::max<int64_t>(section_beg, (int64_t) section_end - (int64_t) segment_length));
    }

    // a candidate segment [beg, end) and its score
    struct candidate_t {
        uint64_t beg;
        uint64_t end;
        float score;
    };

    std::vector<candidate_t> batch;
    batch.reserve(batch_length);
    uint64_t choose_until = choose_share * target_length;

    while (total_chosen_length < choose_until) {
        candidate_t best { 0, 0, 0 };

        // draw and score the candidates of this round batch by batch, until a batch does not improve
        // the best score
        for (uint64_t drawn = 0; drawn < num_candidates; drawn += batch_length) {
            batch.clear();

            for (uint64_t section = 0; section < batch_length; section++) {
                uint64_t beg, end;
                uint8_t num_draws = 0;

                // redraw a candidate that lies within a chosen segment
                do {
                    num_draws++;
                    beg = rlz_draw(random, draw_start[section].first, draw_start[section].second);
                    end = std::min<uint64_t>(beg + segment_length, n);
                    clip_to_free_space(beg, end);
                } while (num_draws < max_draws_per_candidate && end == beg);

                if (end != beg) batch.emplace_back(candidate_t { beg, end, 0 });
            }

            #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1)
            for (uint64_t i = 0; i < batch.size(); i++) {
                batch[i].score = scorer.score(omp_get_thread_num(), batch[i].beg, batch[i].end);
            }

            bool improved = false;

            for (const candidate_t& candidate : batch) {
                if (candidate.score > best.score) {
                    best = candidate;
                    improved = true;
                }
            }

            if (!improved) break;
        }

        // no candidate has a positive score
        if (best.score == 0) {
            if (!choose_random_if_unscored || batch.empty()) break;
            best = batch[rlz_draw(random, 0, batch.size() - 1)];
        }

        if (first_score == 0) first_score = best.score;
        else if (best.score < min_score_share * first_score) break;

        insert_chosen(best.beg, best.end, chosen_by_beg.lower_bound(rlz_segment_t { best.beg, 0 }));
        scorer.cover(0, best.beg, best.end);
        total_chosen_length += best.end - best.beg;
        chosen_in_order.emplace_back(rlz_segment_t { best.beg, best.end });
    }
}

template <typename scorer_t>
float rlz_reference_choice<scorer_t>::gap_score(uint64_t beg_before, uint64_t end_after,
    uint64_t gap_length)
{
    return float(end_after - beg_before) / float(std::max<uint64_t>(1, gap_length));
}

template <typename scorer_t>
void rlz_reference_choice<scorer_t>::close_gaps()
{
    if (chosen_by_beg.size() < 2) return;

    // insert the gaps between all pairs of consecutive segments
    for (segment_it_t it = chosen_by_beg.begin();;) {
        segment_it_t before = it;
        ++it;
        if (it == chosen_by_beg.end()) break;

        gaps.emplace(gap_t { .beg_before = (*before).beg,
            .score = gap_score((*before).beg, (*it).end, (*it).beg - (*before).end) });
    }

    while (!gaps.empty()) {
        // the gap with the highest score
        auto best = gaps.begin();
        segment_it_t left = chosen_by_beg.find(rlz_segment_t { (*best).beg_before, 0 });
        gaps.erase(best);

        segment_it_t right = left;
        ++right;
        uint64_t gap_length = (*right).beg - (*left).end;

        if (total_chosen_length + gap_length > target_length) continue;
        total_chosen_length += gap_length;

        // the two segments are merged; update the scores of the gaps before and after them and remove a gap
        // whose closing would exceed target_length
        if (left != chosen_by_beg.begin()) {
            segment_it_t outer = left;
            --outer;
            uint64_t outer_gap = (*left).beg - (*outer).end;

            if (gaps.erase(gap_t { (*outer).beg, gap_score((*outer).beg, (*left).end, outer_gap) })
                && total_chosen_length + outer_gap <= target_length) {
                gaps.emplace(gap_t { (*outer).beg, gap_score((*outer).beg, (*right).end, outer_gap) });
            }
        }

        segment_it_t outer = right;
        ++outer;

        if (outer != chosen_by_beg.end()) {
            uint64_t outer_gap = (*outer).beg - (*right).end;

            if (gaps.erase(gap_t { (*right).beg, gap_score((*right).beg, (*outer).end, outer_gap) })
                && total_chosen_length + outer_gap <= target_length) {
                gaps.emplace(gap_t { (*left).beg, gap_score((*left).beg, (*outer).end, outer_gap) });
            }
        }

        (*left).end = (*right).end;
        chosen_by_beg.erase(right);
    }
}

template <typename scorer_t>
std::vector<rlz_segment_t> rlz_reference_choice<scorer_t>::segments_for_target(uint64_t target)
{
    target = std::min(target, rlz_max_reference_length);
    segments_t kept_chosen = chosen_by_beg;
    const uint64_t kept_length = total_chosen_length;
    const uint64_t kept_target = target_length;

    // choose the segments in the order of chosen_in_order, then close the gaps
    const uint64_t choose_until = uint64_t(choose_share * target);
    chosen_by_beg = segments_t();
    total_chosen_length = 0;
    target_length = target;

    for (const rlz_segment_t& segment : chosen_in_order) {
        if (total_chosen_length >= choose_until) break;
        uint64_t beg = segment.beg;
        uint64_t end = segment.end;
        const segment_it_t next = clip_to_free_space(beg, end);
        if (end == beg) continue;
        insert_chosen(beg, end, next);
        total_chosen_length += end - beg;
    }

    close_gaps();
    std::vector<rlz_segment_t> result = segments();

    // restore the chosen segments
    chosen_by_beg = std::move(kept_chosen);
    total_chosen_length = kept_length;
    target_length = kept_target;
    gaps = decltype(gaps)();
    return result;
}

#endif // RLZ_REF_RLZ_REF_CONSTRUCTION_REFERENCE_CHOICE_TPP

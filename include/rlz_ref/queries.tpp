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

#ifndef RLZ_REF_RLZ_REF_QUERIES_TPP
#define RLZ_REF_RLZ_REF_QUERIES_TPP

#include <rlz_ref/rlz.hpp>

// ############################# DECODING #############################

template <array_kind_t kind>
inline void rlz<kind>::decode_context_t::init_right(uint64_t i)
{
    this->i = i;

    // the phrase containing i and the starting position of the next phrase
    const auto bounds = parsing->phrase_bounds_at(i);
    x_p = bounds.phrase;
    s_np = bounds.end;
    // number of literal phrases and copy phrases before the phrase containing i; if it is a literal
    // phrase, x_cp is the index of the next copy phrase
    x_lp = parsing->PT().rank_1(x_p);
    x_cp = x_p - x_lp;
    x_r = parsing->PT(x_p) ? parsing->SR(x_cp)
        : parsing->SR(x_cp) + (i - bounds.beg);
}

template <array_kind_t kind>
inline void rlz<kind>::decode_context_t::init_left(uint64_t i)
{
    init_right(i);
    turn_left();
}

template <array_kind_t kind>
inline void rlz<kind>::decode_context_t::turn_left()
{
    if (parsing->PT(x_p)) {
        s_np--;

        if (x_cp > 0) [[likely]] {
            x_cp--;
            x_r = parsing->SR(x_cp) + copy_length(x_cp) - 1;
        }
    } else {
        s_np -= copy_length(x_cp);
        x_lp--;
    }
}

template <array_kind_t kind>
inline void rlz<kind>::decode_context_t::turn_right()
{
    if (parsing->PT(x_p)) {
        s_np++;

        // x_cp is the previous copy phrase; the next copy phrase is computed from x_p (x_cp++ is wrong in
        // the run of literal phrases at the start, which has no previous copy phrase)
        x_cp = parsing->PT().rank_0(x_p);
        x_r = parsing->SR(x_cp);
    } else {
        s_np += copy_length(x_cp);
        x_lp++;
    }
}

template <array_kind_t kind>
inline void rlz<kind>::decode_context_t::skip_right(uint64_t e)
{
    while (i < e) {
        // decode all copy phrases before the next literal phrase
        while (i < e && !parsing->PT(x_p)) {
            // decode the x_cp-th copy phrase
            while (i < s_np && i < e) {
                add_difference<RIGHT>(parsing->reference(), x_r);
                i++;
                x_r++;
            }

            if (i < e || i == s_np) [[likely]] {
                x_p++;
                x_cp++;
                x_r = parsing->SR(x_cp);
                s_np += parsing->phrase_length(x_p);
            }
        }

        // decode all literal phrases before the next copy phrase
        while (i < e && parsing->PT(x_p)) {
            // decode the x_lp-th literal phrase
            add_difference<RIGHT>(parsing->LP(), x_lp);
            i++;
            x_p++;
            x_lp++;

            if (parsing->PT(x_p)) [[likely]] {
                s_np++;
            } else {
                s_np += parsing->phrase_length(x_p);
            }
        }
    }
}

template <array_kind_t kind>
inline void rlz<kind>::decode_context_t::skip_left(uint64_t b)
{
    while (i > b) {
        // skip all copy phrases after the next literal phrase
        while (i > b && !parsing->PT(x_p)) {
            // skip the x_cp-th copy phrase
            while (i >= s_np && i > b) {
                add_difference<LEFT>(parsing->reference(), x_r);
                i--;
                x_r--;
            }

            if (i > b || i + 1 == s_np) [[likely]] {
                if (x_cp > 0) [[likely]] {
                    x_cp--;
                    x_r = parsing->SR(x_cp) + copy_length(x_cp) - 1;
                }

                x_p--;
                s_np -= parsing->phrase_length(x_p);
            }
        }

        // skip all literal phrases after the next copy phrase
        while (i > b && parsing->PT(x_p)) {
            // skip the x_lp-th literal phrase
            add_difference<LEFT>(parsing->LP(), x_lp);
            i--;
            x_p--;
            x_lp--;

            if (parsing->PT(x_p)) [[likely]] {
                s_np--;
            } else {
                s_np -= copy_length(x_cp);
            }
        }
    }
}

template <array_kind_t kind>
inline uint64_t rlz<kind>::decode_context_t::prev()
{
    if (parsing->PT(x_p)) {
        // literal phrase
        add_difference<LEFT>(parsing->LP(), x_lp);
        i--;
        enter_previous_phrase_from_literal();
    } else {
        // copy phrase
        add_difference<LEFT>(parsing->reference(), x_r);
        i--;
        x_r--;

        // i lies within the previous phrase
        if (i < s_np) [[unlikely]] enter_previous_phrase_from_copy();
    }

    return s;
}

template <array_kind_t kind>
inline uint64_t rlz<kind>::decode_context_t::next()
{
    if (parsing->PT(x_p)) {
        // literal phrase
        add_difference<RIGHT>(parsing->LP(), x_lp);
        i++;
        enter_next_phrase_from_literal();
    } else {
        // copy phrase
        add_difference<RIGHT>(parsing->reference(), x_r);
        i++;
        x_r++;

        // there is a new phrase starting at i
        if (s_np <= i && i < parsing->n) [[unlikely]] enter_next_phrase_from_copy();
    }

    return s;
}

template <array_kind_t kind>
template <typename report_fnc_t>
inline void rlz<kind>::decode_context_t::report_left(uint64_t b, report_fnc_t report)
{
    while (true) {
        // decode all copy phrases after the previous literal phrase
        while (!parsing->PT(x_p)) {
            // decode the x_cp-th copy phrase
            while (i >= s_np) {
                add_difference<LEFT>(parsing->reference(), x_r);
                i--;
                x_r--;
                report(i, s);

                // leave the context in the state that prev() leaves
                if (i == b) [[unlikely]] {
                    if (i < s_np) enter_previous_phrase_from_copy();
                    return;
                }
            }

            if (x_cp > 0) [[likely]] {
                x_cp--;
                x_r = parsing->SR(x_cp) + copy_length(x_cp) - 1;
            }

            x_p--;
            s_np -= parsing->phrase_length(x_p);
        }

        // decode all literal phrases after the previous copy phrase
        while (parsing->PT(x_p)) {
            // decode the x_lp-th literal phrase
            add_difference<LEFT>(parsing->LP(), x_lp);
            i--;
            report(i, s);

            // leave the context in the state that prev() leaves
            if (i == b) [[unlikely]] {
                enter_previous_phrase_from_literal();
                return;
            }

            x_p--;
            x_lp--;
            s_np--;
        }

        // set s_np to the starting position of the last (the x_lp-th)
        // literal phrase before the current (the x_cp-th) copy phrase
        s_np -= copy_length(x_cp) - 1;
    }
}

template <array_kind_t kind>
template <typename report_fnc_t>
inline void rlz<kind>::decode_context_t::report_right(uint64_t e, report_fnc_t report)
{
    while (true) {
        // decode all copy phrases before the next literal phrase
        while (!parsing->PT(x_p)) {
            // decode the x_cp-th copy phrase
            while (i < s_np) {
                add_difference<RIGHT>(parsing->reference(), x_r);
                report(i, s);
                i++;
                x_r++;

                // leave the context in the state that next() leaves
                if (i > e) [[unlikely]] {
                    if (s_np <= i && i < parsing->n) enter_next_phrase_from_copy();
                    return;
                }
            }

            x_p++;
            x_cp++;
            x_r = parsing->SR(x_cp);
            s_np += parsing->phrase_length(x_p);
        }

        // decode all literal phrases before the next copy phrase
        while (parsing->PT(x_p)) {
            // decode the x_lp-th literal phrase
            add_difference<RIGHT>(parsing->LP(), x_lp);
            report(i, s);
            i++;

            // leave the context in the state that next() leaves
            if (i > e) [[unlikely]] {
                enter_next_phrase_from_literal();
                return;
            }

            x_p++;
            x_lp++;
            s_np++;
        }

        // set s_np to the starting position of the next (the x_lp-th)
        // literal phrase after the current (the x_cp-th) copy phrase
        s_np += copy_length(x_cp) - 1;
    }
}

#endif // RLZ_REF_RLZ_REF_QUERIES_TPP

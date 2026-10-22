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

#ifndef RLZ_REF_RLZ_REF_RLZ_TPP
#define RLZ_REF_RLZ_REF_RLZ_TPP

#include <rlz_ref/rlz.hpp>

// ############################# MISC METHODS #############################

template <array_kind_t kind>
inline void rlz<kind>::log_data_structures() const
{
    std::cout << "z = " << format_number(z) << std::endl;
    std::cout << "z_l = " << format_number(z_l) << std::endl;
    std::cout << "z_c = " << format_number(z_c) << std::endl;
    std::cout << std::endl;

    // the values of R and LP are printed without the offset n
    aligned_table log_r;
    log_r.add_row("reference:", _reference.size(), [&](uint64_t i) {
        return (int64_t) _reference[i] - (int64_t) n; });
    log_r.print();

    log_indexed("types:", _PT);
    log_indexed("sources:", _SR);

    aligned_table log_lp;
    log_lp.add_row("literals:", _LP.size(), [&](uint64_t i) {
        return (int64_t) _LP[i] - (int64_t) n; });
    log_lp.print();
}

template <array_kind_t kind>
inline void rlz<kind>::log_data_structure_sizes() const
{
    // SA_rlz is a component of an index; its parts are listed below it
    log_component("SA_rlz", size_in_bytes());
    log_component("reference", _reference.size_in_bytes(), 1);
    log_component("phrases", _phrase_starts.size_in_bytes(), 1);
    log_component("types", _PT.size_in_bytes(), 1);
    log_component("sources", _SR.size_in_bytes(), 1);
    log_component("literals", _LP.size_in_bytes(), 1);
}

#endif // RLZ_REF_RLZ_REF_RLZ_TPP

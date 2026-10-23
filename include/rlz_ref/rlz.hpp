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

#ifndef RLZ_REF_RLZ_REF_RLZ_HPP
#define RLZ_REF_RLZ_REF_RLZ_HPP

#include <bit>
#include <cstdint>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

#include <data_structures/elias_fano_bit_vector.hpp>
#include <data_structures/interleaved_bit_aligned_vectors.hpp>
#include <data_structures/plain_bit_vector.hpp>
#include <misc/files.hpp>
#include <misc/log.hpp>
#include <misc/utils.hpp>

/**
 * @brief builds the bit vector marking the starting positions of the phrases
 * @param lengths [0..z-1] phrase lengths
 * @param n length of the input (the sum of the phrase lengths)
 * @param threads number of threads to use
 * @return Elias-Fano coded bit vector of length n + 1 with ones at the z phrase starts and at position n
 */
inline elias_fano_bit_vector rlz_phrase_starts(const std::vector<uint64_t>& lengths, uint64_t n,
    uint64_t threads = 1)
{
    if (lengths.empty()) return elias_fano_bit_vector();
    std::vector<uint64_t> starts;
    starts.reserve(lengths.size() + 1);
    uint64_t pos = 0;

    for (uint64_t length : lengths) {
        starts.emplace_back(pos);
        pos += length;
    }

    starts.emplace_back(n);
    return elias_fano_bit_vector(starts, n + 1, threads);
}

/**
 * @brief the phrases of a relative Lempel-Ziv (RLZ) encoding with random access
 *
 * The input is parsed greedily into z phrases: a copy phrase is a longest prefix of the rest of the input that
 * occurs in the reference R, which consists of segments of the input; a literal phrase stores one symbol (if no
 * prefix of length at least 2 occurs in R).
 * @tparam reference_t type of R
 */
template <typename reference_t>
class rlz_phrases {
protected:
    // ############################# MEMBER VARIABLES #############################

    uint64_t n = 0; // n, the length of the input (including the terminator symbol)
    uint64_t z = 0; // z, the number of phrases
    uint64_t z_l = 0; // z_l, the number of literal phrases
    uint64_t z_c = 0; // z_c, the number of copy phrases

    /* R, the reference: the chosen segments of the input, concatenated in input order; the positions that no
       copy phrase copies are removed after the parse; [0..|R|-1] R[x] is a symbol of the input */
    reference_t _reference;
    // PT, the phrase types: PT[x] = 1 <=> phrase x is literal
    plain_bit_vector _PT;
    // bit vector of length n + 1 marking the starting positions of the phrases and position n (Elias-Fano)
    elias_fano_bit_vector _phrase_starts;
    // SR, the sources: SR[x] is the starting position in R of the x-th copy phrase, bit_width(|R|) bits each
    interleaved_bit_aligned_vectors _SR;
    // LP, the literal phrases: [0..z_l-1] LP[x] is the symbol of the x-th literal phrase
    interleaved_bit_aligned_vectors _LP;

    // ############################# INTERNAL METHODS #############################

    /**
     * @brief returns the length of a phrase
     * @param phrase [0..z-1] phrase index
     */
    inline uint64_t phrase_length(uint64_t phrase) const
    {
        const auto [beg, end] = _phrase_starts.select_1_pair(phrase + 1);
        return end - beg;
    }

    // a phrase together with its boundaries, used to iterate over consecutive phrases
    struct phrase_bounds_t {
        uint64_t phrase; // phrase index
        uint64_t beg; // starting position of the phrase
        uint64_t end; // starting position of the next phrase
    };

    /**
     * @brief returns the phrase containing a position, together with its boundaries
     * @param pos [0..n-1] position in the input
     */
    inline phrase_bounds_t phrase_bounds_at(uint64_t pos) const
    {
        const uint64_t phrase = _phrase_starts.rank_1(pos + 1) - 1;
        const auto [beg, end] = _phrase_starts.select_1_pair(phrase + 1);
        return { phrase, beg, end };
    }

    // advances bounds to the next phrase
    inline void next_phrase(phrase_bounds_t& bounds) const
    {
        bounds.phrase++;
        bounds.beg = bounds.end;
        bounds.end = bounds.phrase + 1 < z ? _phrase_starts.select_1(bounds.phrase + 2) : n;
    }

    // moves bounds back to the previous phrase (bounds must not be the first phrase)
    inline void prev_phrase(phrase_bounds_t& bounds) const
    {
        bounds.phrase--;
        bounds.end = bounds.beg;
        bounds.beg = _phrase_starts.select_1(bounds.phrase + 1);
    }

public:
    // ############################# CONSTRUCTORS #############################

    rlz_phrases() = default;

    /**
     * @brief constructs the phrases from their (already built) components
     * @param n length of the input (including the terminator symbol)
     * @param z number of phrases
     * @param z_l number of literal phrases
     * @param z_c number of copy phrases
     * @param reference R
     * @param PT the phrase types
     * @param phrase_starts the starting positions of the phrases (see rlz_phrase_starts)
     * @param SR the starting positions in R of the copy phrases
     * @param LP the literal phrases
     */
    rlz_phrases(
        uint64_t n, uint64_t z, uint64_t z_l, uint64_t z_c,
        reference_t&& reference,
        plain_bit_vector&& PT,
        elias_fano_bit_vector&& phrase_starts,
        interleaved_bit_aligned_vectors&& SR,
        interleaved_bit_aligned_vectors&& LP)
        : n(n), z(z), z_l(z_l), z_c(z_c)
        , _reference(std::move(reference))
        , _PT(std::move(PT))
        , _phrase_starts(std::move(phrase_starts))
        , _SR(std::move(SR))
        , _LP(std::move(LP))
    { }

    // ############################# ACCESS METHODS #############################

    // returns whether the rlz is empty (i.e, has not been built)
    inline bool empty() const { return z == 0; }

    // returns the length of the input (including the terminator symbol)
    inline uint64_t size() const { return n; }

    // returns the number of phrases in the rlz
    inline uint64_t num_phrases() const { return z; }

    // returns the number of literal phrases in the rlz
    inline uint64_t num_literal_phrases() const { return z_l; }

    // returns the number of copy phrases in the rlz
    inline uint64_t num_copy_phrases() const { return z_c; }

    // returns R
    inline const reference_t& reference() const { return _reference; }

    // returns PT
    inline const plain_bit_vector& PT() const { return _PT; }

    // returns SR
    inline const interleaved_bit_aligned_vectors& SR() const { return _SR; }

    // returns LP
    inline const interleaved_bit_aligned_vectors& LP() const { return _LP; }

    /**
     * @brief returns R[x]
     * @param x [0..|R|-1] index in R
     */
    inline uint64_t reference(uint64_t x) const { return _reference[x]; }

    // returns |R|
    inline uint64_t reference_length() const { return _reference.size(); }

    /**
     * @brief returns PT[x]
     * @param x [0..z-1] index in PT
     */
    inline bool PT(uint64_t x) const { return _PT[x]; }

    /**
     * @brief returns SR[x]
     * @param x [0..z_c-1] index in SR
     */
    inline uint64_t SR(uint64_t x) const { return _SR[x]; }

    /**
     * @brief returns LP[x]
     * @param x [0..z_l-1] index in LP
     */
    inline uint64_t LP(uint64_t x) const { return _LP[x]; }

    // ############################# MISC METHODS #############################

    // returns the size of the data structure in bytes
    uint64_t size_in_bytes() const
    {
        return _reference.size_in_bytes() + _phrase_starts.size_in_bytes() + _SR.size_in_bytes()
            + _LP.size_in_bytes() + _PT.size_in_bytes();
    }

    /**
     * @brief logs the data structure sizes to the output stream out
     * @param out an output stream
     * @param suffix suffix appended to each measurement key
     */
    void log_data_structure_sizes(std::ostream& out, std::string suffix = "") const
    {
        out << " size_reference" << suffix << "=" << _reference.size_in_bytes();
        out << " size_phrases" << suffix << "=" << _phrase_starts.size_in_bytes();
        out << " size_sources" << suffix << "=" << _SR.size_in_bytes();
        out << " size_literals" << suffix << "=" << _LP.size_in_bytes();
        out << " size_pt" << suffix << "=" << _PT.size_in_bytes();
    }

    // ############################# SERIALIZATION METHODS #############################

    /**
     * @brief stores the phrases to an output stream
     * @param out output stream to store the phrases to
     */
    void serialize(std::ostream& out) const
    {
        out.write((char*) &n, sizeof(uint64_t));
        out.write((char*) &z, sizeof(uint64_t));
        out.write((char*) &z_l, sizeof(uint64_t));
        out.write((char*) &z_c, sizeof(uint64_t));

        _reference.serialize(out);
        _phrase_starts.serialize(out);
        _SR.serialize(out);
        _LP.serialize(out);
        _PT.serialize(out);
    }

    /**
     * @brief reads serialized phrases from an input stream
     * @param in an input stream storing serialized phrases
     */
    void load(std::istream& in)
    {
        in.read((char*) &n, sizeof(uint64_t));
        in.read((char*) &z, sizeof(uint64_t));
        in.read((char*) &z_l, sizeof(uint64_t));
        in.read((char*) &z_c, sizeof(uint64_t));

        _reference.load(in);
        _phrase_starts.load(in);
        _SR.load(in);
        _LP.load(in);
        _PT.load(in);
    }
};

/**
 * @brief relative Lempel-Ziv (RLZ) encoding with random access of a differentially encoded integer sequence (e.g.,
 *        the differential suffix array SA^d), whose values are decoded sequentially with a decode_context_t
 *
 * R stores values SA^d[j] = SA[j] - SA[j-1] + n (the offset n makes the differences non-negative), value_width(n)
 * bits each; PT has z + 2 entries (PT[z] = PT[z+1] = 0) with rank_1, select_0 and select_1 support; SR has z_c + 1
 * entries (SR[z_c] = |R|); LP stores the values of SA^d of the literal phrases (LP[0] = SA[0] + n), value_width(n)
 * bits each.
 * @tparam kind _diff_access
 */
template <array_kind_t kind>
class rlz : public rlz_phrases<interleaved_bit_aligned_vectors> {
    static_assert(kind == _diff_access, "rlz supports only the kind _diff_access");

public:
    // ############################# TYPES AND CONSTANTS #############################

    // type of the reference: fixed-width integers
    using reference_t = interleaved_bit_aligned_vectors;

    /**
     * @brief returns the number of bits used to store one differential value
     * @param n length of the input (including the terminator symbol)
     * @return number of bits per value
     */
    static constexpr uint8_t value_width(uint64_t n) { return std::bit_width(2 * n + 1); }

    // ############################# CONSTRUCTORS #############################

    // constructs an empty rlz or one from its (already built) components (see rlz_phrases)
    using rlz_phrases<reference_t>::rlz_phrases;

    // ############################# MISC METHODS #############################

    // logs the contents of the rlz (only useful for small inputs)
    void log_data_structures() const;

    // logs the sizes of the data structures: the component SA_rlz and its parts (see log_component)
    void log_data_structure_sizes() const;

    using rlz_phrases<reference_t>::log_data_structure_sizes;

    // ############################# QUERY METHODS #############################

    /**
     * @brief stores the variables needed to decode consecutive suffix-array values from the rlz
     *
     * A decode context stores a position i in the suffix array and the value s = SA[i]. It is prepared to
     * decode either to the right (after init_right(), turn_right() or a step to the right with next(),
     * skip_right() or report_right()) or to the left (after init_left(), turn_left() or a step to the left
     * with prev(), skip_left() or report_left()). i and s can be read with pos() and value() and set with
     * set_pos() and set_value().
     */
    struct decode_context_t {
      protected:
        const rlz* parsing = nullptr; // the rlz to decode from

        uint64_t i; // current position in the suffix array
        uint64_t s; // current suffix s = SA[i]

        uint64_t x_p;  // index of the phrase containing i
        uint64_t x_lp; // literal-phrase index of the current or next literal phrase
        uint64_t x_cp; // copy-phrase index of the current or next copy phrase
        uint64_t x_r;  // position in R inside the current copy phrase (or the starting position in R of the current or next copy phrase)
        uint64_t s_np; // starting position of the next phrase (of the current phrase when decoding to the left)

        /**
         * @brief returns the length of a copy phrase
         * @param x_cp [0..z_c-1] copy-phrase index
         */
        inline uint64_t copy_length(uint64_t x_cp) const
        {
            return parsing->phrase_length(parsing->PT().select_0(x_cp + 1));
        }

        // returns the length of the phrase x_p; if it is a copy phrase, x_cp has to be its copy-phrase index
        inline uint64_t length_of_phrase_x_p() const
        {
            return parsing->PT(x_p) ? 1 : copy_length(x_cp);
        }

        /**
         * @brief moves x_p, x_cp, x_r and s_np to the previous phrase, after a step to the left has moved i
         *        from the first position of the copy phrase x_p to the last position of the previous phrase
         */
        inline void enter_previous_phrase_from_copy()
        {
            x_p--;

            if (x_cp > 0) [[likely]] {
                x_cp--;
                x_r = parsing->SR(x_cp) + copy_length(x_cp) - 1;
            }

            s_np -= length_of_phrase_x_p();
        }

        /**
         * @brief moves x_p, x_lp and s_np to the previous phrase, after a step to the left has moved i from
         *        the literal phrase x_p to the previous phrase
         */
        inline void enter_previous_phrase_from_literal()
        {
            x_p--;
            x_lp--;
            if (i > 0) [[likely]] s_np -= length_of_phrase_x_p();
        }

        /**
         * @brief moves x_p, x_cp, x_r and s_np to the next phrase, after a step to the right has moved i from
         *        the last position of the copy phrase x_p to the first position of the next phrase (i < n)
         */
        inline void enter_next_phrase_from_copy()
        {
            x_p++;
            x_cp++;
            x_r = parsing->SR(x_cp);
            s_np += length_of_phrase_x_p();
        }

        /**
         * @brief moves x_p, x_lp and s_np to the next phrase, after a step to the right has moved i from the
         *        literal phrase x_p to the next phrase
         */
        inline void enter_next_phrase_from_literal()
        {
            x_p++;
            x_lp++;
            if (i < parsing->n) [[likely]] s_np += length_of_phrase_x_p();
        }

        /**
         * @brief adds (dir = RIGHT) or subtracts (dir = LEFT) the differential value vec[x] to/from s
         * @tparam dir decoding direction
         * @param vec R or LP
         * @param x position in vec
         */
        template <direction_t dir>
        inline void add_difference(const interleaved_bit_aligned_vectors& vec, uint64_t x)
        {
            uint64_t value = vec[x];

            // the values are stored with an offset of n (the differences can be negative)
            if constexpr (dir == RIGHT) { s += value; s -= parsing->n; }
            else                        { s += parsing->n; s -= value; }
        }

      public:
        decode_context_t() = default;

        // constructs a decode context for the rlz enc
        decode_context_t(const rlz& enc)
            : parsing(&enc)
        { }

        // returns the current position i in the suffix array
        uint64_t pos() const { return i; }

        // sets the current position i in the suffix array to p
        inline void set_pos(uint64_t p) { i = p; }

        // returns the current suffix-array value s
        uint64_t value() const { return s; }

        // sets the current suffix-array value s to v (e.g., to seed it before decoding)
        inline void set_value(uint64_t v) { s = v; }

        /**
         * @brief prepares the context to decode SA[i] and the following values to the right; s has to be set
         *        to SA[i-1] with set_value() before decoding
         * @param i position in the suffix array
         */
        inline void init_right(uint64_t i);

        /**
         * @brief prepares the context to decode SA[i-1] and the preceding values to the left (equivalent to
         *        init_right(i) followed by turn_left()); s has to be set to SA[i] with set_value() before
         *        decoding
         * @param i position in the suffix array
         */
        inline void init_left(uint64_t i);

        /**
         * @brief changes the decoding direction of a context prepared to decode to the right to the left, in
         *        O(1) time; pos() is unchanged; s has to be set to SA[pos()] with set_value(); inverse of
         *        turn_right()
         */
        inline void turn_left();

        /**
         * @brief changes the decoding direction of a context prepared to decode to the left to the right, in
         *        O(1) time; pos() is unchanged; s has to be set to SA[pos()-1] with set_value(); inverse of
         *        turn_left()
         */
        inline void turn_right();

        /**
         * @brief advances the context to the right up to position e without reporting the values in between;
         *        the context has to be prepared to decode to the right with s = SA[i-1]; afterwards, i = e and
         *        s = SA[e-1], i.e., a following next() or report_right() returns SA[e]
         * @param e position in the suffix array to advance to (e >= i)
         */
        inline void skip_right(uint64_t e);

        /**
         * @brief advances the context to the left down to position b without reporting the values in between;
         *        the context has to be prepared to decode to the left; afterwards, i = b and s = SA[b], i.e.,
         *        a following prev() or report_left() returns SA[b-1]
         * @param b position in the suffix array to advance to (b <= i)
         */
        inline void skip_left(uint64_t b);

        /**
         * @brief decodes SA[i-1], stores it in s, decrements i and returns s; the context has to be prepared to
         *        decode to the left
         */
        inline uint64_t prev();

        /**
         * @brief decodes SA[i], stores it in s, increments i and returns s; the context has to be prepared to
         *        decode to the right
         */
        inline uint64_t next();

        /**
         * @brief decodes SA[b..i-1] from right to left and reports each value; the context has to be prepared
         *        to decode to the left
         * @param b leftmost position in the suffix array to decode
         * @param report function called as report(j, SA[j]) for each j in [b..i-1], from right to left
         */
        template <typename report_fnc_t>
        inline void report_left(uint64_t b, report_fnc_t report);

        /**
         * @brief decodes SA[i..e] from left to right and reports each value; the context has to be prepared to
         *        decode to the right
         * @param e rightmost position in the suffix array to decode
         * @param report function called as report(j, SA[j]) for each j in [i..e], from left to right
         */
        template <typename report_fnc_t>
        inline void report_right(uint64_t e, report_fnc_t report);
    };

    // returns a decode context for the rlz
    inline decode_context_t decode_context() const { return decode_context_t(*this); }

    // ############################# CONSTRUCTION METHODS #############################

    /**
     * @brief constructs the rlz of SA^d (see construction/construction.hpp)
     * @tparam sad_func_t type of the function returning the values of SA^d, callable as sad_func_t(i_p, i)
     *         for thread i_p and position i
     * @tparam intervals_t type of the disjoint interval sequence of Phi^{-1}
     */
    template <typename sad_func_t = std::monostate, typename intervals_t = std::monostate> class construction;
};

#include <rlz_ref/rlz.tpp>
#include <rlz_ref/queries.tpp>

#endif // RLZ_REF_RLZ_REF_RLZ_HPP

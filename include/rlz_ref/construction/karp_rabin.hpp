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

#ifndef RLZ_REF_RLZ_REF_CONSTRUCTION_KARP_RABIN_HPP
#define RLZ_REF_RLZ_REF_CONSTRUCTION_KARP_RABIN_HPP

#include <cstdint>

#include <misc/hashing.hpp>
#include <misc/utils.hpp>

/**
 * @brief Karp-Rabin fingerprints of k-mers modulo the Mersenne prime p = 2^61 - 1 (see mersenne_61)
 *
 * The fingerprint of a k-mer s[0..k-1] is (s[0] * base^(k-1) + ... + s[k-1] * base^0) mod p, where each
 * symbol s[i] is an integer in [0..p-1]. The k-mers of a sequence are reported as hash_64 of their
 * fingerprints.
 */
struct rlz_karp_rabin {
    static constexpr uint64_t base = 0x1F2E3D4C5B6A7988ULL % mersenne_61::prime;

    // returns base^(k-1) mod p, the factor of the first symbol of a k-mer in its fingerprint (1 for k = 0)
    static inline uint64_t first_symbol_factor(uint64_t k)
    {
        uint64_t factor = 1;
        for (uint64_t i = 1; i < k; i++) factor = mersenne_61::reduce_wide(uint128_t(factor) * base);
        return factor;
    }

    /**
     * @brief reports the hashed fingerprints of the k-mers of a sequence of symbols, from left to right, in
     *        O(length) time
     * @param length length of the sequence; nothing is reported if length < k
     * @param k length of the k-mers
     * @param symbol symbol(i) returns the i-th symbol of the sequence, an integer in [0..p-1]
     * @param first_summand first_summand(i) returns symbol(i) * base^(k-1) mod p, the summand of
     *        the first symbol in the fingerprint of the k-mer starting at position i
     * @param report function called as report(i, h) for each k-mer, where i is its starting position and h
     *        the hashed fingerprint
     */
    template <typename symbol_t, typename first_summand_t, typename report_t>
    static void for_each_kmer(uint64_t length, uint64_t k, symbol_t&& symbol, first_summand_t&& first_summand,
        report_t&& report)
    {
        if (length < k) return;
        uint64_t fingerprint = 0;
        for (uint64_t i = 0; i < k; i++)
            fingerprint = mersenne_61::reduce_wide(uint128_t(fingerprint) * base + symbol(i));

        for (uint64_t i = 0;; i++) {
            report(i, hash_64(fingerprint));
            if (i + k >= length) break;

            // remove the first symbol of the k-mer and append the next one
            fingerprint = mersenne_61::reduce_wide(
                uint128_t(fingerprint + mersenne_61::prime - first_summand(i)) * base + symbol(i + k));
        }
    }

    /**
     * @brief reports the hashed fingerprints of the k-mers of a sequence of symbols that are divisible by
     *        divisor (see for_each_kmer)
     * @param length length of the sequence
     * @param k length of the k-mers
     * @param divisor only the hashed fingerprints divisible by divisor are reported
     * @param symbol symbol(i) returns the i-th symbol of the sequence, an integer in [0..p-1]
     * @param report function called as report(h) for each reported hashed fingerprint h
     */
    template <typename symbol_t, typename report_t>
    static void for_each_sampled_kmer(uint64_t length, uint64_t k, uint64_t divisor, symbol_t&& symbol,
        report_t&& report)
    {
        if (length < k) return;
        const uint64_t factor = first_symbol_factor(k);

        for_each_kmer(length, k, symbol,
            [&](uint64_t i) { return mersenne_61::reduce_wide(uint128_t(symbol(i)) * factor); },
            [&](uint64_t, uint64_t hashed) { if (hashed % divisor == 0) report(hashed); });
    }
};

#endif // RLZ_REF_RLZ_REF_CONSTRUCTION_KARP_RABIN_HPP

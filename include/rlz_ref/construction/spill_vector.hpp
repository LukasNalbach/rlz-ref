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

#ifndef RLZ_REF_RLZ_REF_CONSTRUCTION_SPILL_VECTOR_HPP
#define RLZ_REF_RLZ_REF_CONSTRUCTION_SPILL_VECTOR_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

#include <misc/files.hpp>
#include <misc/platform.hpp>

/**
 * @brief a sequence of integers of a fixed bit width in a temporary file: the values are appended with push_back, then
 *        read with operator[]; the file is written and read with O_DIRECT through one buffer (if the file system does
 *        not support O_DIRECT, through the page cache, whose pages of the file are dropped) and removed by remove() or
 *        the destructor. One object is used by one thread at a time.
 */
class rlz_spill_vector {
private:
    static constexpr uint64_t sector = direct_io_alignment;
    static constexpr uint64_t word_bits = 64;
    static constexpr uint64_t sector_words = sector / sizeof(uint64_t);

    std::string path; // path of the file (empty: removed)
    uint8_t width = 0; // number of bits per value
    uint64_t num_values = 0; // number of values appended
    int fd = -1; // file descriptor (-1 if the file is closed)
    bool direct = false; // true <=> the file is opened with O_DIRECT
    bool writing = true; // true <=> values are appended (false after the first read)
    uint64_t* buffer = nullptr; // the buffer, buffer_words words, aligned to sector
    uint64_t buffer_words = 0; // the size of the buffer in words (a multiple of sector_words)
    // writing: the words [0..num_words-1] of the buffer are the words of the file starting at flushed_words
    // reading: the words [0..buffer_end-buffer_beg-1] of the buffer are the words [buffer_beg, buffer_end) of the file
    uint64_t num_words = 0;
    uint64_t flushed_words = 0;
    uint64_t buffer_beg = 0;
    uint64_t buffer_end = 0;
    uint64_t file_words = 0; // number of words in the file (reading)
    uint64_t pending = 0; // the lowest pending_bits bits are the bits appended after the last complete word
    uint64_t pending_bits = 0;

    // writes the complete words in the buffer to the file
    void flush_words()
    {
        uint64_t bytes = num_words * sizeof(uint64_t);
        if (direct) bytes = div_ceil(bytes, sector) * sector;
        if (bytes != 0 && !pwrite_fully(fd, (const char*) buffer, bytes, flushed_words * sizeof(uint64_t)))
            throw std::runtime_error("cannot write " + path);
        if (!direct) platform::drop_cached(fd, flushed_words * sizeof(uint64_t), bytes);
        flushed_words += num_words;
        num_words = 0;
    }

    // appends a complete word
    void append_word(uint64_t word)
    {
        buffer[num_words++] = word;
        if (num_words == buffer_words) flush_words();
    }

    // writes the last bits, truncates the file to its words and opens it for reading
    void finish_writing()
    {
        if (pending_bits != 0) append_word(pending);
        pending_bits = 0;
        // the bytes after the last word in the last sector are zero
        std::memset(buffer + num_words, 0, (buffer_words - num_words) * sizeof(uint64_t));
        flush_words();
        file_words = flushed_words;
        const bool truncated = platform::set_size(fd, file_words * sizeof(uint64_t));
        platform::close(fd);
        if (!truncated) throw std::runtime_error("cannot write " + path);
        fd = open_direct_or_buffered(path, platform::file_access::read, direct);
        if (fd < 0) throw std::runtime_error("cannot read " + path);
        if (!direct) advise_uncached(fd);
        writing = false;
    }

    // returns the word w of the file; reads the buffer from the sector containing it, if it is not in the buffer
    uint64_t word(uint64_t w)
    {
        if (w < buffer_beg || w >= buffer_end) [[unlikely]] {
            buffer_beg = (w / sector_words) * sector_words;
            const uint64_t num = std::min(buffer_words, file_words - buffer_beg);
            const uint64_t bytes = num * sizeof(uint64_t);
            const uint64_t read = pread_fully(fd, buffer, direct ? div_ceil(bytes, sector) * sector : bytes,
                buffer_beg * sizeof(uint64_t));
            if (read < bytes) throw std::runtime_error("cannot read " + path);
            if (!direct) platform::drop_cached(fd, buffer_beg * sizeof(uint64_t), bytes);
            buffer_end = buffer_beg + num;
        }

        return buffer[w - buffer_beg];
    }

public:
    rlz_spill_vector() = default;

    /**
     * @brief creates the file
     * @param file_path path of the file
     * @param bits number of bits per value (at most 64)
     * @param buffer_bytes size of the buffer in bytes (rounded up to a multiple of the sector size)
     */
    rlz_spill_vector(const std::string& file_path, uint8_t bits, uint64_t buffer_bytes)
        : path(file_path), width(bits),
          buffer_words(std::max<uint64_t>(1, div_ceil(buffer_bytes, sector)) * sector_words)
    {
        fd = open_direct_or_buffered(path, platform::file_access::write, direct);
        if (fd < 0) throw std::runtime_error("cannot create " + path);
        if (!direct) advise_uncached(fd);
        buffer = (uint64_t*) allocate_aligned(buffer_words * sizeof(uint64_t));
    }

    rlz_spill_vector(rlz_spill_vector&& other) noexcept { *this = std::move(other); }

    rlz_spill_vector& operator=(rlz_spill_vector&& other) noexcept
    {
        if (this == &other) return *this;
        remove();
        path = std::exchange(other.path, std::string());
        width = other.width;
        num_values = other.num_values;
        fd = std::exchange(other.fd, -1);
        direct = other.direct;
        writing = other.writing;
        buffer = std::exchange(other.buffer, nullptr);
        buffer_words = other.buffer_words;
        num_words = other.num_words;
        flushed_words = other.flushed_words;
        buffer_beg = other.buffer_beg;
        buffer_end = other.buffer_end;
        file_words = other.file_words;
        pending = other.pending;
        pending_bits = other.pending_bits;
        return *this;
    }

    rlz_spill_vector(const rlz_spill_vector&) = delete;
    rlz_spill_vector& operator=(const rlz_spill_vector&) = delete;

    ~rlz_spill_vector() { remove(); }

    // returns the number of values
    uint64_t size() const { return num_values; }

    // appends a value (precondition: no value has been read)
    void push_back(uint64_t value)
    {
        num_values++;
        if (width == 0) return;
        if (width < word_bits) value &= (uint64_t(1) << width) - 1;
        pending |= value << pending_bits;
        const uint64_t bits = pending_bits + width;

        if (bits < word_bits) {
            pending_bits = bits;
            return;
        }

        append_word(pending);
        pending = pending_bits == 0 ? 0 : value >> (word_bits - pending_bits);
        pending_bits = bits - word_bits;
    }

    // returns the i-th value; the first read ends the appending
    uint64_t operator[](uint64_t i)
    {
        if (writing) [[unlikely]] finish_writing();
        if (width == 0) return 0;
        const uint64_t bit = i * width;
        const uint64_t offset = bit % word_bits;
        uint64_t value = word(bit / word_bits) >> offset;
        if (offset + width > word_bits) value |= word(bit / word_bits + 1) << (word_bits - offset);
        return width < word_bits ? value & ((uint64_t(1) << width) - 1) : value;
    }

    // closes and removes the file and frees the buffer
    void remove()
    {
        if (fd >= 0) platform::close(fd);
        fd = -1;
        if (buffer != nullptr) platform::free_aligned(buffer);
        buffer = nullptr;
        std::error_code ec;
        if (!path.empty()) std::filesystem::remove(path, ec);
        path.clear();
    }
};

#endif // RLZ_REF_RLZ_REF_CONSTRUCTION_SPILL_VECTOR_HPP

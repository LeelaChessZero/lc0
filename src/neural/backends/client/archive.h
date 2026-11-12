/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "utils/expected.h"

namespace lczero::client {

// Possible errors during serialization/deserialization.
// They are returned using Unexpected type.
enum class ArchiveError {
  None,
  BufferOverflow,
  InvalidData,
  ValueOverflow,
  SizeCalculationFailed,
  UnknownType,
  RemoteError,
  OutOfMemory,
};

// ArchiveError logging support.
std::ostream& operator<<(std::ostream& os, ArchiveError error);

// Wrapper type to require fixed-width integer serialization. Fixed-width is
// better when varaible has often highest bits set which would require more
// space in variable-width base 128 encoding.
template <typename T>
struct FixedInteger {
  FixedInteger(T& v) : value(v) {}
  T& value;
};

// Wrapper type to add size limits to vector serialization. Used to prevent
// malicious or corrupted data from causing large memory allocations.
template <typename T>
struct VectorLimits {
  VectorLimits(std::vector<T>& v, size_t min_size, size_t max_size)
      : value(v), min_size_(min_size), max_size_(max_size) {}
  std::vector<T>& value;
  size_t min_size_;
  size_t max_size_;
};

// Default serialization implementation that calls Serialize method of the
// object. Argument-dependent lookup allows overloading this function for types
// which cannot have new member functions.
template <typename T, typename Archive>
typename Archive::ResultType Serialize(T& value, Archive& ar,
                                       const unsigned version) {
  return value.Serialize(ar, version);
}

// Binary serialization archive. It uses base-128 variable length encoding for
// integers. Floating point numbers use fixed width integer serializartion.
class BinaryOArchive {
 public:
  // Boost serialization compatibility flags which can be used to implement
  // different logic for saving and loading.
  static constexpr bool is_saving = true;
  static constexpr bool is_loading = false;

  // Result type uses std::expected fallback implementation to signal errors.
  using ResultType = Expected<BinaryOArchive&, ArchiveError>;

  BinaryOArchive(unsigned version) : version_(version) {}

  // Boost serialization compatibility operator.
  template <typename T>
  [[nodiscard]]
  ResultType operator<<(const T& value) {
    return Save(value);
  }

  // Serialization operator which follows Boost serialization convention.
  template <typename T>
  [[nodiscard]]
  ResultType operator&(const T& value) {
    return *this << value;
  }

  // Helper to add message size to header when starting serialization.
  template <typename T>
  [[nodiscard]]
  ResultType StartSerialize(T& value);

  // Returns the size of the serialized data.
  size_t Size() const { return buffer_.size(); }

  // Returns the buffer which can be sent over the network.
  const std::vector<char>& GetVector() const { return buffer_; }

 protected:
  // Generic serialization function which calls Serialize function. Serialize
  // function can be overloaded using argument-dependent lookup. The default
  // implementation calls Serialize method of the object.
  template <typename T>
  ResultType Save(const T& value) {
    return Serialize(const_cast<T&>(value), *this, version_);
  }

  // Integer serialization using protobuf variable encoding.
  ResultType Save(const uint64_t& value);
  ResultType Save(const int64_t& value);
  ResultType Save(const uint32_t& value);
  ResultType Save(const int32_t& value);
  ResultType Save(const uint16_t& value);
  ResultType Save(const int16_t& value);
  ResultType Save(const uint8_t& value);
  ResultType Save(const int8_t& value);

  ResultType Save(const bool& value);

  // Fixed width integer serialization.
  ResultType Save(const FixedInteger<uint64_t>& value);
  ResultType Save(const FixedInteger<int64_t>& value);
  ResultType Save(const FixedInteger<uint32_t>& value);
  ResultType Save(const FixedInteger<int32_t>& value);
  ResultType Save(const FixedInteger<uint16_t>& value);
  ResultType Save(const FixedInteger<int16_t>& value);
  ResultType Save(const FixedInteger<uint8_t>& value);
  ResultType Save(const FixedInteger<int8_t>& value);

  // Floating point serialization
  // TODO: support lower precision floats.
  ResultType Save(const double& value);
  ResultType Save(const float& value);

  // Container serialization
  template <typename T>
  ResultType Save(const VectorLimits<T>& wrapper) {
    const std::vector<T>& container = wrapper.value;
    uint64_t size = container.size();
    auto r = Save(size);
    for (const T& item : container) {
      r = r.and_then([&item](BinaryOArchive& ar) { return ar.Save(item); });
    }
    return r;
  }
  template <typename T>
  ResultType Save(const std::span<T>& container) {
    uint64_t size = container.size();
    ResultType r = Save(size);
    if (!r) return r;
    for (const T& item : container) {
      if (!(r = Save(item))) return r;
    }
    return r;
  }

  // String serialization
  ResultType Save(const std::string_view& value);

 private:
  std::vector<char> buffer_;
  unsigned version_;
};

class BinaryIArchive {
 public:
  static constexpr bool is_saving = false;
  static constexpr bool is_loading = true;

  using ResultType = Expected<BinaryIArchive&, ArchiveError>;

  // Deserialization archive from a network buffer.
  // @param buffer Buffer to deserialize from.
  // @param temporary_memory Used to deserialize content inside spans. It can be
  //                         point to buffer is spans don't have variable
  //                         ingers.
  // @param temporary_memory_size Size of the temporary memory buffer in bytes.
  //                             If zero, then using input buffer.
  // @param version Serialization version to use.
  BinaryIArchive(std::span<const char> buffer, char* temporary_memory,
                 size_t temporary_memory_size, unsigned version)
      : buffer_(buffer),
        temporary_memory_(temporary_memory),
        temporary_memory_end_(temporary_memory_size == 0
                                  ? nullptr
                                  : temporary_memory + temporary_memory_size),
        version_(version) {}

  // Boost serialization compatibility operator.
  template <typename T>
  [[nodiscard]]
  ResultType operator>>(T& value) {
    return Load(value);
  }
  // Specialization for FixedInteger wrappers. Parameter is passed by value to
  // allow passing temporary FixedInteger objects.
  template <typename T>
  [[nodiscard]]
  ResultType operator>>(FixedInteger<T> value) {
    return Load(value);
  }
  template <typename T>
  [[nodiscard]]
  ResultType operator>>(VectorLimits<T> value) {
    return Load(value);
  }

  // Serialization operator which follows Boost serialization convention.
  template <typename T>
  [[nodiscard]]
  ResultType operator&(T& value) {
    return *this >> value;
  }
  // Specialization for FixedInteger wrappers. Parameter is passed by value to
  // allow passing temporary FixedInteger objects.
  template <typename T>
  [[nodiscard]]
  ResultType operator&(FixedInteger<T> value) {
    return *this >> value;
  }
  template <typename T>
  [[nodiscard]]
  ResultType operator&(VectorLimits<T> value) {
    return *this >> value;
  }

  // Returns remaining buffer size to deserialize.
  size_t Size() const { return buffer_.size(); }

 protected:
  // Generic deserialization function which calls Serialize function. Serialize
  // function can be overloaded using argument-dependent lookup. The default
  // implementation calls Serialize method of the object.
  template <typename T>
  ResultType Load(T& value) {
    return Serialize(value, *this, version_);
  }

  // Integer deserialization using protobuf variable encoding.
  ResultType Load(uint64_t& value);
  ResultType Load(int64_t& value);
  ResultType Load(uint32_t& value);
  ResultType Load(int32_t& value);
  ResultType Load(uint16_t& value);
  ResultType Load(int16_t& value);
  ResultType Load(uint8_t& value);
  ResultType Load(int8_t& value);

  ResultType Load(bool& value);

  // Fixed width integer deserialization.
  ResultType Load(FixedInteger<uint64_t> value);
  ResultType Load(FixedInteger<int64_t> value);
  ResultType Load(FixedInteger<uint32_t> value);
  ResultType Load(FixedInteger<int32_t> value);
  ResultType Load(FixedInteger<uint16_t> value);
  ResultType Load(FixedInteger<int16_t> value);
  ResultType Load(FixedInteger<uint8_t> value);
  ResultType Load(FixedInteger<int8_t> value);

  // Floating point deserialization
  ResultType Load(double& value);
  ResultType Load(float& value);
  // Container deserialization
  template <typename T>
  ResultType Load(VectorLimits<T> wrapper) {
    auto& container = wrapper.value;
    uint64_t size = 0;
    ResultType r = Load(size);
    if (!r) return r;
    if (size < wrapper.min_size_ || size > wrapper.max_size_) {
      return Unexpected(ArchiveError::ValueOverflow);
    }
    container.resize(size);
    for (T& item : container) {
      if (!(r = Load(item))) return r;
    }
    return r;
  }

  // Helper to allocate span from temporary memory.
  // Temporary memory must be large enough to hold all deserialized spans.
  template <typename T>
  std::span<T> AllocateSpan(size_t size) {
    T* ptr = reinterpret_cast<T*>(temporary_memory_);
    temporary_memory_ += size * sizeof(T);
    if (temporary_memory_end_ != nullptr &&
        temporary_memory_ > temporary_memory_end_) {
      ptr = nullptr;
      size = 0;
    }
    return {ptr, size};
  }

  // Span deserialization using temporary memory. Temporary memory must be
  // preallocated to hold all deserialized items. Using input buffer is possible
  // if items won't exceed already parse input buffer size. Practically this
  // means items cannot have variable length integer members.
  template <typename T>
  ResultType Load(std::span<T>& container) {
    uint64_t size = 0;
    ResultType r = Load(size);
    if (!r) return r;
    container = AllocateSpan<T>(size);
    // Check if caller provided enough temporary memory.
    // case: temporary_memory_size != 0
    if (container.data() == nullptr) {
      return Unexpected(ArchiveError::OutOfMemory);
    }
    for (T& item : container) {
      // Check that item won't overwrite input data before it is parsed.
      // case: temporary_memory_size == 0
      if (temporary_memory_end_ == nullptr &&
          buffer_.data() < reinterpret_cast<const char*>(&item + 1)) {
        return Unexpected(ArchiveError::OutOfMemory);
      }
      if (!(r = Load(item))) return r;
    }
    return r;
  }

  // String deserialization
  ResultType Load(std::string_view& value);

 private:
  std::span<const char> buffer_;
  char* temporary_memory_;
  char* temporary_memory_end_;
  unsigned version_;
};

}  // namespace lczero::client

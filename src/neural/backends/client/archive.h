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

enum class ArchiveError {
  None,
  BufferOverflow,
  InvalidData,
  ValueOverflow,
  SizeCalculationFailed,
  UnknownType,
  RemoteError
};

std::ostream& operator<<(std::ostream& os, ArchiveError error);

template <typename T>
struct FixedInteger {
  T& value;
};

template <typename T, typename Archive>
Archive::ResultType Serialize(T& value, Archive& ar, const unsigned version) {
  return value.Serialize(ar, version);
}

class BinaryOArchive {
 public:
  static constexpr bool is_saving = true;
  static constexpr bool is_loading = false;

  using ResultType = Expected<BinaryOArchive&, ArchiveError>;

  BinaryOArchive(unsigned version) : version_(version) {}

  template <typename T>
  [[nodiscard]]
  ResultType operator<<(const T& value) {
    return Save(value);
  }

  template <typename T>
  [[nodiscard]]
  ResultType operator&(const T& value) {
    return *this << value;
  }

  template <typename T>
  [[nodiscard]]
  ResultType StartSerialize(T& value);

  size_t Size() const { return buffer_.size(); }

  const std::vector<char>& GetVector() const { return buffer_; }

 protected:
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
  ResultType Save(const std::vector<T>& container) {
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

  BinaryIArchive(std::span<const char> buffer, char* temporary_memory, unsigned version)
      : buffer_(buffer), temporary_memory_(temporary_memory), version_(version) {}

  template <typename T>
  [[nodiscard]]
  ResultType operator>>(T& value) {
    return Load(value);
  }
  template <typename T>
  [[nodiscard]]
  ResultType operator>>(FixedInteger<T> value) {
    return Load(value);
  }

  template <typename T>
  [[nodiscard]]
  ResultType operator&(T& value) {
    return *this >> value;
  }
  template <typename T>
  [[nodiscard]]
  ResultType operator&(FixedInteger<T> value) {
    return *this >> value;
  }

  size_t Size() const { return buffer_.size(); }

 protected:
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
  ResultType Load(std::vector<T>& container) {
    uint64_t size = 0;
    ResultType r = Load(size);
    if (!r) return r;
    container.resize(size);
    for (T& item : container) {
      if (!(r = Load(item))) return r;
    }
    return r;
  }
  template <typename T>
  std::span<T> AllocateSpan(size_t size) {
    T* ptr = reinterpret_cast<T*>(temporary_memory_);
    temporary_memory_ += size * sizeof(T);
    return {ptr, size};
  }

  template <typename T>
  ResultType Load(std::span<T>& container) {
    uint64_t size = 0;
    ResultType r = Load(size);
    if (!r) return r;
    container = AllocateSpan<T>(size);
    for (T& item : container) {
      if (!(r = Load(item))) return r;
    }
    return r;
  }

  // String deserialization
  ResultType Load(std::string_view& value);

 private:
  std::span<const char> buffer_;
  char* temporary_memory_;
  unsigned version_;
};

}  // namespace lczero::client

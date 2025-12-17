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
#include <functional>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

// #include "utils/expected.h"
#if __has_include(<expected>)
#include <expected>
#endif
#if __cpp_lib_expected < 202202L
#include <variant>
#endif
namespace lczero {
#if __cpp_lib_expected >= 202202L
template <typename E>
using Unexpected = std::unexpected<E>;
template <typename T, typename E>
using ExpectedImpl = std::expected<T, E>;
#else

template <typename E>
class Unexpected {
 public:
  explicit constexpr Unexpected(const E& e) : value(e) {}
  explicit constexpr Unexpected(E&& e) : value(std::move(e)) {}
  E value;
};

template <typename T, typename E>
class ExpectedImpl {
 public:
  using value_type = T;
  using error_type = E;
  using unexpected_type = Unexpected<E>;

  template <typename U>
  using rebind = ExpectedImpl<U, E>;

  constexpr ExpectedImpl() : data_(E{}) {};
  constexpr ExpectedImpl(const ExpectedImpl& other) = default;
  constexpr ExpectedImpl(ExpectedImpl&& other) = default;
  constexpr ExpectedImpl& operator=(const ExpectedImpl& other) = default;
  constexpr ExpectedImpl& operator=(ExpectedImpl&& other) = default;

  constexpr ExpectedImpl(const T& value) : data_(value) {};
  constexpr ExpectedImpl(T&& value) : data_(std::move(value)) {};

  constexpr ExpectedImpl(const unexpected_type& e) : data_(e.value) {}

  constexpr const T* operator->() const { return &std::get<T>(data_); }
  constexpr T* operator->() { return &std::get<T>(data_); }
  constexpr const T& value() const& { return std::get<T>(data_); }
  constexpr T& value() & { return std::get<T>(data_); }
  constexpr const T&& value() const&& { return std::get<T>(std::move(data_)); }
  constexpr T&& value() && { return std::get<T>(std::move(data_)); }

  constexpr explicit operator bool() const noexcept {
    return std::holds_alternative<T>(data_);
  }
  constexpr bool has_value() const noexcept {
    return std::holds_alternative<T>(data_);
  }

  constexpr const E& error() const& { return std::get<E>(data_); }
  constexpr E& error() & { return std::get<E>(data_); }
  constexpr const E&& error() const&& { return std::get<E>(std::move(data_)); }
  constexpr E&& error() && { return std::get<E>(std::move(data_)); }

  template <class U = std::remove_cv_t<T>>
  constexpr T value_or(U&& default_value) const& {
    if (has_value()) {
      return value();
    } else {
      return static_cast<T>(std::forward<U>(default_value));
    }
  }
  template <class U = std::remove_cv_t<T>>
  constexpr T value_or(U&& default_value) && {
    if (has_value()) {
      return std::move(value());
    } else {
      return static_cast<T>(std::forward<U>(default_value));
    }
  }

  template <class G = E>
  constexpr E error_or(G&& default_value) const& {
    if (!has_value()) {
      return error();
    } else {
      return static_cast<E>(std::forward<G>(default_value));
    }
  }
  template <class G = E>
  constexpr E error_or(G&& default_value) && {
    if (!has_value()) {
      return std::move(error());
    } else {
      return static_cast<E>(std::forward<G>(default_value));
    }
  }

  template <typename Func>
  constexpr auto and_then(Func&& func) & {
    using RetType = decltype(func(std::declval<T&>()));
    if (has_value()) {
      return func(value());
    } else {
      return RetType(unexpected_type(error()));
    }
  }
  template <typename Func>
  constexpr auto and_then(Func&& func) const& {
    using RetType = decltype(func(std::declval<const T&>()));
    if (has_value()) {
      return func(value());
    } else {
      return RetType(unexpected_type(error()));
    }
  }
  template <typename Func>
  constexpr auto and_then(Func&& func) && {
    using RetType = decltype(func(std::declval<T&&>()));
    if (has_value()) {
      return func(std::move(value()));
    } else {
      return RetType(unexpected_type(std::move(error())));
    }
  }
  template <typename Func>
  constexpr auto and_then(Func&& func) const&& {
    using RetType = decltype(func(std::declval<const T&&>()));
    if (has_value()) {
      return func(std::move(value()));
    } else {
      return RetType(unexpected_type(std::move(error())));
    }
  }

 private:
  std::variant<T, E> data_;
};

#endif

template <typename T>
using MaybeRef =
    std::conditional_t<std::is_reference_v<T>,
                       std::reference_wrapper<std::remove_reference_t<T>>, T>;

template <typename T, typename E>
using Expected = ExpectedImpl<MaybeRef<T>, E>;
}  // namespace lczero

namespace lczero::client {

enum class ArchiveError {
  None,
  BufferOverflow,
  InvalidData,
  ValueOverflow,
  SizeCalculationFailed,
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
  };

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
  };

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

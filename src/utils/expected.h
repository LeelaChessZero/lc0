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
#include <functional>

#if __has_include(<expected>)
#include <expected>
#endif
#if __cpp_lib_expected < 202211L
#include <variant>
#endif
namespace lczero {
#if __cpp_lib_expected >= 202211L
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


// A fallback implementation for std::expected. It can miss standard features
// which should be added if code needs a missing feature.
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

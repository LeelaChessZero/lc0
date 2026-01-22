/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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
#include <atomic>
#include <cassert>
#include <memory>

namespace lczero {

template <typename T>
class AtomicVector {
  using Allocator = std::allocator<T>;

 public:
  AtomicVector() = default;
  explicit AtomicVector(size_t capacity) : capacity_(capacity), size_(0) {
    data_ = Allocator().allocate(capacity);
  }

  ~AtomicVector() {
    clear();
    if (data_) {
      Allocator().deallocate(data_, capacity_);
    }
  }

  // Not thread safe.
  AtomicVector(AtomicVector&& other)
      : capacity_(other.capacity_),
        size_(other.size_.exchange(0, std::memory_order_relaxed)),
        data_(other.data_) {
    other.data_ = nullptr;
  }

  // Not thread safe.
  AtomicVector& operator=(AtomicVector&& other) {
    AtomicVector temp(std::move(*this));
    const_cast<size_t&>(capacity_) = other.capacity_;
    size_.store(other.size_.exchange(0, std::memory_order_relaxed),
                std::memory_order_relaxed);
    std::swap(data_, other.data_);
    return *this;
  }

  // Thread safe, returns the index of the inserted element.
  template <typename... Args>
  size_t emplace_back(Args&&... args) {
    size_t i = size_.fetch_add(1, std::memory_order_relaxed);
    assert(i < capacity_);
    std::construct_at(&data_[i], std::forward<Args>(args)...);
    return i;
  }

  T& operator[](size_t i) {
    assert(i < size());
    return data_[i];
  }

  const T& operator[](size_t i) const {
    assert(i < size());
    return data_[i];
  }

  bool empty() const { return size() == 0; }
  size_t size() const { return size_.load(std::memory_order_relaxed); }
  size_t capacity() const { return capacity_; }

  // Not thread safe.
  void clear() {
    for (size_t i = size_.load(std::memory_order_relaxed); i-- > 0;) {
      std::destroy_at(&data_[i]);
    }
    size_.store(0, std::memory_order_relaxed);
  }

  // Not thread safe.
  void erase(T* start, T* end) {
    assert(end == this->end());
    size_t first = start - begin();
    for (size_t i = size_.load(std::memory_order_relaxed); i-- > first;) {
      std::destroy_at(&data_[i]);
    }
    size_.fetch_add(start - end, std::memory_order_relaxed);
  }

  T* begin() { return data_; }
  T* end() { return data_ + size(); }
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size(); }

 private:
  const size_t capacity_ = 0;
  std::atomic<size_t> size_ = 0;
  T* data_ = nullptr;
};

}  // namespace lczero

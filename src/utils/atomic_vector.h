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

namespace lczero {

template <typename T>
class AtomicVector {
 public:
  explicit AtomicVector(size_t capacity) : capacity_(capacity), size_(0) {
    data_ = new
        typename std::aligned_storage<sizeof(T), alignof(T)>::type[capacity];
  }

  ~AtomicVector() {
    clear();
    delete[] data_;
  }

  // Thread safe, returns the index of the inserted element.
  template <typename... Args>
  size_t emplace_back(Args&&... args) {
    size_t i = size_.fetch_add(1, std::memory_order_relaxed);
    assert(i < capacity_);
    new (&data_[i]) T(std::forward<Args>(args)...);
    return i;
  }

  T& operator[](size_t i) {
    assert(i < size());
    return *reinterpret_cast<T*>(&data_[i]);
  }

  const T& operator[](size_t i) const {
    assert(i < size());
    return *reinterpret_cast<const T*>(&data_[i]);
  }

  size_t size() const { return size_.load(std::memory_order_relaxed); }
  size_t capacity() const { return capacity_; }

  // Not thread safe.
  void clear() {
    for (size_t i = size_.load(std::memory_order_relaxed); i-- > 0;) {
      reinterpret_cast<T*>(&data_[i])->~T();
    }
    size_.store(0, std::memory_order_relaxed);
  }

  T* begin() { return reinterpret_cast<T*>(data_); }
  T* end() { return reinterpret_cast<T*>(data_) + size(); }
  const T* begin() const { return reinterpret_cast<const T*>(data_); }
  const T* end() const { return reinterpret_cast<const T*>(data_) + size(); }

 private:
  const size_t capacity_;
  std::atomic<size_t> size_;
  typename std::aligned_storage<sizeof(T), alignof(T)>::type* data_;
};

}  // namespace lczero
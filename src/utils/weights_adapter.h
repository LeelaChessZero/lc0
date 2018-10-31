/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#include <iterator>
#include <vector>
#include "proto/net.pb.h"

namespace lczero {

using float16 = uint16_t;

template <typename T>
class LayerAdapter {
 public:
  class Iterator : public std::iterator<std::random_access_iterator_tag, T> {
   public:
    Iterator() = default;
    Iterator(const Iterator& other) = default;

    T operator*() const;
    T operator[](size_t idx) const;
    Iterator& operator++() {
      ++data_;
      return *this;
    }
    Iterator& operator--() {
      --data_;
      return *this;
    }
    ptrdiff_t operator-(const Iterator& other) { return data_ - other.data_; }

    // TODO(crem) implement other iterator functions when they are needed.

   private:
    friend class LayerAdapter;
    Iterator(const LayerAdapter* adapter, const uint16_t* ptr)
        : adapter_(adapter), data_(ptr) {}
    static float ExtractValue(const uint16_t* ptr,
                              const LayerAdapter<T>* adapter) {
      return *ptr / static_cast<float>(0xffff) * adapter->range_ +
             adapter->min_;
    }

    const LayerAdapter* adapter_ = nullptr;
    const uint16_t* data_ = nullptr;
  };

  LayerAdapter(const pblczero::Weights_Layer& layer)
      : data_(reinterpret_cast<const uint16_t*>(layer.params().data())),
        size_(layer.params().size() / sizeof(uint16_t)),
        min_(layer.min_val()),
        range_(layer.max_val() - min_) {}

  std::vector<T> as_vector() const { return std::vector<T>(begin(), end()); }

  size_t size() const { return size_; }
  T operator[](size_t idx) const { return begin()[idx]; }
  Iterator begin() const { return {this, data_}; }
  Iterator end() const { return {this, data_ + size_}; }

 private:
  const uint16_t* data_ = nullptr;
  const size_t size_ = 0;
  const float min_;
  const float range_;
};

template <>
float LayerAdapter<float>::Iterator::operator*() const;

template <>
float16 LayerAdapter<float16>::Iterator::operator*() const;

template <>
float LayerAdapter<float>::Iterator::operator[](size_t idx) const;

template <>
float16 LayerAdapter<float16>::Iterator::operator[](size_t idx) const;

}  // namespace lczero

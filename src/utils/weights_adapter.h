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

#pragma once

#include <iterator>
#include <vector>

#include "proto/net.pb.h"

namespace lczero {

class LayerAdapter {
 public:
  class Iterator {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = float;
    using difference_type = std::ptrdiff_t;
    using pointer = float*;
    using reference = float&;

    Iterator() = default;
    Iterator(const Iterator& other) = default;

    float operator*() const;
    float operator[](size_t idx) const;
    bool operator==(const LayerAdapter::Iterator& other) const {
      return data_ == other.data_;
    }
    bool operator!=(const LayerAdapter::Iterator& other) const {
      return data_ != other.data_;
    }
    Iterator& operator++() {
      ++data_;
      return *this;
    }
    Iterator& operator--() {
      --data_;
      return *this;
    }
    ptrdiff_t operator-(const Iterator& other) const {
      return data_ - other.data_;
    }

    // TODO(crem) implement other iterator functions when they are needed.

   private:
    friend class LayerAdapter;
    Iterator(const LayerAdapter* adapter, const uint16_t* ptr)
        : adapter_(adapter), data_(ptr) {}
    static float ExtractValue(const uint16_t* ptr, const LayerAdapter* adapter);

    const LayerAdapter* adapter_ = nullptr;
    const uint16_t* data_ = nullptr;
  };

  LayerAdapter(const pblczero::Weights::Layer& layer);
  std::vector<float> as_vector() const;
  size_t size() const { return size_; }
  float operator[](size_t idx) const { return begin()[idx]; }
  Iterator begin() const { return {this, data_}; }
  Iterator end() const { return {this, data_ + size_}; }

 private:
  const uint16_t* data_ = nullptr;
  const size_t size_ = 0;
  const float min_;
  const float range_;
};

}  // namespace lczero

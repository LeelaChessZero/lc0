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
*/

#pragma once

#include <memory>

namespace lczero {

// Non resizeable array which can contain up to 255 elements.
template <typename T>
class SmallArray {
 public:
  SmallArray() = delete;
  SmallArray(size_t size) : size_(size), data_(std::make_unique<T[]>(size)) {}
  SmallArray(SmallArray&&);  // TODO implement when needed
  T& operator[](int idx) { return data_[idx]; }
  const T& operator[](int idx) const { return data_[idx]; }
  int size() const { return size_; }

 private:
  unsigned char size_;
  std::unique_ptr<T[]> data_;
};

}  // namespace lczero
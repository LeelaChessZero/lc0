/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors

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

#include <cassert>
#include <numeric>
#include <vector>

namespace lczero {

// Transposes flattened tensor from @from into @to. @to must have space for
// from.size() elements.
// @dims -- Dimensions of @from tensor. For example, {120, 60, 3, 3}
// @order -- New-to-old dimension index mapping. For example {3, 2, 0, 1}
template <class T>
void TransposeTensor(const std::vector<int>& dims, std::vector<int> order,
                     const std::vector<T> from, T* to) {
  assert(from.size() == std::accumulate(dims.begin(), dims.end(), 1u,
                                        std::multiplies<size_t>()));
  if (order.empty()) {
    for (size_t i = 0; i < dims.size(); ++i)
      order.push_back(dims.size() - i - 1);
  }
  std::vector<int> cur_idx(dims.size());
  for (size_t _ = 0; _ < from.size(); ++_) {
    size_t from_idx = 0;
    for (int i : order) {
      from_idx *= dims[i];
      from_idx += cur_idx[i];
    }
    *to++ = from[from_idx];
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
      if (++cur_idx[i] == dims[i]) {
        cur_idx[i] = 0;
      } else {
        break;
      }
    }
  }
}

}  // namespace lczero

/*
This file is part of Leela Chess Zero.
Copyright (C) 2021 The LCZero Authors

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

#include "neural/onnx/adapters.h"

#include "utils/transpose.h"

namespace lczero {

FloatOnnxWeightsAdapter::FloatOnnxWeightsAdapter(
    const std::vector<float>& weights, std::initializer_list<int> dims,
    std::initializer_list<int> order)
    : weights_(weights), dims_(dims), order_(order) {}

pblczero::TensorProto::DataType FloatOnnxWeightsAdapter::GetDataType() const {
  return pblczero::TensorProto::FLOAT;
}

std::vector<int> FloatOnnxWeightsAdapter::GetDimensions() const {
  // TODO factor out to a separate class as soon as there will be something else
  // than FloatOnnxWeightsAdapter.
  return dims_;
}
std::string FloatOnnxWeightsAdapter::GetRawData() const {
  if (order_.empty()) {
    return {reinterpret_cast<const char*>(weights_.data()),
            reinterpret_cast<const char*>(weights_.data() + weights_.size())};
  } else {
    std::vector<float> dst(weights_.size());
    TransposeTensor(dims_, order_, weights_, &dst[0]);
    return {reinterpret_cast<const char*>(dst.data()),
            reinterpret_cast<const char*>(dst.data() + dst.size())};
  }
}

}  // namespace lczero
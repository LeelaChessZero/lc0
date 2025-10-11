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

#include "utils/weights_adapter.h"

#include <absl/base/optimization.h>

#include "utils/bf16_utils.h"
#include "utils/exception.h"
#include "utils/fp16_utils.h"

namespace lczero {

float LayerAdapter::Iterator::ExtractValue(const uint16_t* ptr,
                                           const LayerAdapter* adapter) {
  switch (adapter->encoding_) {
    case pblczero::Weights::Layer::LINEAR16: {
      float theta = *ptr / static_cast<float>(0xffff);
      return adapter->min_ * (1 - theta) + adapter->max_ * theta;
    }
    case pblczero::Weights::Layer::FLOAT16:
      return FP16toFP32(*ptr);
    case pblczero::Weights::Layer::BFLOAT16:
      return BF16toFP32(*ptr);
    [[unlikely]] default:  // To silence a couple of warnings.
#if defined(ABSL_UNREACHABLE)
      ABSL_UNREACHABLE();
#elif defined(__GNUC__)
      __builtin_unreachable();
#else
      __assume(false);
#endif

  }
}

LayerAdapter::LayerAdapter(const pblczero::Weights::Layer& layer)
    : data_(reinterpret_cast<const uint16_t*>(layer.params().data())),
      size_(layer.params().size() / sizeof(uint16_t)),
      min_(layer.min_val()),
      max_(layer.max_val()),
      encoding_(layer.has_encoding() ? layer.encoding()
                                     : pblczero::Weights::Layer::LINEAR16) {
  switch (encoding_) {
    case pblczero::Weights::Layer::LINEAR16:
    case pblczero::Weights::Layer::FLOAT16:
    case pblczero::Weights::Layer::BFLOAT16:
      break;
    default:
      throw Exception("Unknown layer encoding " +
                      pblczero::Weights::Layer::Encoding_Name(encoding_));
  }
}

std::vector<float> LayerAdapter::as_vector() const {
  return std::vector<float>(begin(), end());
}
float LayerAdapter::Iterator::operator*() const {
  return ExtractValue(data_, adapter_);
}
float LayerAdapter::Iterator::operator[](size_t idx) const {
  return ExtractValue(data_ + idx, adapter_);
}

}  // namespace lczero

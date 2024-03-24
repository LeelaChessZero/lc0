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

#include "neural/xla/xla_tensor.h"

namespace lczero {
namespace {
std::string AsHexString(std::string_view buf) {
  std::string result;
  result.reserve(buf.size() * 2);
  constexpr char hex[] = "0123456789abcdef";
  for (unsigned char c : buf) {
    result.push_back(hex[c >> 4]);
    result.push_back(hex[c & 0xf]);
  }
  return result;
}

}  // namespace

std::string XlaTensor::DebugString() {
  constexpr size_t kMaxSize = 1000;
  constexpr size_t kSuffixSize = 200;
  std::string result = "XlaTensor(";
  result += "shape=[";
  for (size_t i = 0; i < shape().size(); ++i) {
    if (i > 0) result += ", ";
    result += std::to_string(shape()[i]);
  }
  result += "], type=";
  result += pblczero::XlaShapeProto::Type_Name(type());
  result += ") size=" + std::to_string(size());
  result += " data=";
  if (size() <= kMaxSize) {
    result += AsHexString({static_cast<const char*>(data()), size()});
  } else {
    result += AsHexString(
        {static_cast<const char*>(data()), kMaxSize - kSuffixSize - 2});
    result += "....";
    result += AsHexString(
        {static_cast<const char*>(data()) + size() - kSuffixSize, kSuffixSize});
  }
  return result;
}

void XlaMutableTensor::Reshape(const std::vector<int64_t>& new_shape) {
  const size_t new_size = GetBufferSize(type_, new_shape);
  if (new_size > capacity_) {
    // TODO Implement allocating of new buffer when needed.
    throw Exception("Reshape to exceeds capacity.");
  }
  shape_ = new_shape;
  size_ = new_size;
}

}  // namespace lczero
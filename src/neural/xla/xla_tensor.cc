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

#include "utils/bf16_utils.h"
#include "utils/fp16_utils.h"
#include "utils/fp8_utils.h"
#include "utils/string.h"

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

namespace {
template <typename T>
struct DeduceType;
template <typename R, typename A>
struct DeduceType<R (&)(A)> {
  using result = R;
  using arg = A;
};
}  // namespace

void XlaMutableTensor::Cast(pblczero::XlaShapeProto::Type new_type) {
  if (new_type == type_) return;
  const size_t new_size = GetBufferSize(new_type, shape_);
  std::unique_ptr<char[]> new_data;
  const void* src = data_.get();
  if (new_size > capacity_ ||
      GetXlaTypeSize(new_type) > GetXlaTypeSize(type_)) {
    capacity_ = std::max(new_size, capacity_);
    new_data.reset(new char[capacity_]);
    std::swap(data_, new_data);
  }
  void* dst = data_.get();
  if (new_type != pblczero::XlaShapeProto::F32 &&
      type_ != pblczero::XlaShapeProto::F32) {
    throw Exception(
        "Only float32 casts are supported, attempting to cast from " +
        pblczero::XlaShapeProto::Type_Name(type_) + " to " +
        pblczero::XlaShapeProto::Type_Name(new_type));
  }
  auto convert = [&](auto&& func) {
    using src_t = typename DeduceType<decltype(func)>::arg;
    using dst_t = typename DeduceType<decltype(func)>::result;
    const size_t count = std::accumulate(shape_.begin(), shape_.end(), 1,
                                         std::multiplies<int64_t>());
    const src_t* src_ptr = static_cast<const src_t*>(src);
    dst_t* dst_ptr = static_cast<dst_t*>(dst);
    for (size_t i = 0; i < count; ++i) dst_ptr[i] = func(src_ptr[i]);
  };
  if (type_ == pblczero::XlaShapeProto::F32) {
    switch (new_type) {
      case pblczero::XlaShapeProto::F16:
        convert(FP32toFP16);
        break;
      case pblczero::XlaShapeProto::BF16:
        convert(FP32toBF16);
        break;
      case pblczero::XlaShapeProto::F8E5M2:
        convert(FP32toFP8E5M2_Saturate);
        break;
      default:
        throw Exception("Unsupported cast F32 -> " +
                        pblczero::XlaShapeProto::Type_Name(new_type));
    }
  } else {
    switch (type_) {
      case pblczero::XlaShapeProto::F16:
        convert(FP16toFP32);
        break;
      case pblczero::XlaShapeProto::BF16:
        convert(BF16toFP32);
        break;
      case pblczero::XlaShapeProto::F8E5M2:
        convert(FP8E5M2toFP32);
        break;
      default:
        throw Exception("Unsupported cast " +
                        pblczero::XlaShapeProto::Type_Name(type_) + " -> F32");
    }
  }
  size_ = new_size;
  type_ = new_type;
}

pblczero::XlaShapeProto::Type StringToXlaType(const std::string& type) {
  for (const auto& types : {
           pblczero::XlaShapeProto::S4,
           pblczero::XlaShapeProto::S8,
           pblczero::XlaShapeProto::S16,
           pblczero::XlaShapeProto::S32,
           pblczero::XlaShapeProto::S64,
           pblczero::XlaShapeProto::U4,
           pblczero::XlaShapeProto::U8,
           pblczero::XlaShapeProto::U16,
           pblczero::XlaShapeProto::U32,
           pblczero::XlaShapeProto::U64,
           pblczero::XlaShapeProto::F16,
           pblczero::XlaShapeProto::F32,
           pblczero::XlaShapeProto::BF16,
           pblczero::XlaShapeProto::F64,
           pblczero::XlaShapeProto::F8E5M2,
           pblczero::XlaShapeProto::F8E4M3FN,
           pblczero::XlaShapeProto::F8E4M3B11FNUZ,
           pblczero::XlaShapeProto::F8E5M2FNUZ,
           pblczero::XlaShapeProto::F8E4M3FNUZ,
           pblczero::XlaShapeProto::C64,
           pblczero::XlaShapeProto::C128,
       }) {
    if (StringsEqualIgnoreCase(type,
                               pblczero::XlaShapeProto::Type_Name(types))) {
      return types;
    }
  }
  throw Exception("Cannot convert to XLA type: " + type);
}

}  // namespace lczero
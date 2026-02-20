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

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "neural/xla/onnx_builder_interface.h"
#include "neural/xla/onnx_builder_proto_utils.h"
#include "proto/hlo.pb.h"
#include "utils/exception.h"

namespace lczero {

template <typename T>
inline T DecodeLittleEndian(const uint8_t* data) {
  T value{};
  std::memcpy(&value, data, sizeof(T));
  return value;
}

inline size_t NumElements(const TensorType& type) {
  if (type.dimensions.empty()) return 1;
  size_t elements = 1;
  for (const int64_t dim : type.dimensions) {
    if (dim < 0) throw Exception("Negative tensor dimension in TensorLiteral");
    elements *= static_cast<size_t>(dim);
  }
  return elements;
}

inline pblczero::XlaLiteralProto ToLiteralProto(const TensorLiteral& literal) {
  pblczero::XlaLiteralProto result;
  *result.mutable_shape() = TensorTypeToProto(literal.type);
  const size_t elements = NumElements(literal.type);

  auto expect_bytes = [&](size_t expected) {
    if (literal.bytes.size() != expected) {
      throw Exception("TensorLiteral byte size mismatch: expected " +
                      std::to_string(expected) + ", got " +
                      std::to_string(literal.bytes.size()));
    }
  };

  switch (literal.type.element_type) {
    case TensorType::ElementType::kPred:
      expect_bytes(elements);
      for (size_t i = 0; i < elements; ++i) {
        result.add_preds(literal.bytes[i] != 0);
      }
      break;
    case TensorType::ElementType::kS32:
      expect_bytes(elements * sizeof(int32_t));
      for (size_t i = 0; i < elements; ++i) {
        result.add_s32s(
            DecodeLittleEndian<int32_t>(&literal.bytes[i * sizeof(int32_t)]));
      }
      break;
    case TensorType::ElementType::kS64:
      expect_bytes(elements * sizeof(int64_t));
      for (size_t i = 0; i < elements; ++i) {
        result.add_s64s(
            DecodeLittleEndian<int64_t>(&literal.bytes[i * sizeof(int64_t)]));
      }
      break;
    case TensorType::ElementType::kF32:
      expect_bytes(elements * sizeof(float));
      for (size_t i = 0; i < elements; ++i) {
        result.add_f32s(
            DecodeLittleEndian<float>(&literal.bytes[i * sizeof(float)]));
      }
      break;
    case TensorType::ElementType::kF16:
      result.set_f16s(std::string(
          reinterpret_cast<const char*>(literal.bytes.data()),
          literal.bytes.size()));
      break;
    case TensorType::ElementType::kBF16:
      result.set_bf16s(std::string(
          reinterpret_cast<const char*>(literal.bytes.data()),
          literal.bytes.size()));
      break;
    case TensorType::ElementType::kInvalid:
      throw Exception("TensorLiteral has invalid element type");
  }
  return result;
}

template <typename T>
inline void AppendValueBytes(T value, std::vector<uint8_t>* bytes) {
  const auto* begin = reinterpret_cast<const uint8_t*>(&value);
  bytes->insert(bytes->end(), begin, begin + sizeof(T));
}

inline TensorLiteral FromLiteralProto(const pblczero::XlaLiteralProto& proto) {
  TensorLiteral result;
  result.type = TensorTypeFromProto(proto.shape());
  const size_t elements = NumElements(result.type);

  auto expect_elements = [&](size_t actual) {
    if (actual != elements) {
      throw Exception("XlaLiteralProto element count mismatch: expected " +
                      std::to_string(elements) + ", got " + std::to_string(actual));
    }
  };

  switch (result.type.element_type) {
    case TensorType::ElementType::kPred:
      expect_elements(static_cast<size_t>(proto.preds_size()));
      result.bytes.reserve(elements);
      for (bool value : proto.preds()) {
        result.bytes.push_back(value ? 1 : 0);
      }
      break;
    case TensorType::ElementType::kS32:
      expect_elements(static_cast<size_t>(proto.s32s_size()));
      result.bytes.reserve(elements * sizeof(int32_t));
      for (int32_t value : proto.s32s()) {
        AppendValueBytes(value, &result.bytes);
      }
      break;
    case TensorType::ElementType::kS64:
      expect_elements(static_cast<size_t>(proto.s64s_size()));
      result.bytes.reserve(elements * sizeof(int64_t));
      for (int64_t value : proto.s64s()) {
        AppendValueBytes(value, &result.bytes);
      }
      break;
    case TensorType::ElementType::kF32:
      expect_elements(static_cast<size_t>(proto.f32s_size()));
      result.bytes.reserve(elements * sizeof(float));
      for (float value : proto.f32s()) {
        AppendValueBytes(value, &result.bytes);
      }
      break;
    case TensorType::ElementType::kF16: {
      const auto raw = proto.f16s();
      const size_t expected = elements * sizeof(uint16_t);
      if (raw.size() != expected) {
        throw Exception("XlaLiteralProto f16 byte size mismatch: expected " +
                        std::to_string(expected) + ", got " +
                        std::to_string(raw.size()));
      }
      result.bytes.assign(raw.begin(), raw.end());
      break;
    }
    case TensorType::ElementType::kBF16: {
      const auto raw = proto.bf16s();
      const size_t expected = elements * sizeof(uint16_t);
      if (raw.size() != expected) {
        throw Exception("XlaLiteralProto bf16 byte size mismatch: expected " +
                        std::to_string(expected) + ", got " +
                        std::to_string(raw.size()));
      }
      result.bytes.assign(raw.begin(), raw.end());
      break;
    }
    case TensorType::ElementType::kInvalid:
      throw Exception("XlaLiteralProto has invalid element type");
  }
  return result;
}

}  // namespace lczero

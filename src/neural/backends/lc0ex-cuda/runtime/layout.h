/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "proto/lc0ex.pb.h"
#include "utils/exception.h"

// Helpers for the strided buffer layouts described in proto/lc0ex.proto.
namespace lczero {
namespace lc0ex {

inline std::size_t ElementSize(pblczero::Buffer::DataType type) {
  switch (type) {
    case pblczero::Buffer::DATA_TYPE_F32:
      return sizeof(float);
    case pblczero::Buffer::DATA_TYPE_U8:
      return sizeof(std::uint8_t);
    case pblczero::Buffer::DATA_TYPE_F16:
      return sizeof(std::uint16_t);
    case pblczero::Buffer::DATA_TYPE_U64:
      return sizeof(std::uint64_t);
    case pblczero::Buffer::DATA_TYPE_BF16:
      return sizeof(std::uint16_t);
    case pblczero::Buffer::DATA_TYPE_UNKNOWN:
      break;
  }
  throw Exception("Unsupported or unknown lc0ex buffer data type.");
}

// Row-major strides, in elements.
inline std::vector<std::int64_t> DefaultStrides(
    const std::vector<std::uint64_t>& shape) {
  std::vector<std::int64_t> strides(shape.size(), 1);
  for (std::size_t i = shape.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * static_cast<std::int64_t>(shape[i - 1]);
  }
  return strides;
}

// Number of bytes spanned by a tensor with the given strides, i.e. one past
// the last byte of its last element. Returns nullopt on negative strides or
// overflow.
inline std::optional<std::uint64_t> StridedExtentBytes(
    const std::vector<std::uint64_t>& shape,
    const std::vector<std::int64_t>& strides, std::size_t elem_size) {
  if (strides.size() != shape.size()) return std::nullopt;
  constexpr auto kMax = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t last_element = 0;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) return 0;
    if (strides[i] < 0) return std::nullopt;
    const auto stride = static_cast<std::uint64_t>(strides[i]);
    if (stride != 0 && shape[i] - 1 > kMax / stride) return std::nullopt;
    const std::uint64_t span = (shape[i] - 1) * stride;
    if (last_element > kMax - span) return std::nullopt;
    last_element += span;
  }
  if (last_element > kMax / elem_size - 1) return std::nullopt;
  return (last_element + 1) * elem_size;
}

// Returns the shape of the logical row-major prefix of `size_bytes` bytes:
// the leading dimension is reduced to the number of rows covered. Throws if
// the prefix does not consist of whole rows.
inline std::vector<std::uint64_t> PrefixShape(
    const std::vector<std::uint64_t>& shape, std::size_t elem_size,
    std::size_t size_bytes) {
  if (shape.empty()) {
    if (size_bytes != elem_size) {
      throw Exception("Partial copies of scalar lc0ex buffers are invalid.");
    }
    return shape;
  }
  std::uint64_t row_bytes = elem_size;
  for (std::size_t i = 1; i < shape.size(); ++i) row_bytes *= shape[i];
  if (row_bytes == 0 || size_bytes % row_bytes != 0 ||
      size_bytes / row_bytes > shape[0]) {
    throw Exception(
        "A strided lc0ex buffer copy must cover whole leading rows.");
  }
  auto result = shape;
  result[0] = size_bytes / row_bytes;
  return result;
}

// Splits a strided copy into maximal contiguous chunks and calls
// fn(dst_offset_bytes, src_offset_bytes, size_bytes) for each of them.
template <typename Fn>
void ForEachContiguousChunk(const std::vector<std::uint64_t>& shape,
                            const std::vector<std::int64_t>& dst_strides,
                            const std::vector<std::int64_t>& src_strides,
                            std::size_t elem_size, Fn&& fn) {
  // Find the innermost dimensions that are contiguous in both layouts.
  std::size_t contiguous_dim = shape.size();
  std::int64_t contiguous_elements = 1;
  while (contiguous_dim > 0) {
    const std::size_t dim = contiguous_dim - 1;
    if (dst_strides[dim] != contiguous_elements ||
        src_strides[dim] != contiguous_elements) {
      break;
    }
    contiguous_elements *= static_cast<std::int64_t>(shape[dim]);
    contiguous_dim = dim;
  }
  const std::size_t contiguous_bytes = contiguous_elements * elem_size;
  if (contiguous_bytes == 0) return;

  auto copy_dim = [&](auto& self, std::size_t dim, std::int64_t dst_offset,
                      std::int64_t src_offset) -> void {
    if (dim == contiguous_dim) {
      fn(dst_offset, src_offset, contiguous_bytes);
      return;
    }
    const auto dst_step =
        dst_strides[dim] * static_cast<std::int64_t>(elem_size);
    const auto src_step =
        src_strides[dim] * static_cast<std::int64_t>(elem_size);
    for (std::uint64_t i = 0; i < shape[dim]; ++i) {
      self(self, dim + 1, dst_offset + static_cast<std::int64_t>(i) * dst_step,
           src_offset + static_cast<std::int64_t>(i) * src_step);
    }
  };
  copy_dim(copy_dim, 0, 0, 0);
}

}  // namespace lc0ex
}  // namespace lczero

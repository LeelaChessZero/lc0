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
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "neural/xla/hlo.pb.h"
#include "utils/exception.h"

namespace lczero {

pblczero::XlaShapeProto::Type StringToXlaType(const std::string& type);

inline size_t GetXlaTypeSize(pblczero::XlaShapeProto::Type type) {
  switch (type) {
    case pblczero::XlaShapeProto::F16:
      return sizeof(uint16_t);
    case pblczero::XlaShapeProto::BF16:
      return sizeof(uint16_t);
    case pblczero::XlaShapeProto::F8E5M2:
      return sizeof(uint8_t);
    case pblczero::XlaShapeProto::F32:
      return sizeof(float);
    case pblczero::XlaShapeProto::F64:
      return sizeof(double);
    case pblczero::XlaShapeProto::S32:
      return sizeof(int32_t);
    case pblczero::XlaShapeProto::S64:
      return sizeof(int64_t);
    default:
      throw Exception("Add size for type " +
                      pblczero::XlaShapeProto::Type_Name(type));
  }
}

// An interface for in-host-memory tensor in XLA format.
class XlaTensor {
 public:
  virtual ~XlaTensor() = default;
  virtual const std::vector<int64_t>& shape() const = 0;
  virtual const void* data() const = 0;
  // Returns amount of valid data in bytes.
  virtual size_t size() const = 0;
  // Returns amount of memory that are allowed to address in bytes. This is
  // useful when the size of the buffer has to be increased to match the
  // supported batch size.
  virtual size_t capacity() const = 0;
  virtual pblczero::XlaShapeProto::Type type() const = 0;

  std::string DebugString();
};

// Not-owned XLA tensor, used e.g. when ONNX buffer can be used directly, to
// avoid unnecessary copy.
class XlaTensorNotOwned : public XlaTensor {
 public:
  XlaTensorNotOwned(pblczero::XlaShapeProto::Type type,
                    const std::vector<int64_t>& shape, std::string_view data)
      : shape_(&shape), data_(data), type_(type) {}

  const std::vector<int64_t>& shape() const override { return *shape_; }
  const void* data() const override { return data_.data(); }
  size_t size() const override { return data_.size(); }
  size_t capacity() const override { return data_.size(); }
  pblczero::XlaShapeProto::Type type() const override { return type_; }

 private:
  const std::vector<int64_t>* shape_;
  std::string_view data_;
  pblczero::XlaShapeProto::Type type_;
};

// Tensor that owns data, used e.g. for XLA output.
class XlaMutableTensor : public XlaTensor {
 public:
  XlaMutableTensor(pblczero::XlaShapeProto::Type type,
                   const std::vector<int64_t>& shape, size_t capacity = 0)
      : shape_(shape),
        type_(type),
        size_(GetBufferSize(type, shape)),
        capacity_(std::max(size_, capacity)),
        // TODO replace with make_unique_for_overwrite() once C++20 is
        // available.
        data_(new char[capacity_]) {}

  const std::vector<int64_t>& shape() const override { return shape_; }
  const void* data() const override { return data_.get(); }
  // TODO replace with std::span once C++20 is available.
  void* mutable_data() { return data_.get(); }
  size_t size() const override { return size_; }
  size_t capacity() const override { return capacity_; }
  pblczero::XlaShapeProto::Type type() const override { return type_; }
  void Reshape(const std::vector<int64_t>& new_shape);
  void Cast(pblczero::XlaShapeProto::Type new_type);

  static size_t GetBufferSize(pblczero::XlaShapeProto::Type type,
                              const std::vector<int64_t>& shape) {
    return GetXlaTypeSize(type) * std::accumulate(shape.begin(), shape.end(), 1,
                                                  std::multiplies<int64_t>());
  }

 private:
  std::vector<int64_t> shape_;
  pblczero::XlaShapeProto::Type type_;
  size_t size_;
  size_t capacity_;
  std::unique_ptr<char[]> data_;
};

}  // namespace lczero
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
#include <string>
#include <vector>

#include "neural/xla/hlo.pb.h"

namespace lczero {

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
  XlaTensorNotOwned(const std::vector<int64_t>& shape, std::string_view data,
                    pblczero::XlaShapeProto::Type type)
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
class XlaTensorStringBuf : public XlaTensor {
 public:
  XlaTensorStringBuf(const std::vector<int64_t>& shape,
                 pblczero::XlaShapeProto::Type type, std::string data)
      : shape_(shape), type_(type), data_(data) {}

  const std::vector<int64_t>& shape() const override { return shape_; }
  const void* data() const override { return data_.data(); }
  size_t size() const override { return data_.size(); }
  size_t capacity() const override { return data_.size(); }
  pblczero::XlaShapeProto::Type type() const override { return type_; }

 private:
  std::vector<int64_t> shape_;
  pblczero::XlaShapeProto::Type type_;
  std::string data_;
};

}  // namespace lczero
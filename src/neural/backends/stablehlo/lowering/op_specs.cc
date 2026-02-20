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

#include "neural/backends/stablehlo/lowering/op_specs.h"

#include <array>
#include <string>

#include "utils/exception.h"

namespace lczero::stablehlo::lowering {

const OpSpec& GetOpSpec(semantic::OpKind kind) {
  // Canonical semantic-lowering op surface. If an op is added here, update
  // semantic lowering and its smoke coverage in lockstep.
  static constexpr std::array<OpSpec, static_cast<size_t>(semantic::OpKind::kReturn) + 1>
      kSpecs = {{
      {"parameter_v1", 0, false, false},            // kParameter
      {"constant_v1", 0, true, false},              // kConstant
      {"convert_v1", 1, false, false},              // kConvert
      {"convolution_v1", 2, true, false},           // kConvolution
      {"broadcast_in_dim_v1", 1, true, false},      // kBroadcastInDim
      {"add_v1", 2, false, false},                  // kAdd
      {"subtract_v1", 2, false, false},             // kSubtract
      {"multiply_v1", 2, false, false},             // kMultiply
      {"divide_v1", 2, false, false},               // kDivide
      {"maximum_v1", 2, false, false},              // kMaximum
      {"reshape_v1", 1, false, false},              // kReshape
      {"dot_general_v1", 2, true, false},           // kDotGeneral
      {"slice_v1", 1, true, false},                 // kSlice
      {"concatenate_v1", 0, true, false},           // kConcatenate
      {"tanh_v1", 1, false, false},                 // kTanh
      {"log_plus_one_v1", 1, false, false},         // kLogPlusOne
      {"exponential_minus_one_v1", 1, false, false},  // kExponentialMinusOne
      {"negate_v1", 1, false, false},               // kNegate
      {"exponential_v1", 1, false, false},          // kExponential
      {"sqrt_v1", 1, false, false},                 // kSqrt
      {"rsqrt_v1", 1, false, false},                // kRsqrt
      {"tuple_v1", 0, false, false},                // kTuple
      {"reduce_v1", 2, true, true},                 // kReduce
      {"transpose_v1", 1, true, false},             // kTranspose
      {"gather_v1", 2, true, false},                // kGather
      {"compare_v1", 2, true, false},               // kCompare
      {"select_v1", 3, false, false},               // kSelect
      {"return_v1", 0, false, false},               // kReturn
  }};
  static_assert(kSpecs.size() == static_cast<size_t>(semantic::OpKind::kReturn) + 1,
                "kSpecs size must match OpKind enum count (through kReturn)");

  const size_t index = static_cast<size_t>(kind);
  if (index >= kSpecs.size()) {
    throw Exception("Missing OpSpec for OpKind index " + std::to_string(index));
  }
  return kSpecs.at(index);
}

}  // namespace lczero::stablehlo::lowering

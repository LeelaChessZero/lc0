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

#include "neural/backends/stablehlo/semantic/semantic_ir.h"

#include <vector>

#include "utils/exception.h"

int main() {
  lczero::stablehlo::semantic::SemanticBuilder builder;
  lczero::IBuilder* iface = &builder;

  const lczero::TensorType scalar{lczero::TensorType::ElementType::kF32, {}};
  const lczero::ValueId lhs = iface->Parameter(scalar);
  const lczero::ValueId rhs = iface->Parameter(scalar);
  const lczero::ValueId sum = iface->Add(lhs, rhs);
  const lczero::TensorType mat_lhs{lczero::TensorType::ElementType::kF32, {2, 3}};
  const lczero::TensorType mat_rhs{lczero::TensorType::ElementType::kF32, {3, 4}};
  const lczero::ValueId dot_lhs = iface->Parameter(mat_lhs);
  const lczero::ValueId dot_rhs = iface->Parameter(mat_rhs);
  lczero::DotParams dot_params;
  dot_params.dimension_numbers.add_lhs_contracting_dimensions(1);
  dot_params.dimension_numbers.add_rhs_contracting_dimensions(0);
  const lczero::ValueId dot = iface->Dot(dot_lhs, dot_rhs, dot_params);
  const auto dot_type = iface->GetType(dot);
  if (dot_type.dimensions != std::vector<int64_t>{2, 4}) return 5;

  const lczero::TensorType gather_input_type{
      lczero::TensorType::ElementType::kF32, {2, 3}};
  const lczero::TensorType gather_indices_type{
      lczero::TensorType::ElementType::kS32, {4, 1}};
  const lczero::ValueId gather_input = iface->Parameter(gather_input_type);
  const lczero::ValueId gather_indices = iface->Parameter(gather_indices_type);
  lczero::GatherParams gather_params;
  gather_params.index_vector_dim = 1;
  gather_params.offset_dims = {1};
  gather_params.slice_sizes = {1, 3};
  gather_params.collapsed_slice_dims = {0};
  gather_params.start_index_map = {0};
  const lczero::ValueId gather =
      iface->Gather(gather_input, gather_indices, gather_params);
  const auto gather_type = iface->GetType(gather);
  if (gather_type.dimensions != std::vector<int64_t>{4, 3}) return 6;

  // Negative test: invalid compare direction must throw.
  try {
    lczero::CompareParams bad_compare;
    bad_compare.direction = "LTE";
    (void)iface->Compare(lhs, rhs, bad_compare);
    return 11;  // should not reach here
  } catch (const lczero::Exception&) {
    // expected
  }

  // Negative test: invalid transpose permutation must throw.
  try {
    (void)iface->Transpose(dot_lhs, {0, 2, 3, 1});
    return 12;  // should not reach here
  } catch (const lczero::Exception&) {
    // expected
  }

  // Negative test: invalid ValueId must throw.
  try {
    (void)iface->Add(9999, 0);
    return 10;  // should not reach here
  } catch (const lczero::Exception&) {
    // expected
  }

  builder.Return({sum});

  const auto module = builder.BuildModule();
  if (module.functions.size() != 1) return 1;
  const auto& entry = module.functions.front();
  if (entry.name != "main") return 2;
  if (entry.param_types.size() != 6) return 3;
  if (entry.ops.size() != 4) return 4;  // add + dot + gather + return

  // Tuple type test: tuple GetType should fail-fast once PR11.3 is enabled.
  {
    auto param_a =
        iface->Parameter({lczero::TensorType::ElementType::kF32, {2, 3}});
    auto param_b = iface->Parameter({lczero::TensorType::ElementType::kS32, {4}});
    auto tuple_id = iface->Tuple({param_a, param_b});
    try {
      (void)iface->GetType(tuple_id);
      return 23;  // should not reach here
    } catch (...) {
      // expected
    }
  }

  return 0;
}

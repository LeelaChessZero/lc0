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

#include <cstdint>
#include <vector>

#include "neural/xla/hlo_builder_adapter.h"
#include "neural/xla/tensor_literal_utils.h"
#include "utils/exception.h"

namespace {

template <typename T>
std::vector<uint8_t> BuildBytes(const std::vector<T>& values) {
  std::vector<uint8_t> bytes;
  bytes.reserve(values.size() * sizeof(T));
  for (const T value : values) {
    lczero::AppendValueBytes(value, &bytes);
  }
  return bytes;
}

void ExpectRoundTrip(const lczero::TensorLiteral& original) {
  const pblczero::XlaLiteralProto proto = lczero::ToLiteralProto(original);
  const lczero::TensorLiteral round_tripped = lczero::FromLiteralProto(proto);
  if (round_tripped.type.element_type != original.type.element_type ||
      round_tripped.type.dimensions != original.type.dimensions ||
      round_tripped.bytes != original.bytes) {
    throw lczero::Exception("TensorLiteral round-trip mismatch");
  }
}

void RunLiteralRoundTripSmoke() {
  ExpectRoundTrip({{lczero::TensorType::ElementType::kF32, {}},
                   BuildBytes<float>({1.5f})});
  ExpectRoundTrip({{lczero::TensorType::ElementType::kS32, {2}},
                   BuildBytes<int32_t>({7, -3})});
  ExpectRoundTrip({{lczero::TensorType::ElementType::kS64, {2}},
                   BuildBytes<int64_t>({42, -9})});
  ExpectRoundTrip({{lczero::TensorType::ElementType::kF16, {2}},
                   BuildBytes<uint16_t>({0x3c00, 0xc000})});
  ExpectRoundTrip({{lczero::TensorType::ElementType::kBF16, {2}},
                   BuildBytes<uint16_t>({0x3f80, 0xc000})});
  ExpectRoundTrip({{lczero::TensorType::ElementType::kPred, {3}},
                   std::vector<uint8_t>{1, 0, 1}});
}

}  // namespace

int main() {
  try {
    RunLiteralRoundTripSmoke();

    lczero::HloBuilderAdapter adapter;
    lczero::IBuilder* builder = &adapter;
    lczero::BuilderContext context(builder);
    context.SetOpType("adapter-smoke");
    context.SetOpName("adapter-smoke");

    const lczero::TensorType scalar{lczero::TensorType::ElementType::kF32, {}};
    const lczero::ValueId lhs = builder->Parameter(scalar);
    const lczero::ValueId rhs = builder->Parameter(scalar);
    const lczero::ValueId sum = builder->Add(lhs, rhs);
    (void)builder->GetType(sum);

    (void)adapter.BuildModule("adapter_smoke");

    // Negative test: invalid ValueId must throw.
    try {
      (void)builder->GetType(9999);
      return 10;  // should not reach here
    } catch (const lczero::Exception&) {
      // expected
    }

    return 0;
  } catch (const lczero::Exception&) {
    return 1;
  }
}

/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include "proto/net.pb.h"

namespace lczero {

void DescribeNetworkCmd();
void ShowNetworkGenericInfo(const pblczero::Net& weights);
void ShowNetworkFormatInfo(const pblczero::Net& weights);
void ShowNetworkTrainingInfo(const pblczero::Net& weights);
void ShowNetworkWeightsInfo(const pblczero::Net& weights);
void ShowNetworkOnnxInfo(const pblczero::Net& weights,
                         bool show_onnx_internals);
void ShowAllNetworkInfo(const pblczero::Net& weights);

}  // namespace lczero
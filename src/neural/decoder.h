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

#pragma once

#include "chess/position.h"
#include "neural/network.h"
#include "proto/net.pb.h"

namespace lczero {

// Decodes the move that led to current position using the current and previous
// input planes. Move is from the perspective of the current position to move
// player, so will need flipping if it is to be applied to the prior position.
//
// NOTE: Assumes InputPlanes are not transformed. Any canonical transforms must
// have already been reverted.
Move DecodeMoveFromInput(const InputPlanes& planes, const InputPlanes& prev);

// Decodes the current position into a board, rule50 and gameply.
//
// NOTE: Assumes InputPlanes are not transformed, regardless of input_format.
void PopulateBoard(pblczero::NetworkFormat::InputFormat input_format,
                   InputPlanes planes, ChessBoard* board, int* rule50,
                   int* gameply);

}  // namespace lczero
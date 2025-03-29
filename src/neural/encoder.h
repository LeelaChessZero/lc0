/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#include <span>

#include "chess/position.h"
#include "neural/network.h"
#include "proto/net.pb.h"

namespace lczero {

constexpr int kMoveHistory = 8;
constexpr int kPlanesPerBoard = 13;
constexpr int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;

enum class FillEmptyHistory { NO, FEN_ONLY, ALWAYS };

// Returns the transform that would be used in EncodePositionForNN.
int TransformForPosition(pblczero::NetworkFormat::InputFormat input_format,
                         const PositionHistory& history);

// Encodes the last position in history for the neural network request.
InputPlanes EncodePositionForNN(
    pblczero::NetworkFormat::InputFormat input_format,
    const PositionHistory& history, int history_planes,
    FillEmptyHistory fill_empty_history, int* transform_out);

InputPlanes EncodePositionForNN(
    pblczero::NetworkFormat::InputFormat input_format,
    std::span<const Position> positions, int history_planes,
    FillEmptyHistory fill_empty_history, int* transform_out);

bool IsCanonicalFormat(pblczero::NetworkFormat::InputFormat input_format);
bool IsCanonicalArmageddonFormat(
    pblczero::NetworkFormat::InputFormat input_format);
bool IsHectopliesFormat(pblczero::NetworkFormat::InputFormat input_format);
bool Is960CastlingFormat(pblczero::NetworkFormat::InputFormat input_format);

uint16_t MoveToNNIndex(Move move, int transform);
Move MoveFromNNIndex(int idx, int transform);

}  // namespace lczero

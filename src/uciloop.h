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
*/

#pragma once

#include <functional>
#ifdef _MSC_VER
#include "utils/optional.h"
#else
#include <optional>
using std::optional;
#endif
#include <string>
#include <vector>
#include "chess/bitboard.h"

namespace lczero {

// Implements Uci loop.
void UciLoop(int argc, const char** argv);

struct BestMoveInfo {
  BestMoveInfo(Move bestmove) : bestmove(bestmove) {}
  Move bestmove;
  Move ponder;
  using Callback = std::function<void(const BestMoveInfo&)>;
};

struct UciInfo {
  // Full depth.
  int depth = -1;
  // Maximum depth.
  int seldepth = -1;
  // Time since start of thinking.
  int64_t time = -1;
  // Nodes visited.
  int64_t nodes = -1;
  // Nodes per second.
  int nps = -1;
  // Win in centipawns.
  optional<int> score;
  // Best line found. Moves are from perspective of white player.
  std::vector<Move> pv;
  // Freeform comment.
  std::string comment;

  using Callback = std::function<void(const UciInfo&)>;
};

}  // namespace lczero
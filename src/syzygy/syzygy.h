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

#include <algorithm>
#include <atomic>
#include <deque>
#include <memory>
#include <tuple>
#include <vector>
#include "chess/position.h"

namespace lczero {

enum WDLScore {
  WDL_LOSS = -2,          // Loss
  WDL_BLESSED_LOSS = -1,  // Loss, but draw under 50-move rule
  WDL_DRAW = 0,           // Draw
  WDL_CURSED_WIN = 1,     // Win, but draw under 50-move rule
  WDL_WIN = 2,            // Win
};

// Possible states after a probing operation
enum ProbeState {
  FAIL = 0,              // Probe failed (missing file table)
  OK = 1,                // Probe succesful
  CHANGE_STM = -1,       // DTZ should check the other side
  ZEROING_BEST_MOVE = 2  // Best move zeroes DTZ (capture or pawn move)
};

class SyzygyTablebaseImpl;

// Provides methods to load and probe syzygy tablebases.
// Thread safe methods are thread safe subject to the non-thread sfaety
// conditions of the init method.
class SyzygyTablebase {
 public:
  SyzygyTablebase();
  virtual ~SyzygyTablebase();
  // Current maximum number of pieces on board that can be probed for. Will
  // be 0 unless initialized with tablebase paths.
  // Thread safe.
  int max_cardinality() { return max_cardinality_; }
  // Allows for the tablebases being used to be changed. This method is not
  // thread safe, there must be no concurrent usage while this method is
  // running. All other thread safe method calls must be strictly ordered with
  // respect to this method.
  bool init(const std::string& paths);
  // Probes WDL tables for the given position to determine a WDLScore.
  // Thread safe.
  // Result is only strictly valid for positions with 0 ply 50 move counter.
  // Probe state will return FAIL if the position is not in the tablebase.
  WDLScore probe_wdl(const Position& pos, ProbeState* result);
  // Probes DTZ tables for the given position to determine the number of ply
  // before a zeroing move under optimal play.
  // Thread safe.
  // Probe state will return FAIL if the position is not in the tablebase.
  int probe_dtz(const Position& pos, ProbeState* result);
  // Probes DTZ tables to determine which moves are on the optimal play path.
  // Assumes the position is one reached such that the side to move has been
  // performing optimal play moves since the last 50 move counter reset.
  // has_repeated should be whether there are any repeats since last 50 move
  // counter reset.
  // Thread safe.
  // Returns false if the position is not in the tablebase.
  // Safe moves are added to the safe_moves output paramater.
  bool root_probe(const Position& pos, bool has_repeated,
                  std::vector<Move>* safe_moves);
  // Probes WDL tables to determine which moves might be on the optimal play
  // path. If 50 move ply counter is non-zero some (or maybe even all) of the
  // returned safe moves in a 'winning' position, may actually be draws.
  // Returns false if the position is not in the tablebase.
  // Safe moves are added to the safe_moves output paramater.
  bool root_probe_wdl(const Position& pos, std::vector<Move>* safe_moves);
  std::string get_paths() { return paths_; }

 private:
  template <bool CheckZeroingMoves = false>
  WDLScore search(const Position& pos, ProbeState* result);

  std::string paths_;
  // Caches the max_cardinality from the impl, as max_cardinality may be a hot
  // path.
  int max_cardinality_;
  std::unique_ptr<SyzygyTablebaseImpl> impl_;
};

}  // namespace lczero

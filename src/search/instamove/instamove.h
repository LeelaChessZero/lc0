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

#include <atomic>

#include "chess/uciloop.h"
#include "search/search.h"

// Base class for instamove searches (e.g. policy head and value head).
// The classes should only implement GetBestMove() method.

namespace lczero {

class InstamoveSearch : public SearchBase {
 public:
  using SearchBase::SearchBase;

 private:
  virtual Move GetBestMove() = 0;

  void Start(const GoParams& go_params) final {
    responded_bestmove_.store(false, std::memory_order_relaxed);
    bestmove_ = GetBestMove();
    if (!go_params.infinite && !go_params.ponder) RespondBestMove();
  }
  void Wait() final {
    while (!responded_bestmove_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  void Stop() final { RespondBestMove(); }
  void Abort() final { responded_bestmove_.store(true); }
  void RespondBestMove() {
    if (responded_bestmove_.exchange(true)) return;
    BestMoveInfo info{bestmove_};
    uci_responder()->OutputBestMove(&info);
  }

  Move bestmove_;
  std::atomic<bool> responded_bestmove_{false};
};

template <typename SearchClass>
class InstamoveEnvironment : public SearchEnvironment {
 public:
  using SearchEnvironment::SearchEnvironment;

 private:
  std::unique_ptr<SearchBase> CreateSearch(
      const GameState& game_state) override {
    return std::make_unique<SearchClass>(context_, game_state);
  }
};

}  // namespace lczero
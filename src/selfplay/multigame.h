/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors

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

#include "selfplay/game.h"

namespace lczero {

class Evaluator {
 public:
  virtual ~Evaluator() = default;
  // Run before each batch before any Gather.
  virtual void Reset(const PlayerOptions& player) = 0;
  // Run for each tree.
  virtual void Gather(classic::NodeTree* tree) = 0;
  // Run once between Gather and Move.
  virtual void Run() = 0;
  // Run for each tree in the same order as Gather.
  virtual void MakeBestMove(classic::NodeTree* tree) = 0;
};

// Plays a bunch of games vs itself.
class MultiSelfPlayGames {
 public:
  // Player options may point to the same network/cache/etc.
  MultiSelfPlayGames(PlayerOptions player1, PlayerOptions player2,
                     const std::vector<Opening>& openings,
                     SyzygyTablebase* syzygy_tb, bool use_value);

  // Starts the games and blocks until all games are finished.
  void Play();
  // Aborts the game currently played, doesn't matter if it's synchronous or
  // not.
  void Abort();

  GameResult GetGameResult(int index) const { return results_[index]; }

  std::vector<Move> GetMoves(int index) const {
    std::vector<Move> moves;
    bool flip = !trees_[index]->IsBlackToMove();
    for (classic::Node* node = trees_[index]->GetCurrentHead();
         node != trees_[index]->GetGameBeginNode(); node = node->GetParent()) {
      moves.push_back(node->GetParent()->GetEdgeToNode(node)->GetMove(flip));
      flip = !flip;
    }
    std::reverse(moves.begin(), moves.end());
    return moves;
  }

 private:
  // options_[0] is for white player, [1] for black.
  PlayerOptions options_[2];
  // Node tree for player1 and player2. If the tree is shared between players,
  // tree_[0] == tree_[1].
  std::vector<std::shared_ptr<classic::NodeTree>> trees_;
  std::vector<GameResult> results_;
  bool abort_ = false;
  std::mutex mutex_;
  SyzygyTablebase* syzygy_tb_;
  std::unique_ptr<Evaluator> eval_;
};

}  // namespace lczero

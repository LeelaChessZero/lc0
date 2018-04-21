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

#include "mcts/search.h"
#include "neural/cache.h"
#include "neural/network.h"
#include "optionsparser.h"
#include "uciloop.h"

namespace lczero {

struct PlayerOptions {
  // Network to use by the player.
  Network* network;
  // Callback when player moves.
  BestMoveInfo::Callback best_move_callback;
  // Callback when player outputs info.
  ThinkingInfo::Callback info_callback;
  // NNcache to use.
  NNCache* cache;
  // User options dictionary.
  const OptionsDict* uci_options;
  // Limits to use for every move.
  SearchLimits search_limits;
};

// Plays a single game vs itself.
class SelfPlayGame {
 public:
  // Player options may point to the same network/cache/etc.
  // If shared_tree is true, search tree is reused between players.
  // (useful for training games). Otherwise the tree is separate for black
  // and white (useful i.e. when they use different networks).
  SelfPlayGame(PlayerOptions player1, PlayerOptions player2, bool shared_tree,
               NodePool* node_pool_);

  // Starts the game and blocks until the game is finished.
  void Play(int white_threads, int black_threads);
  // Aborts the game currently played, doesn't matter if it's synchronous or
  // not.
  void Abort();

  GameInfo::GameResult GetGameResult() const { return game_result_; }

 private:
  // options_[0] is for white player, [1] for black.
  PlayerOptions options_[2];
  NodePool* node_pool_;
  // Node tree for player1 and player2. If the tree is shared between players,
  // tree_[0] == tree_[1].
  std::shared_ptr<NodeTree> tree_[2];
  // Search that is currently in progress. Stored in members so that Abort()
  // can stop it.
  std::unique_ptr<Search> search_;
  bool abort_ = false;
  GameInfo::GameResult game_result_ = GameInfo::UNDECIDED;
  std::mutex mutex_;
};

}  // namespace lczero
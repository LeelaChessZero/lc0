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

#include "chess/position.h"
#include "chess/uciloop.h"
#include "mcts/search.h"
#include "neural/cache.h"
#include "neural/network.h"
#include "utils/optionsparser.h"

namespace lczero {

struct SelfPlayLimits : SearchLimits {
  // Movetime
  std::int64_t movetime;
};

struct PlayerOptions {
  // Callback when player moves.
  BestMoveInfo::Callback best_move_callback;
  // Callback when player outputs info.
  ThinkingInfo::Callback info_callback;
  // NNcache to use.
  NNCache* cache;
  // User options dictionary.
  const OptionsDict* uci_options;
  // Limits to use for every move.
  SelfPlayLimits search_limits;
};

// Plays a single game vs itself.
class SelfPlayGame {
 public:
  // Player options may point to the same network/cache/etc.
  // If shared_tree is true, search tree is reused between players.
  // (useful for training games). Otherwise the tree is separate for black
  // and white (useful i.e. when they use different networks).
  SelfPlayGame(int game_number, PlayerOptions player1, PlayerOptions player2,
               bool shared_tree, bool enable_resign);

  // Populate command line options that it uses.
  static void PopulateUciParams(OptionsParser* options);

  // Writes training data to a file.
  void WriteTrainingData(TrainingDataWriter* writer) const;

  GameResult GetGameResult() const { return game_result_; }
  std::vector<Move> GetMoves() const;
  // Gets the eval which required the biggest swing up to get the final outcome.
  // Eval is the expected outcome in the range 0<->1.
  float GetWorstEvalForWinnerOrDraw() const;

  // Index of the current game.
  int GetGameNumber() const { return game_number_; }

  // Returns whether resign is enabled.
  bool IsResignEnabled() const { return enable_resign_; }

  // Returns whether white player is currently thinking or about to start
  // thinking.
  bool IsWhiteToMove() const { return !black_to_move_; }

  // Executes part of an iteration before NN evaluation.
  void PrepareBatch(std::unique_ptr<NetworkComputation>);

  // Executes part of an iteration after NN evalutation.
  void ProcessBatch();

  // Returns whether the game is finished, so no more iterations have to be
  // called.
  bool IsGameFinished() const { return game_result_ != GameResult::UNDECIDED; }

 private:
  void PrepareForNextMove();
  void ProcessMoveEnd();

  // options_[0] is for white player, [1] for black.
  PlayerOptions options_[2];
  // Node tree for player1 and player2. If the tree is shared between players,
  // tree_[0] == tree_[1].
  std::shared_ptr<NodeTree> tree_[2];

  GameResult game_result_ = GameResult::UNDECIDED;
  // Track minimum eval for each player so that GetWorstEvalForWinnerOrDraw()
  // can be calculated after end of game.
  float min_eval_[2] = {1.0f, 1.0f};
  std::mutex mutex_;

  // Training data to send.
  std::vector<V3TrainingData> training_data_;

  bool black_to_move_ = false;
  bool search_ended_ = false;

  std::unique_ptr<Search> search_;
  std::unique_ptr<SearchWorker> search_worker_;

  const int game_number_;
  const bool enable_resign_;
};

}  // namespace lczero

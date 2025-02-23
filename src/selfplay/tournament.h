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

#include <list>

#include "chess/pgn.h"
#include "neural/backend.h"
#include "neural/factory.h"
#include "selfplay/game.h"
#include "selfplay/multigame.h"
#include "utils/mutex.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero {

// Runs many selfplay games, possibly in parallel.
class SelfPlayTournament {
 public:
  SelfPlayTournament(const OptionsDict& options,
                     CallbackUciResponder::BestMoveCallback best_move_info,
                     CallbackUciResponder::ThinkingCallback thinking_info,
                     GameInfo::Callback game_info,
                     TournamentInfo::Callback tournament_info);

  // Populate command line options that it uses.
  static void PopulateOptions(OptionsParser* options);

  // Starts worker threads and exists immediately.
  void StartAsync();

  // Starts tournament and waits until it finishes.
  void RunBlocking();

  // Blocks until all worker threads finish.
  void Wait();

  // Tells worker threads to finish ASAP. Does not block.
  void Abort();

  // Stops any more games from starting, in progress games will complete.
  void Stop();

  // If there are ongoing games, aborts and waits.
  ~SelfPlayTournament();

 private:
  void Worker();
  void PlayOneGame(int game_id);
  void PlayMultiGames(int game_id, size_t game_count);
  void SaveResults() REQUIRES(mutex_);

  Mutex mutex_;
  // Whether first game will be black for player1.
  bool first_game_black_ GUARDED_BY(mutex_) = false;
  std::unique_ptr<SyzygyTablebase> syzygy_tb_ GUARDED_BY(mutex_);
  std::vector<Opening> discard_pile_ GUARDED_BY(mutex_);
  // Number of games which already started.
  int games_count_ GUARDED_BY(mutex_) = 0;
  bool abort_ GUARDED_BY(mutex_) = false;
  std::vector<Opening> openings_ GUARDED_BY(mutex_);
  // Games in progress. Exposed here to be able to abort them in case if
  // Abort(). Stored as list and not vector so that threads can keep iterators
  // to them and not worry that it becomes invalid.
  std::list<std::unique_ptr<SelfPlayGame>> games_ GUARDED_BY(mutex_);
  std::list<std::unique_ptr<MultiSelfPlayGames>> multigames_ GUARDED_BY(mutex_);
  // Place to store tournament stats.
  TournamentInfo tournament_info_ GUARDED_BY(mutex_);

  Mutex threads_mutex_;
  std::vector<std::thread> threads_ GUARDED_BY(threads_mutex_);

  // Map from the backend configuration to a network.
  std::map<NetworkFactory::BackendConfiguration, std::unique_ptr<Backend>>
      backends_;
  // [player1 or player2][white or black].
  const OptionsDict player_options_[2][2];
  SelfPlayLimits search_limits_[2][2];

  CallbackUciResponder::BestMoveCallback best_move_callback_;
  CallbackUciResponder::ThinkingCallback info_callback_;
  GameInfo::Callback game_callback_;
  TournamentInfo::Callback tournament_callback_;
  const int kTotalGames;
  const bool kShareTree;
  const size_t kParallelism;
  const bool kTraining;
  const float kResignPlaythrough;
  const int kPolicyGamesSize;
  const int kValueGamesSize;
  int multi_games_size_;
  const std::string kTournamentResultsFile;
  const float kDiscardedStartChance;
};

}  // namespace lczero

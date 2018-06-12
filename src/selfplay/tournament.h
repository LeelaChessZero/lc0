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

#include <list>
#include "selfplay/game.h"
#include "utils/mutex.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero {

// Runs many selfplay games, possibly in parallel.
class SelfPlayTournament {
 public:
  SelfPlayTournament(const OptionsDict& options,
                     BestMoveInfo::Callback best_move_info,
                     ThinkingInfo::Callback thinking_info,
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

  // If there are ongoing games, aborts and waits.
  ~SelfPlayTournament();

 private:
  void Worker();
  void PlayOneGame(int game_id);

  Mutex mutex_;
  // Whether next game will be black for player1.
  bool next_game_black_ GUARDED_BY(mutex_) = false;
  // Number of games which already started.
  int games_count_ GUARDED_BY(mutex_) = 0;
  bool abort_ GUARDED_BY(mutex_) = false;
  // Games in progress. Exposed here to be able to abort them in case if
  // Abort(). Stored as list and not vector so that threads can keep iterators
  // to them and not worry that it becomes invalid.
  std::list<std::unique_ptr<SelfPlayGame>> games_ GUARDED_BY(mutex_);
  // Place to store tournament stats.
  TournamentInfo tournament_info_ GUARDED_BY(mutex_);

  Mutex threads_mutex_;
  std::vector<std::thread> threads_ GUARDED_BY(threads_mutex_);

  // All those are [0] for player1 and [1] for player2
  // Shared pointers for both players may point to the same object.
  std::shared_ptr<Network> networks_[2];
  std::shared_ptr<NNCache> cache_[2];
  const OptionsDict player_options_[2];
  SearchLimits search_limits_[2];

  BestMoveInfo::Callback best_move_callback_;
  ThinkingInfo::Callback info_callback_;
  GameInfo::Callback game_callback_;
  TournamentInfo::Callback tournament_callback_;
  const int kThreads[2];
  const int kTotalGames;
  const bool kShareTree;
  const size_t kParallelism;
  const bool kTraining;
  const float kResignPlaythrough;
};

}  // namespace lczero

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

#include "selfplay/tournament.h"
#include <random>
#include "mcts/search.h"
#include "neural/loader.h"
#include "neural/network_mux.h"
#include "neural/network_tf.h"
#include "optionsparser.h"
#include "selfplay/game.h"

namespace lczero {

namespace {
const char* kShareTreesStr = "Share game trees for two players";
const char* kTotalGamesStr = "Number of games to play";
const char* kParallelGamesStr = "Number of games to play in parallel";
const char* kGpuThreadsStr = "Number of GPU threads";
const char* kMaxGpuBatchStr = "Maximum GPU batch size";
const char* kThreadsStr = "Number of CPU threads for every game";
const char* kNnCacheSizeStr = "NNCache size";
const char* kNetFileStr = "Network weights file path";
const char* kNodesStr = "Number of nodes per move to search";
const char* kTimeMsStr = "Time per move, in milliseconds";
// Value for network autodiscover.
const char* kAutoDiscover = "<autodiscover>";

}  // namespace

void SelfPlayTournament::PopulateOptions(OptionsParser* options) {
  options->AddContext("player1");
  options->AddContext("player2");

  options->Add<CheckOption>(kShareTreesStr, "share-trees") = false;
  options->Add<SpinOption>(kTotalGamesStr, -1, 999999, "games") = -1;
  options->Add<SpinOption>(kParallelGamesStr, 1, 256, "parallelism") = 1;
  options->Add<SpinOption>(kGpuThreadsStr, 1, 16, "gpu-threads") = 1;
  options->Add<SpinOption>(kMaxGpuBatchStr, 1, 1024, "gpu-batch") = 128;
  options->Add<SpinOption>(kThreadsStr, 1, 8, "threads", 't') = 1;
  options->Add<SpinOption>(kNnCacheSizeStr, 0, 999999999, "nncache") = 200000;
  options->Add<StringOption>(kNetFileStr, "weights", 'w') = kAutoDiscover;
  options->Add<SpinOption>(kNnCacheSizeStr, 0, 999999999, "nncache") = 200000;
  options->Add<SpinOption>(kNodesStr, -1, 999999999, "nodes") = -1;
  options->Add<SpinOption>(kTimeMsStr, -1, 999999999, "movetime") = -1;

  Search::PopulateUciParams(options);
}

SelfPlayTournament::SelfPlayTournament(const OptionsDict& options,
                                       BestMoveInfo::Callback best_move_info,
                                       ThinkingInfo::Callback thinking_info,
                                       GameInfo::Callback game_info,
                                       TournamentInfo::Callback tournament_info)
    : player_options_{options.GetSubdict("player1"),
                      options.GetSubdict("player2")},
      best_move_callback_(best_move_info),
      info_callback_(thinking_info),
      game_callback_(game_info),
      tournament_callback_(tournament_info),
      kThreads{
          options.GetSubdict("player1").Get<int>(kThreadsStr),
          options.GetSubdict("player2").Get<int>(kThreadsStr),
      },
      kTotalGames(options.Get<int>(kTotalGamesStr)),
      kShareTree(options.Get<bool>(kShareTreesStr)),
      kParallelism(options.Get<int>(kParallelGamesStr)),
      kGpuThreads(options.Get<int>(kGpuThreadsStr)),
      kMaxGpuBatch(options.Get<int>(kMaxGpuBatchStr)) {
  // If playing just one game, the player1 is white, otherwise randomize.
  if (kTotalGames != 1) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
    next_game_black_ = dis(gen);
  }

  // Initializing networks.
  static const char* kPlayerNames[2] = {"player1", "player2"};
  for (int idx : {0, 1}) {
    // If two players have the same network, no need to load two.
    if (idx == 1 &&
        options.GetSubdict("player1").Get<std::string>(kNetFileStr) ==
            options.GetSubdict("player2").Get<std::string>(kNetFileStr)) {
      networks_[1] = networks_[0];
      break;
    }
    std::string path =
        options.GetSubdict(kPlayerNames[idx]).Get<std::string>(kNetFileStr);
    if (path == kAutoDiscover) {
      path = DiscoveryWeightsFile();
    }
    Weights weights = LoadWeightsFromFile(path);
    auto network = MakeTensorflowNetwork(weights);
    if (kParallelism == 1) {
      // If one game will be run in parallel, no need to mux computations.
      networks_[idx] = std::move(network);
    } else {
      networks_[idx] =
          MakeMuxingNetwork(std::move(network), kGpuThreads, kMaxGpuBatch);
    }
  }

  // Initializing cache.
  cache_[0] = std::make_shared<NNCache>(
      options.GetSubdict("player1").Get<int>(kNnCacheSizeStr));
  if (kShareTree) {
    cache_[1] = cache_[0];
  } else {
    cache_[1] = std::make_shared<NNCache>(
        options.GetSubdict("player2").Get<int>(kNnCacheSizeStr));
  }

  // SearchLimits.
  for (int idx : {0, 1}) {
    search_limits_[idx].nodes =
        options.GetSubdict(kPlayerNames[idx]).Get<int>(kNodesStr);
    search_limits_[idx].time_ms =
        options.GetSubdict(kPlayerNames[idx]).Get<int>(kTimeMsStr);
  }
}

void SelfPlayTournament::PlayOneGame(int game_number) {
  bool player1_black;  // Whether player1 will player as black in this game.
  {
    Mutex::Lock lock(mutex_);
    player1_black = next_game_black_;
    next_game_black_ = !next_game_black_;
  }
  const int color_idx[2] = {player1_black ? 1 : 0, player1_black ? 0 : 1};

  PlayerOptions options[2];

  for (int pl_idx : {0, 1}) {
    // Populate per-player options.
    PlayerOptions& opt = options[color_idx[pl_idx]];
    opt.network = networks_[pl_idx].get();
    opt.cache = cache_[pl_idx].get();
    opt.uci_options = &player_options_[pl_idx];
    opt.search_limits = search_limits_[pl_idx];

    opt.best_move_callback = [this, game_number, pl_idx,
                              player1_black](const BestMoveInfo& info) {
      BestMoveInfo rich_info = info;
      rich_info.player = pl_idx + 1;
      rich_info.is_black = player1_black ? pl_idx == 0 : pl_idx != 0;
      rich_info.game_id = game_number;
      best_move_callback_(rich_info);
    };

    opt.info_callback = [this, game_number, pl_idx,
                         player1_black](const ThinkingInfo& info) {
      ThinkingInfo rich_info = info;
      rich_info.player = pl_idx + 1;
      rich_info.is_black = player1_black ? pl_idx == 0 : pl_idx != 0;
      rich_info.game_id = game_number;
      info_callback_(rich_info);
    };
  }

  // Iterator to store the game in. Have to keep it so that later we can
  // delete it. Need to expose it in games_ member variable only because
  // of possible Abort() that should stop them all.
  std::list<std::unique_ptr<SelfPlayGame>>::iterator game_iter;
  {
    Mutex::Lock lock(mutex_);
    games_.emplace_front(std::make_unique<SelfPlayGame>(
        options[0], options[1], kShareTree, &node_pool_));
    game_iter = games_.begin();
  }
  auto& game = **game_iter;

  // PLAY GAME!
  game.Play(kThreads[color_idx[0]], kThreads[color_idx[1]]);

  // If game was aborted, it's still undecided.
  if (game.GetGameResult() != GameInfo::UNDECIDED) {
    // Game callback.
    GameInfo game_info;
    game_info.game_result = game.GetGameResult();
    game_info.is_black = player1_black;
    game_info.game_id = game_number;
    game_callback_(game_info);

    // Update tournament stats.
    {
      Mutex::Lock lock(mutex_);
      int result = game.GetGameResult() == GameInfo::DRAW
                       ? 1
                       : game.GetGameResult() == GameInfo::WHITE_WON ? 0 : 2;
      if (player1_black) result = 2 - result;
      ++tournament_info_.results[result][player1_black ? 1 : 0];
      tournament_callback_(tournament_info_);
    }
  }

  {
    Mutex::Lock lock(mutex_);
    games_.erase(game_iter);
  }
}

void SelfPlayTournament::Worker() {
  // Play games while game limit is not reached (or while not aborted).
  while (true) {
    int game_id;
    {
      Mutex::Lock lock(mutex_);
      if (abort_) break;
      if (kTotalGames != -1 && games_count_ >= kTotalGames) break;
      game_id = games_count_++;
    }
    PlayOneGame(game_id);
  }
}

void SelfPlayTournament::StartAsync() {
  Mutex::Lock lock(threads_mutex_);
  while (threads_.size() < kParallelism) {
    threads_.emplace_back([&]() { Worker(); });
  }
}

void SelfPlayTournament::RunBlocking() {
  if (kParallelism == 1) {
    // No need for multiple threads if there is one worker.
    Worker();
  } else {
    StartAsync();
    Wait();
  }
}

void SelfPlayTournament::Wait() {
  Mutex::Lock lock(threads_mutex_);
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
}

void SelfPlayTournament::Abort() {
  Mutex::Lock lock(mutex_);
  abort_ = true;
  for (auto& game : games_)
    if (game) game->Abort();
}

SelfPlayTournament::~SelfPlayTournament() {
  Abort();
  Wait();
}

}  // namespace lczero
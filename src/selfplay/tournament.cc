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
#include "mcts/search.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "selfplay/game.h"
#include "utils/optionsparser.h"
#include "utils/random.h"

namespace lczero {

namespace {
const char* kShareTreesStr = "Share game trees for two players";
const char* kTotalGamesStr = "Number of games to play";
const char* kParallelGamesStr = "Number of games to play in parallel";
const char* kThreadsStr = "Number of CPU threads for every game";
const char* kNnCacheSizeStr = "NNCache size";
const char* kNetFileStr = "Network weights file path";
const char* kPlayoutsStr = "Number of playouts per move to search";
const char* kVisitsStr = "Number of visits per move to search";
const char* kTimeMsStr = "Time per move, in milliseconds";
const char* kTrainingStr = "Write training data";
const char* kNnBackendStr = "NN backend to use";
const char* kNnBackendOptionsStr = "NN backend parameters";
const char* kVerboseThinkingStr = "Show verbose thinking messages";
const char* kResignPlaythroughStr =
              "The percentage of games which ignore resign";

// Value for network autodiscover.
const char* kAutoDiscover = "<autodiscover>";

}  // namespace

void SelfPlayTournament::PopulateOptions(OptionsParser* options) {
  options->AddContext("player1");
  options->AddContext("player2");

  options->Add<BoolOption>(kShareTreesStr, "share-trees") = true;
  options->Add<IntOption>(kTotalGamesStr, -1, 999999, "games") = -1;
  options->Add<IntOption>(kParallelGamesStr, 1, 256, "parallelism") = 8;
  options->Add<IntOption>(kThreadsStr, 1, 8, "threads", 't') = 1;
  options->Add<IntOption>(kNnCacheSizeStr, 0, 999999999, "nncache") = 200000;
  options->Add<StringOption>(kNetFileStr, "weights", 'w') = kAutoDiscover;
  options->Add<IntOption>(kPlayoutsStr, -1, 999999999, "playouts", 'p') = -1;
  options->Add<IntOption>(kVisitsStr, -1, 999999999, "visits", 'v') = -1;
  options->Add<IntOption>(kTimeMsStr, -1, 999999999, "movetime") = -1;
  options->Add<BoolOption>(kTrainingStr, "training") = false;
  const auto backends = NetworkFactory::Get()->GetBackendsList();
  options->Add<ChoiceOption>(kNnBackendStr, backends, "backend") =
      "multiplexing";
  options->Add<StringOption>(kNnBackendOptionsStr, "backend-opts");
  options->Add<BoolOption>(kVerboseThinkingStr, "verbose-thinking") = false;
  options->Add<FloatOption>(kResignPlaythroughStr, 0.0f, 100.0f,
                            "resign-playthrough") = 0.0f;

  Search::PopulateUciParams(options);
  SelfPlayGame::PopulateUciParams(options);
  auto defaults = options->GetMutableDefaultsOptions();
  defaults->Set<int>(Search::kMiniBatchSizeStr, 32);     // Minibatch size
  defaults->Set<bool>(Search::kSmartPruningStr, false);  // No smart pruning
  defaults->Set<float>(Search::kTemperatureStr, 1.0);    // Temperature = 1.0
  defaults->Set<bool>(Search::kNoiseStr, true);          // Dirichlet noise
  defaults->Set<float>(Search::kFpuReductionStr, 0.0);   // No FPU reduction.
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
      kTraining(options.Get<bool>(kTrainingStr)),
      kResignPlaythrough(options.Get<float>(kResignPlaythroughStr)) {
  // If playing just one game, the player1 is white, otherwise randomize.
  if (kTotalGames != 1) {
    next_game_black_ = Random::Get().GetBool();
  }

  static const char* kPlayerNames[2] = {"player1", "player2"};
  // Initializing networks.
  for (int idx : {0, 1}) {
    // If two players have the same network, no need to load two.
    if (idx == 1) {
      bool network_identical = true;
      for (const auto& option_str :
           {kNetFileStr, kNnBackendStr, kNnBackendOptionsStr}) {
        if (options.GetSubdict("player1").Get<std::string>(option_str) !=
            options.GetSubdict("player2").Get<std::string>(option_str)) {
          network_identical = false;
          break;
        }
      }
      if (network_identical) {
        networks_[1] = networks_[0];
        break;
      }
    }

    std::string path =
        options.GetSubdict(kPlayerNames[idx]).Get<std::string>(kNetFileStr);
    if (path == kAutoDiscover) {
      path = DiscoveryWeightsFile();
    }
    Weights weights = LoadWeightsFromFile(path);
    std::string backend =
        options.GetSubdict(kPlayerNames[idx]).Get<std::string>(kNnBackendStr);
    std::string backend_options = options.GetSubdict(kPlayerNames[idx])
                                      .Get<std::string>(kNnBackendOptionsStr);

    OptionsDict network_options = OptionsDict::FromString(
        backend_options, &options.GetSubdict(kPlayerNames[idx]));

    networks_[idx] =
        NetworkFactory::Get()->Create(backend, weights, network_options);
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
    search_limits_[idx].playouts =
        options.GetSubdict(kPlayerNames[idx]).Get<int>(kPlayoutsStr);
    search_limits_[idx].visits =
        options.GetSubdict(kPlayerNames[idx]).Get<int>(kVisitsStr);
    search_limits_[idx].time_ms =
        options.GetSubdict(kPlayerNames[idx]).Get<int>(kTimeMsStr);

    if (search_limits_[idx].playouts == -1 &&
        search_limits_[idx].visits == -1 && search_limits_[idx].time_ms == -1) {
      throw Exception(
          "Please define --visits, --playouts or --movetime, otherwise it's "
          "not clear when to stop search.");
    }
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

  ThinkingInfo last_thinking_info;
  last_thinking_info.depth = -1;
  for (int pl_idx : {0, 1}) {
    const bool verbose_thinking =
        player_options_[pl_idx].Get<bool>(kVerboseThinkingStr);
    // Populate per-player options.
    PlayerOptions& opt = options[color_idx[pl_idx]];
    opt.network = networks_[pl_idx].get();
    opt.cache = cache_[pl_idx].get();
    opt.uci_options = &player_options_[pl_idx];
    opt.search_limits = search_limits_[pl_idx];

    // "bestmove" callback.
    opt.best_move_callback = [this, game_number, pl_idx, player1_black,
                              verbose_thinking,
                              &last_thinking_info](const BestMoveInfo& info) {
      // In non-verbose mode, output the last "info" message.
      if (!verbose_thinking && last_thinking_info.depth >= 0) {
        info_callback_(last_thinking_info);
        last_thinking_info.depth = -1;
      }
      BestMoveInfo rich_info = info;
      rich_info.player = pl_idx + 1;
      rich_info.is_black = player1_black ? pl_idx == 0 : pl_idx != 0;
      rich_info.game_id = game_number;
      best_move_callback_(rich_info);
    };

    opt.info_callback = [this, game_number, pl_idx, player1_black,
                         verbose_thinking,
                         &last_thinking_info](const ThinkingInfo& info) {
      ThinkingInfo rich_info = info;
      rich_info.player = pl_idx + 1;
      rich_info.is_black = player1_black ? pl_idx == 0 : pl_idx != 0;
      rich_info.game_id = game_number;
      if (verbose_thinking) {
        info_callback_(rich_info);
      } else {
        // In non-verbose mode, remeber the last "info" message.
        last_thinking_info = rich_info;
      }
    };
  }

  // Iterator to store the game in. Have to keep it so that later we can
  // delete it. Need to expose it in games_ member variable only because
  // of possible Abort() that should stop them all.
  std::list<std::unique_ptr<SelfPlayGame>>::iterator game_iter;
  {
    Mutex::Lock lock(mutex_);
    games_.emplace_front(
        std::make_unique<SelfPlayGame>(options[0], options[1], kShareTree));
    game_iter = games_.begin();
  }
  auto& game = **game_iter;

  // If kResignPlaythrough == 0, then this comparison is unconditionally true
  bool enable_resign = Random::Get().GetFloat(100.0f) >= kResignPlaythrough;

  // PLAY GAME!
  game.Play(kThreads[color_idx[0]], kThreads[color_idx[1]], enable_resign);

  // If game was aborted, it's still undecided.
  if (game.GetGameResult() != GameResult::UNDECIDED) {
    // Game callback.
    GameInfo game_info;
    game_info.game_result = game.GetGameResult();
    game_info.is_black = player1_black;
    game_info.game_id = game_number;
    game_info.moves = game.GetMoves();
    if (kTraining) {
      TrainingDataWriter writer(game_number);
      game.WriteTrainingData(&writer);
      writer.Finalize();
      game_info.training_filename = writer.GetFileName();
    }
    game_callback_(game_info);

    // Update tournament stats.
    {
      Mutex::Lock lock(mutex_);
      int result = game.GetGameResult() == GameResult::DRAW
                       ? 1
                       : game.GetGameResult() == GameResult::WHITE_WON ? 0 : 2;
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
    Mutex::Lock lock(mutex_);
    if (!abort_) {
      tournament_info_.finished = true;
      tournament_callback_(tournament_info_);
    }
  } else {
    StartAsync();
    Wait();
  }
}

void SelfPlayTournament::Wait() {
  {
    Mutex::Lock lock(threads_mutex_);
    while (!threads_.empty()) {
      threads_.back().join();
      threads_.pop_back();
    }
  }
  {
    Mutex::Lock lock(mutex_);
    if (!abort_) {
      tournament_info_.finished = true;
      tournament_callback_(tournament_info_);
    }
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

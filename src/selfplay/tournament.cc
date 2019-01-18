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

#include "selfplay/tournament.h"
#include "mcts/search.h"
#include "neural/factory.h"
#include "selfplay/game.h"
#include "utils/optionsparser.h"
#include "utils/random.h"

namespace lczero {
namespace {
const OptionId kShareTreesId{"share-trees", "ShareTrees",
                             "When on, game tree is shared for two players; "
                             "when off, each side has a separate tree."};
const OptionId kTotalGamesId{"games", "Games", "Number of games to play."};
const OptionId kParallelGamesId{"parallelism", "Parallelism",
                                "Number of games to play in parallel."};
const OptionId kThreadsId{
    "threads", "Threads",
    "Number of (CPU) worker threads to use for every game,", 't'};
const OptionId kNnCacheSizeId{
    "nncache", "NNCache",
    "Number of positions to store in a memory cache. A large cache can speed "
    "up searching, but takes memory."};
const OptionId kPlayoutsId{"playouts", "Playouts",
                           "Number of playouts per move to search."};
const OptionId kVisitsId{"visits", "Visits",
                         "Number of visits per move to search."};
const OptionId kTimeMsId{"movetime", "MoveTime",
                         "Time per move, in milliseconds."};
const OptionId kTrainingId{
    "training", "Training",
    "Enables writing training data. The training data is stored into a "
    "temporary subdirectory that the engine creates."};
const OptionId kVerboseThinkingId{"verbose-thinking", "VerboseThinking",
                                  "Show verbose thinking messages."};
const OptionId kResignPlaythroughId{
    "resign-playthrough", "ResignPlaythrough",
    "The percentage of games which ignore resign."};

}  // namespace

void SelfPlayTournament::PopulateOptions(OptionsParser* options) {
  options->AddContext("player1");
  options->AddContext("player2");

  options->Add<BoolOption>(kShareTreesId) = true;
  options->Add<IntOption>(kTotalGamesId, -1, 999999) = -1;
  options->Add<IntOption>(kParallelGamesId, 1, 256) = 8;
  options->Add<IntOption>(kThreadsId, 1, 8) = 1;
  options->Add<IntOption>(kNnCacheSizeId, 0, 999999999) = 200000;
  options->Add<IntOption>(kPlayoutsId, -1, 999999999) = -1;
  options->Add<IntOption>(kVisitsId, -1, 999999999) = -1;
  options->Add<IntOption>(kTimeMsId, -1, 999999999) = -1;
  options->Add<BoolOption>(kTrainingId) = false;
  options->Add<BoolOption>(kVerboseThinkingId) = false;
  options->Add<FloatOption>(kResignPlaythroughId, 0.0f, 100.0f) = 0.0f;

  NetworkFactory::PopulateOptions(options);
  SearchParams::Populate(options);
  SelfPlayGame::PopulateUciParams(options);
  auto defaults = options->GetMutableDefaultsOptions();
  defaults->Set<int>(SearchParams::kMiniBatchSizeId.GetId(), 32);
  defaults->Set<float>(SearchParams::kCpuctId.GetId(), 1.2f);
  defaults->Set<float>(SearchParams::kCpuctFactorId.GetId(), 0.0f);
  defaults->Set<float>(SearchParams::kPolicySoftmaxTempId.GetId(), 1.0f);
  defaults->Set<int>(SearchParams::kMaxCollisionVisitsId.GetId(), 1);
  defaults->Set<int>(SearchParams::kMaxCollisionEventsId.GetId(), 1);
  defaults->Set<int>(SearchParams::kCacheHistoryLengthId.GetId(), 7);
  defaults->Set<bool>(SearchParams::kOutOfOrderEvalId.GetId(), false);
  defaults->Set<float>(SearchParams::kSmartPruningFactorId.GetId(), 0.0f);
  defaults->Set<float>(SearchParams::kTemperatureId.GetId(), 1.0f);
  defaults->Set<bool>(SearchParams::kNoiseId.GetId(), true);
  defaults->Set<float>(SearchParams::kFpuReductionId.GetId(), 0.0f);
  defaults->Set<std::string>(SearchParams::kHistoryFillId.GetId(), "no");
  defaults->Set<std::string>(NetworkFactory::kBackendId.GetId(),
                             "multiplexing");
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
          options.GetSubdict("player1").Get<int>(kThreadsId.GetId()),
          options.GetSubdict("player2").Get<int>(kThreadsId.GetId()),
      },
      kTotalGames(options.Get<int>(kTotalGamesId.GetId())),
      kShareTree(options.Get<bool>(kShareTreesId.GetId())),
      kParallelism(options.Get<int>(kParallelGamesId.GetId())),
      kTraining(options.Get<bool>(kTrainingId.GetId())),
      kResignPlaythrough(options.Get<float>(kResignPlaythroughId.GetId())) {
  // If playing just one game, the player1 is white, otherwise randomize.
  if (kTotalGames != 1) {
    next_game_black_ = Random::Get().GetBool();
  }

  static const char* kPlayerNames[2] = {"player1", "player2"};
  // Initializing networks.
  const auto& player1_opts = options.GetSubdict(kPlayerNames[0]);
  const auto& player2_opts = options.GetSubdict(kPlayerNames[1]);
  networks_[0] = NetworkFactory::LoadNetwork(player1_opts);
  networks_[1] = NetworkFactory::BackendConfiguration(player1_opts) ==
                         NetworkFactory::BackendConfiguration(player2_opts)
                     ? networks_[0]
                     : NetworkFactory::LoadNetwork(player2_opts);

  // Initializing cache.
  cache_[0] = std::make_shared<NNCache>(
      options.GetSubdict("player1").Get<int>(kNnCacheSizeId.GetId()));
  if (kShareTree) {
    cache_[1] = cache_[0];
  } else {
    cache_[1] = std::make_shared<NNCache>(
        options.GetSubdict("player2").Get<int>(kNnCacheSizeId.GetId()));
  }

  // SearchLimits.
  for (int idx : {0, 1}) {
    search_limits_[idx].playouts =
        options.GetSubdict(kPlayerNames[idx]).Get<int>(kPlayoutsId.GetId());
    search_limits_[idx].visits =
        options.GetSubdict(kPlayerNames[idx]).Get<int>(kVisitsId.GetId());
    search_limits_[idx].movetime =
        options.GetSubdict(kPlayerNames[idx]).Get<int>(kTimeMsId.GetId());

    if (search_limits_[idx].playouts == -1 &&
        search_limits_[idx].visits == -1 &&
        search_limits_[idx].movetime == -1) {
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

  std::vector<ThinkingInfo> last_thinking_info;
  for (int pl_idx : {0, 1}) {
    const bool verbose_thinking =
        player_options_[pl_idx].Get<bool>(kVerboseThinkingId.GetId());
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
      if (!verbose_thinking && !last_thinking_info.empty()) {
        info_callback_(last_thinking_info);
        last_thinking_info.clear();
      }
      BestMoveInfo rich_info = info;
      rich_info.player = pl_idx + 1;
      rich_info.is_black = player1_black ? pl_idx == 0 : pl_idx != 0;
      rich_info.game_id = game_number;
      best_move_callback_(rich_info);
    };

    opt.info_callback =
        [this, game_number, pl_idx, player1_black, verbose_thinking,
         &last_thinking_info](const std::vector<ThinkingInfo>& infos) {
          std::vector<ThinkingInfo> rich_info = infos;
          for (auto& info : rich_info) {
            info.player = pl_idx + 1;
            info.is_black = player1_black ? pl_idx == 0 : pl_idx != 0;
            info.game_id = game_number;
          }
          if (verbose_thinking) {
            info_callback_(rich_info);
          } else {
            // In non-verbose mode, remember the last "info" messages.
            last_thinking_info = std::move(rich_info);
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
  game.Play(kThreads[color_idx[0]], kThreads[color_idx[1]], kTraining,
            enable_resign);

  // If game was aborted, it's still undecided.
  if (game.GetGameResult() != GameResult::UNDECIDED) {
    // Game callback.
    GameInfo game_info;
    game_info.game_result = game.GetGameResult();
    game_info.is_black = player1_black;
    game_info.game_id = game_number;
    game_info.moves = game.GetMoves();
    if (!enable_resign) {
      game_info.min_false_positive_threshold =
          game.GetWorstEvalForWinnerOrDraw();
    }
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

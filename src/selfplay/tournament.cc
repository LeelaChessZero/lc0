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
#include "neural/network_st_batch.h"
#include "selfplay/game.h"
#include "utils/optionsparser.h"
#include "utils/random.h"

namespace lczero {
namespace {
const OptionId kShareTreesId{"share-trees", "ShareTrees",
                             "When on, game tree is shared for two players; "
                             "when off, each side has a separate tree."};
const OptionId kTotalGamesId{"games", "Games", "Number of games to play."};
const OptionId kParallelismId{"parallelism", "Parallelism",
                              "Number of selfplay threads to run in parallel."};
const OptionId kThreadParallelismId{
    "parallelism-per-thread", "ParallelismPerThread",
    "Number of games to play in parallel within single thread."};
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
  options->Add<IntOption>(kParallelismId, 1, 256) = 2;
  options->Add<IntOption>(kThreadParallelismId, 1, 256) = 16;
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
  defaults->Set<int>(SearchParams::kMiniBatchSizeId.GetId(), 4);
  defaults->Set<int>(SearchParams::kMaxPrefetchBatchId.GetId(), 8);
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
      kTotalGames(options.Get<int>(kTotalGamesId.GetId())),
      kShareTree(options.Get<bool>(kShareTreesId.GetId())),
      kThreads(options.Get<int>(kParallelismId.GetId())),
      kThreadParallelism(options.Get<int>(kThreadParallelismId.GetId())),
      kTraining(options.Get<bool>(kTrainingId.GetId())),
      kResignPlaythrough(options.Get<float>(kResignPlaythroughId.GetId())),
      // If playing just one game, the player1 is white, otherwise randomize.
      first_game_is_flipped_(kTotalGames == 1 ? false
                                              : Random::Get().GetBool()) {
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

// Creates a fresh SelfPlayGame.
std::unique_ptr<SelfPlayGame> SelfPlayTournament::CreateNewGame(
    int game_number) {
  const bool player1_black = (game_number % 2) != first_game_is_flipped_;
  const int color_idx[2] = {player1_black ? 1 : 0, player1_black ? 0 : 1};

  PlayerOptions options[2];

  for (int pl_idx : {0, 1}) {
    const bool verbose_thinking =
        player_options_[pl_idx].Get<bool>(kVerboseThinkingId.GetId());
    // Populate per-player options.
    PlayerOptions& opt = options[color_idx[pl_idx]];
    opt.cache = cache_[pl_idx].get();
    opt.uci_options = &player_options_[pl_idx];
    opt.search_limits = search_limits_[pl_idx];

    // "bestmove" callback.
    opt.best_move_callback = [this, game_number, pl_idx,
                              player1_black](const BestMoveInfo& info) {
      BestMoveInfo rich_info = info;
      rich_info.player = pl_idx + 1;
      rich_info.is_black = player1_black ? pl_idx == 0 : pl_idx != 0;
      rich_info.game_id = game_number;
      best_move_callback_(rich_info);
    };

    opt.info_callback =
        [this, game_number, pl_idx, player1_black,
         verbose_thinking](const std::vector<ThinkingInfo>& infos) {
          if (verbose_thinking) {
            std::vector<ThinkingInfo> rich_info = infos;
            for (auto& info : rich_info) {
              info.player = pl_idx + 1;
              info.is_black = player1_black ? pl_idx == 0 : pl_idx != 0;
              info.game_id = game_number;
            }
            info_callback_(rich_info);
          }
        };
  }

  // If kResignPlaythrough == 0, then this comparison is unconditionally true
  bool enable_resign = Random::Get().GetFloat(100.0f) >= kResignPlaythrough;

  return std::make_unique<SelfPlayGame>(game_number, options[color_idx[0]],
                                        options[color_idx[1]], kShareTree,
                                        enable_resign);
}

// Called when the SelfPlayGame is finished, so that it's processed.
void SelfPlayTournament::SendGameReport(const SelfPlayGame& game) {
  const int game_number = game.GetGameNumber();
  const bool player1_black = (game_number % 2) != first_game_is_flipped_;

  // If game was aborted, it's still undecided, in this case don't output any
  // reports.
  if (game.GetGameResult() == GameResult::UNDECIDED) return;

  // Prepare "game ended" callback.
  GameInfo game_info;
  game_info.game_result = game.GetGameResult();
  game_info.is_black = player1_black;
  game_info.game_id = game_number;
  game_info.moves = game.GetMoves();
  if (!game.IsResignEnabled()) {
    // For games which happened to be without resign, gather stats of possible
    // incorrect resign.
    game_info.min_false_positive_threshold = game.GetWorstEvalForWinnerOrDraw();
  }
  if (kTraining) {
    // If training is enabled, write training file.
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

void SelfPlayTournament::Worker() {
  // Games that this thread will handle.
  std::vector<std::unique_ptr<SelfPlayGame>> games(kThreadParallelism);

  // Wrap network(s) with SingleThreadBatchingNetwork to be able to batch
  // several games together.
  std::shared_ptr<SingleThreadBatchingNetwork> nets[2];
  nets[0] = std::make_shared<SingleThreadBatchingNetwork>(networks_[0].get());
  nets[1] =
      networks_[0] == networks_[1]
          ? nets[0]
          : std::make_shared<SingleThreadBatchingNetwork>(networks_[1].get());

  // While there are games to play, play.
  while (!games.empty()) {
    // If aborted, return.
    {
      Mutex::Lock lock(mutex_);
      if (abort_) break;
    }

    bool remove_remaining = false;
    // Go over games to see whether some of them have to be restarted.
    for (size_t i = 0; i < games.size(); ++i) {
      if (games[i]) continue;  // Active game.
      int game_index = 0;
      {
        Mutex::Lock lock(mutex_);
        if (kTotalGames != -1 && games_count_ >= kTotalGames) {
          // If no more games are needed, remove the entry from the vector.
          remove_remaining = true;
          break;
        }
        game_index = games_count_++;
      }
      // Create new game.
      games[i] = CreateNewGame(game_index);
    }

    // If limit is reached (total number of games), then remove all finished
    // games from the vector.
    if (remove_remaining) {
      games.erase(std::remove_if(games.begin(), games.end(),
                                 [](const std::unique_ptr<SelfPlayGame>& x) {
                                   return !x;
                                 }),
                  games.end());
    }

    // Prepare network for a new iteration.
    nets[0]->Reset();
    if (nets[0] != nets[1]) nets[1]->Reset();
    // Initialize for the next batch.
    for (auto& game : games) {
      const bool alternate_color = (game->GetGameNumber() % 2) == 1;
      const bool player1_white = first_game_is_flipped_ == alternate_color;
      const bool white_to_move = game->IsWhiteToMove();
      const auto& network_to_use = nets[white_to_move == player1_white ? 0 : 1];

      game->PrepareBatch(network_to_use->NewComputation());
    }

    // Do actual NN eval.
    nets[0]->ComputeAll();
    if (nets[0] != nets[1]) nets[1]->ComputeAll();
    // Process.
    for (auto& game : games) game->ProcessBatch();

    // Maybe some game is finished. Report it and destroy game object.
    for (auto& game : games) {
      if (game->IsGameFinished()) {
        SendGameReport(*game);
        game.reset();
      }
    }
  }
}

void SelfPlayTournament::StartAsync() {
  Mutex::Lock lock(threads_mutex_);
  while (threads_.size() < kThreads) {
    threads_.emplace_back([&]() { Worker(); });
  }
}

void SelfPlayTournament::RunBlocking() {
  if (kThreads == 1) {
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
}

SelfPlayTournament::~SelfPlayTournament() {
  Abort();
  Wait();
}

}  // namespace lczero

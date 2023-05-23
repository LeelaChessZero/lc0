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

#include "chess/pgn.h"
#include "mcts/search.h"
#include "mcts/stoppers/factory.h"
#include "neural/factory.h"
#include "selfplay/game.h"
#include "utils/optionsparser.h"
#include "utils/random.h"

namespace lczero {
namespace {
const OptionId kShareTreesId{"share-trees", "ShareTrees",
                             "When on, game tree is shared for two players; "
                             "when off, each side has a separate tree."};
const OptionId kTotalGamesId{
    "games", "Games",
    "Number of games to play. -1 to play forever, -2 to play equal to book "
    "length, or double book length if mirrored."};
const OptionId kParallelGamesId{"parallelism", "Parallelism",
                                "Number of games to play in parallel."};
const OptionId kThreadsId{
    "threads", "Threads",
    "Number of (CPU) worker threads to use for every game,", 't'};
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
const OptionId kMoveThinkingId{"move-thinking", "MoveThinking",
                               "Show all the per-move thinking."};
const OptionId kResignPlaythroughId{
    "resign-playthrough", "ResignPlaythrough",
    "The percentage of games which ignore resign."};
const OptionId kDiscardedStartChanceId{
    "discarded-start-chance", "DiscardedStartChance",
    "The percentage chance each game will attempt to start from a position "
    "discarded due to not getting enough visits."};
const OptionId kOpeningsFileId{
    "openings-pgn", "OpeningsPgnFile",
    "A path name to a pgn file containing openings to use."};
const OptionId kOpeningsMirroredId{
    "mirror-openings", "MirrorOpenings",
    "If true, each opening will be played in pairs. "
    "Not really compatible with openings mode random."};
const OptionId kOpeningsModeId{"openings-mode", "OpeningsMode",
                               "A choice of sequential, shuffled, or random."};
const OptionId kSyzygyTablebaseId{
    "syzygy-paths", "SyzygyPath",
    "List of Syzygy tablebase directories, list entries separated by system "
    "separator (\";\" for Windows, \":\" for Linux).",
    's'};

}  // namespace

void SelfPlayTournament::PopulateOptions(OptionsParser* options) {
  options->AddContext("player1");
  options->AddContext("player2");
  options->AddContext("white");
  options->AddContext("black");
  for (const auto context : {"player1", "player2"}) {
    auto* dict = options->GetMutableOptions(context);
    dict->AddSubdict("white")->AddAliasDict(&options->GetOptionsDict("white"));
    dict->AddSubdict("black")->AddAliasDict(&options->GetOptionsDict("black"));
  }

  NetworkFactory::PopulateOptions(options);
  options->Add<IntOption>(kThreadsId, 1, 8) = 1;
  options->Add<IntOption>(kNNCacheSizeId, 0, 999999999) = 2000000;
  SearchParams::Populate(options);

  options->Add<BoolOption>(kShareTreesId) = true;
  options->Add<IntOption>(kTotalGamesId, -2, 999999) = -1;
  options->Add<IntOption>(kParallelGamesId, 1, 256) = 8;
  options->Add<IntOption>(kPlayoutsId, -1, 999999999) = -1;
  options->Add<IntOption>(kVisitsId, -1, 999999999) = -1;
  options->Add<IntOption>(kTimeMsId, -1, 999999999) = -1;
  options->Add<BoolOption>(kTrainingId) = false;
  options->Add<BoolOption>(kVerboseThinkingId) = false;
  options->Add<BoolOption>(kMoveThinkingId) = false;
  options->Add<FloatOption>(kResignPlaythroughId, 0.0f, 100.0f) = 0.0f;
  options->Add<FloatOption>(kDiscardedStartChanceId, 0.0f, 100.0f) = 0.0f;
  options->Add<StringOption>(kOpeningsFileId) = "";
  options->Add<BoolOption>(kOpeningsMirroredId) = false;
  std::vector<std::string> openings_modes = {"sequential", "shuffled",
                                             "random"};
  options->Add<ChoiceOption>(kOpeningsModeId, openings_modes) = "sequential";

  options->Add<StringOption>(kSyzygyTablebaseId);
  SelfPlayGame::PopulateUciParams(options);

  auto defaults = options->GetMutableDefaultsOptions();
  defaults->Set<int>(SearchParams::kMiniBatchSizeId, 32);
  defaults->Set<float>(SearchParams::kCpuctId, 1.2f);
  defaults->Set<float>(SearchParams::kCpuctFactorId, 0.0f);
  defaults->Set<float>(SearchParams::kPolicySoftmaxTempId, 1.0f);
  defaults->Set<int>(SearchParams::kMaxCollisionVisitsId, 1);
  defaults->Set<int>(SearchParams::kMaxCollisionEventsId, 1);
  defaults->Set<int>(SearchParams::kCacheHistoryLengthId, 7);
  defaults->Set<bool>(SearchParams::kOutOfOrderEvalId, false);
  defaults->Set<float>(SearchParams::kTemperatureId, 1.0f);
  defaults->Set<float>(SearchParams::kNoiseEpsilonId, 0.25f);
  defaults->Set<float>(SearchParams::kFpuValueId, 0.0f);
  defaults->Set<std::string>(SearchParams::kHistoryFillId, "no");
  defaults->Set<std::string>(NetworkFactory::kBackendId, "multiplexing");
  defaults->Set<bool>(SearchParams::kStickyEndgamesId, false);
  defaults->Set<bool>(SearchParams::kTwoFoldDrawsId, false);
  defaults->Set<int>(SearchParams::kTaskWorkersPerSearchWorkerId, 0);
}

SelfPlayTournament::SelfPlayTournament(
    const OptionsDict& options,
    CallbackUciResponder::BestMoveCallback best_move_info,
    CallbackUciResponder::ThinkingCallback thinking_info,
    GameInfo::Callback game_info, TournamentInfo::Callback tournament_info)
    : player_options_{{options.GetSubdict("player1").GetSubdict("white"),
                       options.GetSubdict("player1").GetSubdict("black")},
                      {options.GetSubdict("player2").GetSubdict("white"),
                       options.GetSubdict("player2").GetSubdict("black")}},
      best_move_callback_(best_move_info),
      info_callback_(thinking_info),
      game_callback_(game_info),
      tournament_callback_(tournament_info),
      kTotalGames(options.Get<int>(kTotalGamesId)),
      kShareTree(options.Get<bool>(kShareTreesId)),
      kParallelism(options.Get<int>(kParallelGamesId)),
      kTraining(options.Get<bool>(kTrainingId)),
      kResignPlaythrough(options.Get<float>(kResignPlaythroughId)),
      kDiscardedStartChance(options.Get<float>(kDiscardedStartChanceId)) {
  std::string book = options.Get<std::string>(kOpeningsFileId);
  if (!book.empty()) {
    PgnReader book_reader;
    book_reader.AddPgnFile(book);
    openings_ = book_reader.ReleaseGames();
    if (options.Get<std::string>(kOpeningsModeId) == "shuffled") {
      Random::Get().Shuffle(openings_.begin(), openings_.end());
    }
  }
  // If playing just one game, the player1 is white, otherwise randomize.
  if (kTotalGames != 1) {
    first_game_black_ = Random::Get().GetBool();
  }

  // Initializing networks.
  for (const auto& name : {"player1", "player2"}) {
    for (const auto& color : {"white", "black"}) {
      const auto& opts = options.GetSubdict(name).GetSubdict(color);
      const auto config = NetworkFactory::BackendConfiguration(opts);
      if (networks_.find(config) == networks_.end()) {
        networks_.emplace(config, NetworkFactory::LoadNetwork(opts));
      }
    }
  }

  // Initializing cache.
  cache_[0] = std::make_shared<NNCache>(
      options.GetSubdict("player1").Get<int>(kNNCacheSizeId));
  if (kShareTree) {
    cache_[1] = cache_[0];
  } else {
    cache_[1] = std::make_shared<NNCache>(
        options.GetSubdict("player2").Get<int>(kNNCacheSizeId));
  }

  // SearchLimits.
  static constexpr const char* kPlayerNames[2] = {"player1", "player2"};
  static constexpr const char* kPlayerColors[2] = {"white", "black"};
  for (int name_idx : {0, 1}) {
    for (int color_idx : {0, 1}) {
      auto& limits = search_limits_[name_idx][color_idx];
      const auto& dict = options.GetSubdict(kPlayerNames[name_idx])
                             .GetSubdict(kPlayerColors[color_idx]);
      limits.playouts = dict.Get<int>(kPlayoutsId);
      limits.visits = dict.Get<int>(kVisitsId);
      limits.movetime = dict.Get<int>(kTimeMsId);

      if (limits.playouts == -1 && limits.visits == -1 &&
          limits.movetime == -1) {
        throw Exception(
            "Please define --visits, --playouts or --movetime, otherwise it's "
            "not clear when to stop search.");
      }
    }
  }

  // Take syzygy tablebases from options.
  std::string tb_paths = options.Get<std::string>(kSyzygyTablebaseId);
  if (!tb_paths.empty()) {
    syzygy_tb_ = std::make_unique<SyzygyTablebase>();
    CERR << "Loading Syzygy tablebases from " << tb_paths;
    if (!syzygy_tb_->init(tb_paths)) {
      CERR << "Failed to load Syzygy tablebases!";
      syzygy_tb_ = nullptr;
    }
  }
}

void SelfPlayTournament::PlayOneGame(int game_number) {
  bool player1_black;  // Whether player1 will player as black in this game.
  Opening opening;
  {
    Mutex::Lock lock(mutex_);
    player1_black = ((game_number % 2) == 1) != first_game_black_;
    if (!openings_.empty()) {
      if (player_options_[0][0].Get<bool>(kOpeningsMirroredId)) {
        opening = openings_[(game_number / 2) % openings_.size()];
      } else if (player_options_[0][0].Get<std::string>(kOpeningsModeId) ==
                 "random") {
        opening = openings_[Random::Get().GetInt(0, openings_.size() - 1)];
      } else {
        opening = openings_[game_number % openings_.size()];
      }
    }
    if (discard_pile_.size() > 0 &&
        Random::Get().GetFloat(100.0f) < kDiscardedStartChance) {
      const size_t idx = Random::Get().GetInt(0, discard_pile_.size() - 1);
      if (idx != discard_pile_.size() - 1) {
        std::swap(discard_pile_[idx], discard_pile_.back());
      }
      opening = discard_pile_.back();
      discard_pile_.pop_back();
    }
  }
  const int color_idx[2] = {player1_black ? 1 : 0, player1_black ? 0 : 1};

  PlayerOptions options[2];

  std::vector<ThinkingInfo> last_thinking_info;
  for (int pl_idx : {0, 1}) {
    const int color = color_idx[pl_idx];
    const bool verbose_thinking =
        player_options_[pl_idx][color].Get<bool>(kVerboseThinkingId);
    const bool move_thinking =
        player_options_[pl_idx][color].Get<bool>(kMoveThinkingId);
    // Populate per-player options.
    PlayerOptions& opt = options[color_idx[pl_idx]];
    opt.network = networks_[NetworkFactory::BackendConfiguration(
                                player_options_[pl_idx][color])]
                      .get();
    opt.cache = cache_[pl_idx].get();
    opt.uci_options = &player_options_[pl_idx][color];
    opt.search_limits = search_limits_[pl_idx][color];

    // "bestmove" callback.
    opt.best_move_callback = [this, game_number, pl_idx, player1_black,
                              verbose_thinking, move_thinking,
                              &last_thinking_info](const BestMoveInfo& info) {
      if (!move_thinking) {
        last_thinking_info.clear();
        return;
      }
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
    opt.discarded_callback = [this](const Opening& moves) {
      // Only track discards if discard start chance is non-zero.
      if (kDiscardedStartChance == 0.0f) return;
      Mutex::Lock lock(mutex_);
      discard_pile_.push_back(moves);
      // 10k seems it should be enough to keep a good mix and avoid running out
      // of ram.
      if (discard_pile_.size() > 10000) {
        // Swap a random element to end and pop it to avoid growing.
        const size_t idx = Random::Get().GetInt(0, discard_pile_.size() - 1);
        if (idx != discard_pile_.size() - 1) {
          std::swap(discard_pile_[idx], discard_pile_.back());
        }
        discard_pile_.pop_back();
      }
    };
  }

  // Iterator to store the game in. Have to keep it so that later we can
  // delete it. Need to expose it in games_ member variable only because
  // of possible Abort() that should stop them all.
  std::list<std::unique_ptr<SelfPlayGame>>::iterator game_iter;
  {
    Mutex::Lock lock(mutex_);
    games_.emplace_front(std::make_unique<SelfPlayGame>(options[0], options[1],
                                                        kShareTree, opening));
    game_iter = games_.begin();
  }
  auto& game = **game_iter;

  // If kResignPlaythrough == 0, then this comparison is unconditionally true
  const bool enable_resign =
      Random::Get().GetFloat(100.0f) >= kResignPlaythrough;

  // PLAY GAME!
  auto player1_threads = player_options_[0][color_idx[0]].Get<int>(kThreadsId);
  auto player2_threads = player_options_[1][color_idx[1]].Get<int>(kThreadsId);
  game.Play(player1_threads, player2_threads, kTraining, syzygy_tb_.get(),
            enable_resign);

  // If game was aborted, it's still undecided.
  if (game.GetGameResult() != GameResult::UNDECIDED) {
    // Game callback.
    GameInfo game_info;
    game_info.game_result = game.GetGameResult();
    game_info.is_black = player1_black;
    game_info.game_id = game_number;
    game_info.initial_fen = opening.start_fen;
    game_info.moves = game.GetMoves();
    game_info.play_start_ply = game.GetStartPly();
    if (!enable_resign) {
      game_info.min_false_positive_threshold =
          game.GetWorstEvalForWinnerOrDraw();
    }
    if (kTraining &&
        game_info.play_start_ply < static_cast<int>(game_info.moves.size())) {
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
      tournament_info_.move_count_ += game.move_count_;
      tournament_info_.nodes_total_ += game.nodes_total_;
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
      bool mirrored = player_options_[0][0].Get<bool>(kOpeningsMirroredId);
      if ((kTotalGames >= 0 && games_count_ >= kTotalGames) ||
          (kTotalGames == -2 && !openings_.empty() &&
           games_count_ >=
               static_cast<int>(openings_.size()) * (mirrored ? 2 : 1)))
        break;
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

void SelfPlayTournament::Stop() {
  Mutex::Lock lock(mutex_);
  abort_ = true;
}

SelfPlayTournament::~SelfPlayTournament() {
  Abort();
  Wait();
}

}  // namespace lczero

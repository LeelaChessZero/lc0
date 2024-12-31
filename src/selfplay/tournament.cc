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

#include <fstream>

#include "chess/pgn.h"
#include "neural/factory.h"
#include "neural/shared_params.h"
#include "search/classic/search.h"
#include "search/classic/stoppers/factory.h"
#include "selfplay/game.h"
#include "selfplay/multigame.h"
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
const OptionId kPolicyModeSizeId{"policy-mode-size", "PolicyModeSize",
                                 "Number of games per thread in policy only "
                                 "mode. Set to 0 to not use policy only mode."};
const OptionId kValueModeSizeId{"value-mode-size", "ValueModeSize",
                                "Number of games per thread in value only "
                                "mode. Set to 0 to not use value only mode."};
const OptionId kTournamentResultsFileId{
    "tournament-results-file", "TournamentResultsFile",
    "Name of file to append the tournament results in fake pgn format."};
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

  SharedBackendParams::Populate(options);
  options->Add<IntOption>(kThreadsId, 1, 8) = 1;
  options->Add<IntOption>(classic::kNNCacheSizeId, 0, 999999999) = 2000000;
  classic::SearchParams::Populate(options);

  options->Add<BoolOption>(kShareTreesId) = true;
  options->Add<IntOption>(kTotalGamesId, -2, 999999) = -1;
  options->Add<IntOption>(kParallelGamesId, 1, 256) = 8;
  options->Add<IntOption>(kPlayoutsId, -1, 999999999) = -1;
  options->Add<IntOption>(kVisitsId, -1, 999999999) = -1;
  options->Add<IntOption>(kTimeMsId, -1, 999999999) = -1;
  options->Add<BoolOption>(kTrainingId) = false;
  options->Add<BoolOption>(kVerboseThinkingId) = false;
  options->Add<IntOption>(kPolicyModeSizeId, 0, 1024) = 0;
  options->Add<IntOption>(kValueModeSizeId, 0, 64) = 0;
  options->Add<StringOption>(kTournamentResultsFileId) = "";
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
  defaults->Set<int>(classic::SearchParams::kMiniBatchSizeId, 32);
  defaults->Set<float>(classic::SearchParams::kCpuctId, 1.2f);
  defaults->Set<float>(classic::SearchParams::kCpuctFactorId, 0.0f);
  defaults->Set<float>(SharedBackendParams::kPolicySoftmaxTemp, 1.0f);
  defaults->Set<int>(classic::SearchParams::kMaxCollisionVisitsId, 1);
  defaults->Set<int>(classic::SearchParams::kMaxCollisionEventsId, 1);
  defaults->Set<int>(classic::SearchParams::kCacheHistoryLengthId, 7);
  defaults->Set<bool>(classic::SearchParams::kOutOfOrderEvalId, false);
  defaults->Set<float>(classic::SearchParams::kTemperatureId, 1.0f);
  defaults->Set<float>(classic::SearchParams::kNoiseEpsilonId, 0.25f);
  defaults->Set<float>(classic::SearchParams::kFpuValueId, 0.0f);
  defaults->Set<std::string>(SharedBackendParams::kHistoryFill, "no");
  defaults->Set<std::string>(SharedBackendParams::kBackendId, "multiplexing");
  defaults->Set<bool>(classic::SearchParams::kStickyEndgamesId, false);
  defaults->Set<bool>(classic::SearchParams::kTwoFoldDrawsId, false);
  defaults->Set<int>(classic::SearchParams::kTaskWorkersPerSearchWorkerId, 0);
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
      kPolicyGamesSize(options.Get<int>(kPolicyModeSizeId)),
      kValueGamesSize(options.Get<int>(kValueModeSizeId)),
      kTournamentResultsFile(
          options.Get<std::string>(kTournamentResultsFileId)),
      kDiscardedStartChance(options.Get<float>(kDiscardedStartChanceId)) {
  multi_games_size_ = std::max(kPolicyGamesSize, kValueGamesSize);
  std::string book = options.Get<std::string>(kOpeningsFileId);
  if (!book.empty()) {
    PgnReader book_reader;
    book_reader.AddPgnFile(book);
    openings_ = book_reader.ReleaseGames();
    if (options.Get<std::string>(kOpeningsModeId) == "shuffled") {
      Random::Get().Shuffle(openings_.begin(), openings_.end());
    }
  }
  if (kPolicyGamesSize > 0 && kValueGamesSize > 0) {
    throw Exception("Can't do both policy and value games at the same time.");
  }
  if (multi_games_size_ > 0 && openings_.size() == 0) {
    throw Exception(
        "Policy/Value games are deterministic, needs opening book to be "
        "useful.");
  }
  if (multi_games_size_ > 0 &&
      (kTotalGames == -1 ||
       (kTotalGames > 0 &&
        static_cast<size_t>(kTotalGames) > openings_.size() * 2))) {
    throw Exception(
        "Policy/Value games are deterministic, you do not want to go through "
        "the "
        "opening book more than once.");
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
      options.GetSubdict("player1").Get<int>(classic::kNNCacheSizeId));
  if (kShareTree) {
    cache_[1] = cache_[0];
  } else {
    cache_[1] = std::make_shared<NNCache>(
        options.GetSubdict("player2").Get<int>(classic::kNNCacheSizeId));
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

      if (multi_games_size_ == 0 && limits.playouts == -1 &&
          limits.visits == -1 && limits.movetime == -1) {
        throw Exception(
            "Please define --visits, --playouts or --movetime, otherwise it's "
            "not clear when to stop search.");
      } else if (multi_games_size_ > 0 &&
                 (limits.playouts != -1 || limits.visits != -1 ||
                  limits.movetime != -1)) {
        throw Exception(
            "Policy mode does not need --visits, --playouts or --movetime as "
            "it has fixed 1 node behaviour.");
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
  SyzygyTablebase* syzygy_tb;
  {
    Mutex::Lock lock(mutex_);
    games_.emplace_front(std::make_unique<SelfPlayGame>(options[0], options[1],
                                                        kShareTree, opening));
    game_iter = games_.begin();
    syzygy_tb = syzygy_tb_.get();
  }
  auto& game = **game_iter;

  // If kResignPlaythrough == 0, then this comparison is unconditionally true
  const bool enable_resign =
      Random::Get().GetFloat(100.0f) >= kResignPlaythrough;

  // PLAY GAME!
  auto player1_threads = player_options_[0][color_idx[0]].Get<int>(kThreadsId);
  auto player2_threads = player_options_[1][color_idx[1]].Get<int>(kThreadsId);
  game.Play(player1_threads, player2_threads, kTraining, syzygy_tb,
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
      int result = game.GetGameResult() == GameResult::DRAW        ? 1
                   : game.GetGameResult() == GameResult::WHITE_WON ? 0
                                                                   : 2;
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

void SelfPlayTournament::PlayMultiGames(int game_id, size_t game_count) {
  bool use_value = kValueGamesSize > 0;
  std::vector<Opening> openings;
  openings.reserve(game_count / 2);
  size_t opening_basis = game_id / 2;
  {
    Mutex::Lock lock(mutex_);
    for (size_t i = 0; i < game_count / 2; i++) {
      openings.push_back(openings_[(opening_basis + i) % openings_.size()]);
    }
  }

  PlayerOptions options[2];
  options[0].network =
      networks_[NetworkFactory::BackendConfiguration(player_options_[0][0])]
          .get();
  options[1].network =
      networks_[NetworkFactory::BackendConfiguration(player_options_[1][1])]
          .get();

  std::list<std::unique_ptr<MultiSelfPlayGames>>::iterator game1_iter;
  auto aborted = false;
  {
    Mutex::Lock lock(mutex_);
    multigames_.emplace_front(std::make_unique<MultiSelfPlayGames>(
        options[0], options[1], openings, syzygy_tb_.get(), use_value));
    game1_iter = multigames_.begin();
    aborted = abort_;
  }
  auto& game1 = **game1_iter;

  // PLAY GAMEs!
  if (!aborted) game1.Play();

  options[0].network =
      networks_[NetworkFactory::BackendConfiguration(player_options_[0][1])]
          .get();
  options[1].network =
      networks_[NetworkFactory::BackendConfiguration(player_options_[1][0])]
          .get();

  std::list<std::unique_ptr<MultiSelfPlayGames>>::iterator game2_iter;
  {
    Mutex::Lock lock(mutex_);
    multigames_.emplace_front(std::make_unique<MultiSelfPlayGames>(
        options[1], options[0], openings, syzygy_tb_.get(), use_value));
    game2_iter = multigames_.begin();
    aborted = abort_;
  }
  auto& game2 = **game2_iter;
  // PLAY reverse GAMEs!
  if (!aborted) game2.Play();

  for (size_t i = 0; i < openings.size(); i++) {
    auto game1_res = game1.GetGameResult(i);
    if (game1_res != GameResult::UNDECIDED) {
      // Game callback.
      GameInfo game_info;
      game_info.game_result = game1_res;
      game_info.is_black = false;
      game_info.game_id = game_id + 2 * i;
      game_info.moves = game1.GetMoves(i);
      game_info.initial_fen = openings[i].start_fen;
      game_info.play_start_ply = openings[i].moves.size();
      game_callback_(game_info);

      // Update tournament stats.
      {
        Mutex::Lock lock(mutex_);
        int result = game1_res == GameResult::DRAW        ? 1
                     : game1_res == GameResult::WHITE_WON ? 0
                                                          : 2;
        ++tournament_info_.results[result][0];
        tournament_callback_(tournament_info_);
      }
    }
    auto game2_res = game2.GetGameResult(i);
    if (game2_res != GameResult::UNDECIDED) {
      // Game callback.
      GameInfo game_info;
      game_info.game_result = game2_res;
      game_info.is_black = true;
      game_info.game_id = game_id + 2 * i + 1;
      game_info.moves = game2.GetMoves(i);
      game_info.initial_fen = openings[i].start_fen;
      game_info.play_start_ply = openings[i].moves.size();
      game_callback_(game_info);

      // Update tournament stats.
      {
        Mutex::Lock lock(mutex_);
        int result = game2_res == GameResult::DRAW        ? 1
                     : game2_res == GameResult::WHITE_WON ? 2
                                                          : 0;
        ++tournament_info_.results[result][1];
        tournament_callback_(tournament_info_);
      }
    }
  }
  {
    Mutex::Lock lock(mutex_);
    multigames_.erase(game2_iter);
    multigames_.erase(game1_iter);
  }
}

void SelfPlayTournament::Worker() {
  // Play games while game limit is not reached (or while not aborted).
  while (true) {
    int game_id;
    int count = 0;
    {
      Mutex::Lock lock(mutex_);
      if (abort_) break;
      if (multi_games_size_) {
        if (!player_options_[0][0].Get<bool>(kOpeningsMirroredId)) {
          throw Exception(
              "Policy/Value multi games mode only supports mirrored openings.");
        }
        if (kTotalGames > 0 && kTotalGames % 2 == 1) {
          throw Exception(
              "Policy/Value multi games can't support mirrored with an odd "
              "number "
              "of games.");
        }
        int to_take = 2 * multi_games_size_;
        int max_take = 2 * multi_games_size_;
        if (kTotalGames != -1) {
          int cap = kTotalGames == -2 ? openings_.size() * 2 : kTotalGames;
          to_take = std::min(max_take, cap - games_count_);
        }
        if (to_take <= 0) {
          break;
        }
        game_id = games_count_;
        count = to_take;
        games_count_ += to_take;
      } else {
        bool mirrored = player_options_[0][0].Get<bool>(kOpeningsMirroredId);
        if ((kTotalGames >= 0 && games_count_ >= kTotalGames) ||
            (kTotalGames == -2 && !openings_.empty() &&
             games_count_ >=
                 static_cast<int>(openings_.size()) * (mirrored ? 2 : 1)))
          break;
        game_id = games_count_++;
      }
    }
    if (multi_games_size_) {
      PlayMultiGames(game_id, count);
    } else {
      PlayOneGame(game_id);
    }
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
      SaveResults();
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
      SaveResults();
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
  for (auto& game : multigames_)
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

void SelfPlayTournament::SaveResults() {
  if (kTournamentResultsFile.empty()) return;
  std::ofstream output(kTournamentResultsFile, std::ios_base::app);
  auto p1name =
      player_options_[0][0].Get<std::string>(SharedBackendParams::kWeightsId);
  auto p2name =
      player_options_[1][0].Get<std::string>(SharedBackendParams::kWeightsId);

  output << std::endl;
  output << "[White \"" << p1name << "\"]" << std::endl;
  output << "[Black \"" << p2name << "\"]" << std::endl;
  output << "[Results \"" << tournament_info_.results[0][0] << " "
         << tournament_info_.results[2][0] << " "
         << tournament_info_.results[1][0] << "\"]" << std::endl;
  output << std::endl;
  output << "[White \"" << p2name << "\"]" << std::endl;
  output << "[Black \"" << p1name << "\"]" << std::endl;
  output << "[Results \"" << tournament_info_.results[2][1] << " "
         << tournament_info_.results[0][1] << " "
         << tournament_info_.results[1][1] << "\"]" << std::endl;
}

}  // namespace lczero

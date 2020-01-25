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
#include "mcts/stoppers/factory.h"
#include "neural/factory.h"
#include "selfplay/game.h"
#include "utils/optionsparser.h"
#include "utils/random.h"

#include <fstream>

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
const OptionId kQuietThinkingId{"quiet-thinking", "QuietThinking",
                                "Hide all the per-move thinking."};
const OptionId kResignPlaythroughId{
    "resign-playthrough", "ResignPlaythrough",
    "The percentage of games which ignore resign."};
const OptionId kDiscardedStartChanceId{
    "discarded-start-chance", "DiscardedStartChance",
    "The percentage chance each game will attempt to start from a position "
    "discarded due to not getting enough visits."};
const OptionId kBookFileId{
    "pgn-book", "PGNBook",
    "A path name to a pgn file containing openings to use."};
const OptionId kBookMirroredId{
    "book-mirrored", "BookMirrored",
    "If true, each opening will be played in pairs. Not really compatible with book mode random."};
const OptionId kBookModeId{
    "book-mode", "BookMode",
    "A choice of sequential, shuffled, or random."};

Move MoveFor(int r1, int c1, int r2, int c2, int p2) {
  Move m;
  if (p2 != -1) {
    if (p2 == 2)
      m = Move(BoardSquare(r1, c1), BoardSquare(r2, c2),
               Move::Promotion::Queen);
    else if (p2 == 3)
      m = Move(BoardSquare(r1, c1), BoardSquare(r2, c2),
               Move::Promotion::Bishop);
    else if (p2 == 4)
      m = Move(BoardSquare(r1, c1), BoardSquare(r2, c2),
               Move::Promotion::Knight);
    else if (p2 == 5)
      m = Move(BoardSquare(r1, c1), BoardSquare(r2, c2), Move::Promotion::Rook);
  } else {
    m = Move(BoardSquare(r1, c1), BoardSquare(r2, c2));
  }
  return m;
}

Move SanToMove(const std::string& san, const ChessBoard& board) {
  int p = 0;
  int idx = 0;
  if (san[0] == 'K') {
    p = 1;
  } else if (san[0] == 'Q') {
    p = 2;
  } else if (san[0] == 'B') {
    p = 3;
  } else if (san[0] == 'N') {
    p = 4;
  } else if (san[0] == 'R') {
    p = 5;
  } else if (san[0] == 'O' && san.size() > 2 && san[1] == '-' &&
             san[2] == 'O') {
    Move m;
    if (san.size() > 4 && san[3] == '-' && san[4] == 'O') {
      m = Move(BoardSquare(0, 4), BoardSquare(0, 2));
    } else {
      m = Move(BoardSquare(0, 4), BoardSquare(0, 6));
    }
    // std::cerr << m.as_string() << std::endl;
    return m;
  }
  if (p != 0) idx++;
  // Formats e4 1e5 de5 d1e5 - with optional x's - followed by =Q for
  // promotions, and even more characters after that also optional.
  int r1 = -1;
  int c1 = -1;
  int r2 = -1;
  int c2 = -1;
  int p2 = -1;
  bool pPending = false;
  for (; idx < san.size(); idx++) {
    if (san[idx] == 'x') continue;
    if (san[idx] == '=') {
      pPending = true;
      continue;
    }
    if (san[idx] >= '1' && san[idx] <= '9') {
      r1 = r2;
      r2 = san[idx] - '1';
      continue;
    }
    if (san[idx] >= 'a' && san[idx] <= 'h') {
      c1 = c2;
      c2 = san[idx] - 'a';
      continue;
    }
    if (pPending) {
      if (san[idx] == 'Q') {
        p2 = 2;
      } else if (san[idx] == 'B') {
        p2 = 3;
      } else if (san[idx] == 'N') {
        p2 = 4;
      } else if (san[idx] == 'R') {
        p2 = 5;
      }
      pPending = false;
      break;
    }
    break;
  }
  if (r1 == -1 || c1 == -1) {
    // Need to find the from cell based on piece.
    int sr1 = r1;
    int sr2 = r2;
    if (board.flipped()) {
      if (sr1 != -1) sr1 = 7 - sr1;
      sr2 = 7 - sr2;
    }
    BitBoard searchBits;
    if (p == 0) {
      searchBits = (board.pawns() & board.ours());
    } else if (p == 1) {
      searchBits = board.our_king();
    } else if (p == 2) {
      searchBits = (board.queens() & board.ours());
    } else if (p == 3) {
      searchBits = (board.bishops() & board.ours());
    } else if (p == 4) {
      searchBits = board.our_knights();
    } else if (p == 5) {
      searchBits = (board.rooks() & board.ours());
    }
    auto plm = board.GenerateLegalMoves();
    int pr1 = -1;
    int pc1 = -1;
    for (BoardSquare sq : searchBits) {
      if (sr1 != -1 && sq.row() != sr1) continue;
      if (c1 != -1 && sq.col() != c1) continue;
      if (std::find(plm.begin(), plm.end(),
                    MoveFor(sq.row(), sq.col(), sr2, c2, p2)) == plm.end())
        continue;
      if (pc1 != -1) {
        std::cerr << "Ambiguous!!" << std::endl;
      }
      pr1 = sq.row();
      pc1 = sq.col();
    }
    if (pc1 == -1) {
      std::cerr << "No Match!!" << std::endl;
    }
    r1 = pr1;
    c1 = pc1;
    if (board.flipped()) {
      r1 = 7 - r1;
    }
  }
  Move m = MoveFor(r1, c1, r2, c2, p2);
  if (board.flipped()) m.Mirror();
  // std::cerr << m.as_string() << std::endl;
  return m;
}

std::vector<std::vector<Move>> ReadBook(std::string path) {
  std::vector<std::vector<Move>> result;
  std::vector<Move> cur;
  ChessBoard board;
  board.SetFromFen(ChessBoard::kStartposFen);
  std::ifstream file(path);
  std::string line;
  bool in_comment = false;
  while (std::getline(file, line)) {
    if (line.size() == 0 || line[0] == '[') {
      if (cur.size() > 0) {
        result.push_back(cur);
        cur.clear();
        board.SetFromFen(ChessBoard::kStartposFen);
        // std::cerr << "" << std::endl;
      }
      continue;
    }
    // Handle braced comments.
    int cur_offset = 0;
    while (in_comment && line.find('}', cur_offset) != std::string::npos ||
           !in_comment && line.find('{', cur_offset) != std::string::npos) {
      if (in_comment && line.find('}', cur_offset) != std::string::npos) {
        line = line.substr(0, cur_offset) +
               line.substr(line.find('}', cur_offset) + 1);
        in_comment = false;
      } else {
        cur_offset = line.find('{', cur_offset);
        in_comment = true;
      }
    }
    if (in_comment) {
      line = line.substr(0, cur_offset);
    }
    // Trim trailing comment.
    if (line.find(';') != std::string::npos) {
      line = line.substr(0, line.find(';'));
    }
    if (line.size() == 0) continue;
    std::istringstream iss(line);
    // std::cerr << line << std::endl;
    std::string word;
    while (!iss.eof()) {
      word.clear();
      iss >> word;
      if (word.size() < 2) continue;
      // Trim move numbers from front.
      int idx = word.find('.');
      if (idx != std::string::npos) {
        bool all_nums = true;
        for (int i = 0; i < idx; i++) {
          if (word[i] < '0' || word[i] > '9') {
            all_nums = false;
            break;
          }
        }
        if (all_nums) {
          word = word.substr(idx + 1);
        }
      }
      // Pure move numbers can be skipped.
      if (word.size() < 2) continue;
      // Ignore score line.
      if (word == "1/2-1/2" || word == "1-0" || word == "0-1" || word == "*")
        continue;
      // std::cerr << word << std::endl;
      cur.push_back(SanToMove(word, board));
      board.ApplyMove(cur.back());
      // Board ApplyMove wants mirrored for black, but outside code wants
      // normal, so mirror it back again.
      // Check equal to 0 since we've already added the position.
      if ((cur.size() % 2) == 0) {
        cur.back().Mirror();
      }
      board.Mirror();
    }
  }
  if (cur.size() > 0) {
    result.push_back(cur);
    cur.clear();
  }
  return result;
}

}  // namespace

void SelfPlayTournament::PopulateOptions(OptionsParser* options) {
  options->AddContext("player1");
  options->AddContext("player2");

  NetworkFactory::PopulateOptions(options);
  options->Add<IntOption>(kThreadsId, 1, 8) = 1;
  options->Add<IntOption>(kNNCacheSizeId, 0, 999999999) = 200000;
  SearchParams::Populate(options);

  options->Add<BoolOption>(kShareTreesId) = true;
  options->Add<IntOption>(kTotalGamesId, -1, 999999) = -1;
  options->Add<IntOption>(kParallelGamesId, 1, 256) = 8;
  options->Add<IntOption>(kPlayoutsId, -1, 999999999) = -1;
  options->Add<IntOption>(kVisitsId, -1, 999999999) = -1;
  options->Add<IntOption>(kTimeMsId, -1, 999999999) = -1;
  options->Add<BoolOption>(kTrainingId) = false;
  options->Add<BoolOption>(kVerboseThinkingId) = false;
  options->Add<BoolOption>(kQuietThinkingId) = false;
  options->Add<FloatOption>(kResignPlaythroughId, 0.0f, 100.0f) = 0.0f;
  options->Add<FloatOption>(kDiscardedStartChanceId, 0.0f, 100.0f) = 0.0f;
  options->Add<StringOption>(kBookFileId) = "";
  options->Add<BoolOption>(kBookMirroredId) = false;
  std::vector<std::string> book_modes = {"sequential", "shuffled", "random"};
  options->Add<ChoiceOption>(kBookModeId, book_modes) = "sequential";

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
  defaults->Set<float>(SearchParams::kTemperatureId.GetId(), 1.0f);
  defaults->Set<float>(SearchParams::kNoiseEpsilonId.GetId(), 0.25f);
  defaults->Set<float>(SearchParams::kFpuValueId.GetId(), 0.0f);
  defaults->Set<std::string>(SearchParams::kHistoryFillId.GetId(), "no");
  defaults->Set<std::string>(NetworkFactory::kBackendId.GetId(),
                             "multiplexing");
  defaults->Set<bool>(SearchParams::kStickyEndgamesId.GetId(), false);
  defaults->Set<bool>(SearchParams::kLogitQId.GetId(), false);
}

SelfPlayTournament::SelfPlayTournament(
    const OptionsDict& options,
    CallbackUciResponder::BestMoveCallback best_move_info,
    CallbackUciResponder::ThinkingCallback thinking_info,
    GameInfo::Callback game_info, TournamentInfo::Callback tournament_info)
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
      kResignPlaythrough(options.Get<float>(kResignPlaythroughId.GetId())),
      kDiscardedStartChance(
          options.Get<float>(kDiscardedStartChanceId.GetId())) {
  std::string book = options.Get<std::string>(kBookFileId.GetId());
  if (!book.empty()) {
    openings_ = ReadBook(book);
    if (options.Get<std::string>(kBookModeId.GetId()) == "shuffled") {
      Random::Get().Shuffle(openings_.begin(), openings_.end());
    }
  }
  // If playing just one game, the player1 is white, otherwise randomize.
  // If mirrored opening book, also not randomized since there is no point.
  if (kTotalGames != 1 && !options.Get<bool>(kBookMirroredId.GetId())) {
    first_game_black_ = Random::Get().GetBool();
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
      options.GetSubdict("player1").Get<int>(kNNCacheSizeId.GetId()));
  if (kShareTree) {
    cache_[1] = cache_[0];
  } else {
    cache_[1] = std::make_shared<NNCache>(
        options.GetSubdict("player2").Get<int>(kNNCacheSizeId.GetId()));
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
  MoveList opening;
  {
    Mutex::Lock lock(mutex_);
    player1_black = ((game_number % 2) == 1) ^ first_game_black_;
    bool mirrored = player_options_[0].Get<bool>(kBookMirroredId.GetId());
    if (mirrored) {
      if (static_cast<int>(openings_.size()) > game_number / 2) {
        opening = openings_[game_number / 2];
      }
    } else if (!openings_.empty() && player_options_[0].Get<std::string>(kBookModeId.GetId()) ==
               "random") {
      opening = openings_[Random::Get().GetInt(0, openings_.size() - 1)];
    } else {
      if (static_cast<int>(openings_.size()) > game_number) {
        opening = openings_[game_number];
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
    const bool verbose_thinking =
        player_options_[pl_idx].Get<bool>(kVerboseThinkingId.GetId());
    const bool quiet_thinking =
        player_options_[pl_idx].Get<bool>(kQuietThinkingId.GetId());
    // Populate per-player options.
    PlayerOptions& opt = options[color_idx[pl_idx]];
    opt.network = networks_[pl_idx].get();
    opt.cache = cache_[pl_idx].get();
    opt.uci_options = &player_options_[pl_idx];
    opt.search_limits = search_limits_[pl_idx];

    // "bestmove" callback.
    opt.best_move_callback = [this, game_number, pl_idx, player1_black,
                              verbose_thinking, quiet_thinking,
                              &last_thinking_info](const BestMoveInfo& info) {
      if (quiet_thinking) {
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
    opt.discarded_callback = [this](const MoveList& moves) {
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
    game_info.play_start_ply = opening.size();
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
      bool mirrored = player_options_[0].Get<bool>(kBookMirroredId.GetId());
      if (kTotalGames != -1 && games_count_ >= kTotalGames ||
          kTotalGames == -1 && !openings_.empty() &&
              games_count_ >= static_cast<int>(openings_.size()) * (mirrored ? 2 : 1))
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

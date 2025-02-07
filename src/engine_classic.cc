/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

#include "engine_classic.h"

#include <algorithm>
#include <cmath>
#include <functional>

#include "neural/shared_params.h"
#include "search/classic/search.h"
#include "search/classic/stoppers/factory.h"
#include "utils/commandline.h"
#include "utils/configfile.h"
#include "utils/logging.h"

namespace lczero {
namespace {
const OptionId kThreadsOptionId{
    "threads", "Threads",
    "Number of (CPU) worker threads to use, 0 for the backend default.", 't'};
const OptionId kSyzygyTablebaseId{
    "syzygy-paths", "SyzygyPath",
    "List of Syzygy tablebase directories, list entries separated by system "
    "separator (\";\" for Windows, \":\" for Linux).",
    's'};
const OptionId kPonderId{"", "Ponder",
                         "This option is ignored. Here to please chess GUIs."};
const OptionId kUciChess960{
    "chess960", "UCI_Chess960",
    "Castling moves are encoded as \"king takes rook\"."};
const OptionId kShowWDL{"show-wdl", "UCI_ShowWDL",
                        "Show win, draw and lose probability."};
const OptionId kShowMovesleft{"show-movesleft", "UCI_ShowMovesLeft",
                              "Show estimated moves left."};
const OptionId kStrictUciTiming{"strict-uci-timing", "StrictTiming",
                                "The UCI host compensates for lag, waits for "
                                "the 'readyok' reply before sending 'go' and "
                                "only then starts timing."};
const OptionId kValueOnly{
    "value-only", "ValueOnly",
    "In value only mode all search parameters are ignored and the position is "
    "evaluated by getting the valuation of every child position and choosing "
    "the worst for the opponent."};
const OptionId kClearTree{"", "ClearTree",
                          "Clear the tree before the next search."};

MoveList StringsToMovelist(const std::vector<std::string>& moves,
                           const ChessBoard& board) {
  MoveList result;
  if (moves.size()) {
    result.reserve(moves.size());
    const auto legal_moves = board.GenerateLegalMoves();
    const auto end = legal_moves.end();
    for (const auto& move : moves) {
      const auto m = board.GetModernMove({move, board.flipped()});
      if (std::find(legal_moves.begin(), end, m) != end) result.emplace_back(m);
    }
    if (result.empty()) throw Exception("No legal searchmoves.");
  }
  return result;
}

}  // namespace

EngineClassic::EngineClassic(UciResponder& uci_responder,
                             const OptionsDict& options)
    : options_(options),
      uci_responder_(&uci_responder),
      current_position_{ChessBoard::kStartposFen, {}} {}

void EngineClassic::PopulateOptions(OptionsParser* options) {
  using namespace std::placeholders;
  const bool is_simple =
      CommandLine::BinaryName().find("simple") != std::string::npos;
  options->Add<IntOption>(kThreadsOptionId, 0, 128) = 0;
  classic::SearchParams::Populate(options);

  ConfigFile::PopulateOptions(options);
  if (is_simple) {
    options->HideAllOptions();
    options->UnhideOption(kThreadsOptionId);
    options->UnhideOption(SharedBackendParams::kWeightsId);
    options->UnhideOption(classic::SearchParams::kContemptId);
    options->UnhideOption(classic::SearchParams::kMultiPvId);
  }
  options->Add<StringOption>(kSyzygyTablebaseId);
  // Add "Ponder" option to signal to GUIs that we support pondering.
  // This option is currently not used by lc0 in any way.
  options->Add<BoolOption>(kPonderId) = true;
  options->Add<BoolOption>(kUciChess960) = false;
  options->Add<BoolOption>(kShowWDL) = false;
  options->Add<BoolOption>(kShowMovesleft) = false;

  PopulateTimeManagementOptions(
      is_simple ? classic::RunType::kSimpleUci : classic::RunType::kUci,
      options);

  options->Add<BoolOption>(kStrictUciTiming) = false;
  options->HideOption(kStrictUciTiming);

  options->Add<BoolOption>(kValueOnly) = false;
  options->Add<ButtonOption>(kClearTree);
  options->HideOption(kClearTree);
}

void EngineClassic::ResetMoveTimer() {
  move_start_time_ = std::chrono::steady_clock::now();
}

// Updates values from Uci options.
void EngineClassic::UpdateFromUciOptions() {
  SharedLock lock(busy_mutex_);

  // Syzygy tablebases.
  std::string tb_paths = options_.Get<std::string>(kSyzygyTablebaseId);
  if (!tb_paths.empty() && tb_paths != tb_paths_) {
    syzygy_tb_ = std::make_unique<SyzygyTablebase>();
    CERR << "Loading Syzygy tablebases from " << tb_paths;
    if (!syzygy_tb_->init(tb_paths)) {
      CERR << "Failed to load Syzygy tablebases!";
      syzygy_tb_ = nullptr;
    }
    tb_paths_ = tb_paths;
  } else if (tb_paths.empty()) {
    syzygy_tb_ = nullptr;
    tb_paths_.clear();
  }

  // Network.
  const auto network_configuration =
      NetworkFactory::BackendConfiguration(options_);
  if (network_configuration_ != network_configuration) {
    backend_ =
        CreateMemCache(BackendManager::Get()->CreateFromParams(options_),
                       options_.Get<int>(SharedBackendParams::kNNCacheSizeId));
    network_configuration_ = network_configuration;
  } else {
    // If network is not changed, cache size still may have changed.
    backend_->SetCacheCapacity(
        options_.Get<int>(SharedBackendParams::kNNCacheSizeId));
  }

  // Check whether we can update the move timer in "Go".
  strict_uci_timing_ = options_.Get<bool>(kStrictUciTiming);
}

void EngineClassic::EnsureReady() {
  std::unique_lock<RpSharedMutex> lock(busy_mutex_);
  // If a UCI host is waiting for our ready response, we can consider the move
  // not started until we're done ensuring ready.
  ResetMoveTimer();
}

void EngineClassic::NewGame() {
  // In case anything relies upon defaulting to default position and just calls
  // newgame and goes straight into go.
  ResetMoveTimer();
  SharedLock lock(busy_mutex_);
  search_.reset();
  tree_.reset();
  CreateFreshTimeManager();
  current_position_ = {ChessBoard::kStartposFen, {}};
  UpdateFromUciOptions();
  backend_->ClearCache();
}

void EngineClassic::SetPosition(const std::string& fen,
                                const std::vector<std::string>& moves_str) {
  // Some UCI hosts just call position then immediately call go, while starting
  // the clock on calling 'position'.
  ResetMoveTimer();
  SharedLock lock(busy_mutex_);
  current_position_ = CurrentPosition{fen, moves_str};
  search_.reset();
}

Position EngineClassic::ApplyPositionMoves() {
  ChessBoard board;
  int no_capture_ply;
  int game_move;
  board.SetFromFen(current_position_.fen, &no_capture_ply, &game_move);
  int game_ply = 2 * game_move - (board.flipped() ? 1 : 2);
  Position pos(board, no_capture_ply, game_ply);
  for (std::string move_str : current_position_.moves) {
    Move move(move_str);
    if (pos.IsBlackToMove()) move.Mirror();
    pos = Position(pos, move);
  }
  return pos;
}

void EngineClassic::SetupPosition(const std::string& fen,
                                  const std::vector<std::string>& moves_str) {
  SharedLock lock(busy_mutex_);
  search_.reset();

  UpdateFromUciOptions();

  if (!tree_) tree_ = std::make_unique<classic::NodeTree>();

  std::vector<Move> moves;
  for (const auto& move : moves_str) moves.emplace_back(move);
  const bool is_same_game = tree_->ResetToPosition(fen, moves);
  if (!is_same_game) CreateFreshTimeManager();
}

void EngineClassic::CreateFreshTimeManager() {
  time_manager_ = classic::MakeTimeManager(options_);
}

namespace {

class PonderResponseTransformer : public TransformingUciResponder {
 public:
  PonderResponseTransformer(std::unique_ptr<UciResponder> parent,
                            std::string ponder_move)
      : TransformingUciResponder(std::move(parent)),
        ponder_move_(std::move(ponder_move)) {}

  void TransformThinkingInfo(std::vector<ThinkingInfo>* infos) override {
    // Output all stats from main variation (not necessary the ponder move)
    // but PV only from ponder move.
    ThinkingInfo ponder_info;
    for (const auto& info : *infos) {
      if (info.multipv <= 1) {
        ponder_info = info;
        if (ponder_info.mate) ponder_info.mate = -*ponder_info.mate;
        if (ponder_info.score) ponder_info.score = -*ponder_info.score;
        if (ponder_info.depth > 1) ponder_info.depth--;
        if (ponder_info.seldepth > 1) ponder_info.seldepth--;
        if (ponder_info.wdl) std::swap(ponder_info.wdl->w, ponder_info.wdl->l);
        ponder_info.pv.clear();
      }
      if (!info.pv.empty() && info.pv[0].as_string() == ponder_move_) {
        ponder_info.pv.assign(info.pv.begin() + 1, info.pv.end());
      }
    }
    infos->clear();
    infos->push_back(ponder_info);
  }

 private:
  std::string ponder_move_;
};

void ValueOnlyGo(classic::NodeTree* tree, Backend* backend,
                 std::unique_ptr<UciResponder> responder) {
  const auto& board = tree->GetPositionHistory().Last().GetBoard();
  auto legal_moves = board.GenerateLegalMoves();
  tree->GetCurrentHead()->CreateEdges(legal_moves);
  PositionHistory history = tree->GetPositionHistory();
  std::vector<float> comp_q;
  comp_q.reserve(legal_moves.size());
  auto comp = backend->CreateComputation();
  for (auto edge : tree->GetCurrentHead()->Edges()) {
    history.Append(edge.GetMove());
    if (history.ComputeGameResult() == GameResult::UNDECIDED) {
      comp_q.emplace_back();
      comp->AddInput(
          EvalPosition{
              .pos = history.GetPositions(),
              .legal_moves = {},
          },
          EvalResultPtr{.q = &comp_q.back()});
    }
    history.Pop();
  }

  Move best;
  float max_q = std::numeric_limits<float>::lowest();
  for (size_t comp_idx = 0; auto edge : tree->GetCurrentHead()->Edges()) {
    history.Append(edge.GetMove());
    auto result = history.ComputeGameResult();
    float q = -1;
    if (result == GameResult::UNDECIDED) {
      // NN eval is for side to move perspective - so if its good, its bad for
      // us.
      q = -comp_q[comp_idx++];
    } else if (result == GameResult::DRAW) {
      q = 0;
    } else {
      // A legal move to a non-drawn terminal without tablebases must be a
      // win.
      q = 1;
    }
    if (q >= max_q) {
      max_q = q;
      best = edge.GetMove(tree->GetPositionHistory().IsBlackToMove());
    }
    history.Pop();
  }
  std::vector<ThinkingInfo> infos;
  ThinkingInfo thinking;
  thinking.depth = 1;
  infos.push_back(thinking);
  responder->OutputThinkingInfo(&infos);
  BestMoveInfo info(best);
  responder->OutputBestMove(&info);
}

}  // namespace

void EngineClassic::Go(const GoParams& params) {
  // TODO: should consecutive calls to go be considered to be a continuation and
  // hence have the same start time like this behaves, or should we check start
  // time hasn't changed since last call to go and capture the new start time
  // now?
  if (strict_uci_timing_ || !move_start_time_) ResetMoveTimer();
  go_params_ = params;

  std::unique_ptr<UciResponder> responder =
      std::make_unique<NonOwningUciRespondForwarder>(uci_responder_);

  // Setting up current position, now that it's known whether it's ponder or
  // not.
  if (params.ponder && !current_position_.moves.empty()) {
    std::vector<std::string> moves(current_position_.moves);
    std::string ponder_move = moves.back();
    moves.pop_back();
    SetupPosition(current_position_.fen, moves);
    responder = std::make_unique<PonderResponseTransformer>(
        std::move(responder), ponder_move);
  } else {
    SetupPosition(current_position_.fen, current_position_.moves);
  }

  if (!options_.Get<bool>(kUciChess960)) {
    // Remap FRC castling to legacy castling.
    responder = std::make_unique<Chess960Transformer>(
        std::move(responder), tree_->HeadPosition().GetBoard());
  }

  if (!options_.Get<bool>(kShowWDL)) {
    // Strip WDL information from the response.
    responder = std::make_unique<WDLResponseFilter>(std::move(responder));
  }

  if (!options_.Get<bool>(kShowMovesleft)) {
    // Strip movesleft information from the response.
    responder = std::make_unique<MovesLeftResponseFilter>(std::move(responder));
  }
  if (options_.Get<bool>(kValueOnly)) {
    ValueOnlyGo(tree_.get(), backend_.get(), std::move(responder));
    return;
  }

  if (options_.Get<Button>(kClearTree).TestAndReset()) {
    tree_->TrimTreeAtHead();
  }

  auto stopper = time_manager_->GetStopper(params, *tree_.get());
  search_ = std::make_unique<classic::Search>(
      *tree_, backend_.get(), std::move(responder),
      StringsToMovelist(params.searchmoves, tree_->HeadPosition().GetBoard()),
      *move_start_time_, std::move(stopper), params.infinite, params.ponder,
      options_, syzygy_tb_.get());

  LOGFILE << "Timer started at "
          << FormatTime(SteadyClockToSystemClock(*move_start_time_));
  search_->StartThreads(options_.Get<int>(kThreadsOptionId));
}

void EngineClassic::PonderHit() {
  ResetMoveTimer();
  go_params_.ponder = false;
  Go(go_params_);
}

void EngineClassic::Stop() {
  if (search_) search_->Stop();
}

}  // namespace lczero

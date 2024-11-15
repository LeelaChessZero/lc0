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

#include "engine.h"

#include <algorithm>
#include <cmath>
#include <functional>

#include "mcts/search.h"
#include "mcts/stoppers/factory.h"
#include "utils/commandline.h"
#include "utils/configfile.h"
#include "utils/logging.h"

namespace lczero {
namespace {
const OptionId kThreadsOptionId{
    "threads", "Threads",
    "Number of (CPU) worker threads to use, 0 for the backend default.", 't'};
const OptionId kLogFileId{"logfile", "LogFile",
                          "Write log to that file. Special value <stderr> to "
                          "output the log to the console.",
                          'l'};
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
const OptionId kPreload{"preload", "",
                        "Initialize backend and load net on engine startup."};
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

EngineController::EngineController(std::unique_ptr<UciResponder> uci_responder,
                                   const OptionsDict& options)
    : options_(options),
      uci_responder_(std::move(uci_responder)),
      current_position_{ChessBoard::kStartposFen, {}} {}

void EngineController::PopulateOptions(OptionsParser* options) {
  using namespace std::placeholders;
  const bool is_simple =
      CommandLine::BinaryName().find("simple") != std::string::npos;
  options->AddContext("white");
  options->AddContext("black");
  NetworkFactory::PopulateOptions(options);
  options->Add<IntOption>(kThreadsOptionId, 0, 128) = 0;
  options->Add<IntOption>(kNNCacheSizeId, 0, 999999999) = 2000000;
  SearchParams::Populate(options);

  ConfigFile::PopulateOptions(options);
  if (is_simple) {
    options->HideAllOptions();
    options->UnhideOption(kThreadsOptionId);
    options->UnhideOption(NetworkFactory::kWeightsId);
    options->UnhideOption(SearchParams::kContemptId);
    options->UnhideOption(SearchParams::kMultiPvId);
  }
  options->Add<StringOption>(kSyzygyTablebaseId);
  // Add "Ponder" option to signal to GUIs that we support pondering.
  // This option is currently not used by lc0 in any way.
  options->Add<BoolOption>(kPonderId) = true;
  options->Add<BoolOption>(kUciChess960) = false;
  options->Add<BoolOption>(kShowWDL) = false;
  options->Add<BoolOption>(kShowMovesleft) = false;

  PopulateTimeManagementOptions(is_simple ? RunType::kSimpleUci : RunType::kUci,
                                options);

  options->Add<BoolOption>(kStrictUciTiming) = false;
  options->HideOption(kStrictUciTiming);

  options->Add<BoolOption>(kPreload) = false;
  options->Add<BoolOption>(kValueOnly) = false;
  options->Add<ButtonOption>(kClearTree);
  options->HideOption(kClearTree);
}

void EngineController::ResetMoveTimer() {
  move_start_time_ = std::chrono::steady_clock::now();
}

// Updates values from Uci options.
void EngineController::UpdateFromUciOptions(const OptionsDict& options) {
  SharedLock lock(busy_mutex_);

  // Syzygy tablebases.
  std::string tb_paths = options.Get<std::string>(kSyzygyTablebaseId);
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
      NetworkFactory::BackendConfiguration(options);
  if (network_configuration_ != network_configuration) {
    network_ = NetworkFactory::LoadNetwork(options);
    network_configuration_ = network_configuration;
  }

  // Cache size.
  cache_.SetCapacity(options.Get<int>(kNNCacheSizeId));

  // Check whether we can update the move timer in "Go".
  strict_uci_timing_ = options.Get<bool>(kStrictUciTiming);
}

void EngineController::EnsureReady() {
  std::unique_lock<RpSharedMutex> lock(busy_mutex_);
  // If a UCI host is waiting for our ready response, we can consider the move
  // not started until we're done ensuring ready.
  ResetMoveTimer();
}

void EngineController::NewGame() {
  // In case anything relies upon defaulting to default position and just calls
  // newgame and goes straight into go.
  ResetMoveTimer();
  SharedLock lock(busy_mutex_);
  cache_.Clear();
  search_.reset();
  tree_.reset();
  CreateFreshTimeManager(options_.GetSubdict("white"));
  current_position_ = {ChessBoard::kStartposFen, {}};
}

void EngineController::SetPosition(const std::string& fen,
                                   const std::vector<std::string>& moves_str) {
  // Some UCI hosts just call position then immediately call go, while starting
  // the clock on calling 'position'.
  ResetMoveTimer();
  SharedLock lock(busy_mutex_);
  current_position_ = CurrentPosition{fen, moves_str};
  search_.reset();
}

Position EngineController::ApplyPositionMoves() {
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

bool EngineController::SetupPosition(
    const std::string& fen, const std::vector<std::string>& moves_str) {
  SharedLock lock(busy_mutex_);
  search_.reset();

  if (!tree_) tree_ = std::make_unique<NodeTree>();

  std::vector<Move> moves;
  for (const auto& move : moves_str) moves.emplace_back(move);
  return tree_->ResetToPosition(fen, moves);
}

void EngineController::CreateFreshTimeManager(const OptionsDict& options) {
  time_manager_ = MakeTimeManager(options);
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

void ValueOnlyGo(NodeTree* tree, Network* network, const OptionsDict& options,
                 std::unique_ptr<UciResponder> responder) {
  auto input_format = network->GetCapabilities().input_format;

  const auto& board = tree->GetPositionHistory().Last().GetBoard();
  auto legal_moves = board.GenerateLegalMoves();
  tree->GetCurrentHead()->CreateEdges(legal_moves);
  PositionHistory history = tree->GetPositionHistory();
  std::vector<InputPlanes> planes;
  for (auto edge : tree->GetCurrentHead()->Edges()) {
    history.Append(edge.GetMove());
    if (history.ComputeGameResult() == GameResult::UNDECIDED) {
      planes.emplace_back(EncodePositionForNN(
          input_format, history, 8, FillEmptyHistory::FEN_ONLY, nullptr));
    }
    history.Pop();
  }

  std::vector<float> comp_q;
  int batch_size = options.Get<int>(SearchParams::kMiniBatchSizeId);
  if (batch_size == 0) batch_size = network->GetMiniBatchSize();

  for (size_t i = 0; i < planes.size(); i += batch_size) {
    auto comp = network->NewComputation();
    for (int j = 0; j < batch_size; j++) {
      comp->AddInput(std::move(planes[i + j]));
      if (i + j + 1 == planes.size()) break;
    }
    comp->ComputeBlocking();

    for (int j = 0; j < batch_size; j++) comp_q.push_back(comp->GetQVal(j));
  }

  Move best;
  int comp_idx = 0;
  float max_q = std::numeric_limits<float>::lowest();
  for (auto edge : tree->GetCurrentHead()->Edges()) {
    history.Append(edge.GetMove());
    auto result = history.ComputeGameResult();
    float q = -1;
    if (result == GameResult::UNDECIDED) {
      // NN eval is for side to move perspective - so if its good, its bad for
      // us.
      q = -comp_q[comp_idx];
      comp_idx++;
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

void EngineController::Go(const GoParams& params) {
  // TODO: should consecutive calls to go be considered to be a continuation and
  // hence have the same start time like this behaves, or should we check start
  // time hasn't changed since last call to go and capture the new start time
  // now?
  if (strict_uci_timing_ || !move_start_time_) ResetMoveTimer();
  go_params_ = params;

  std::unique_ptr<UciResponder> responder =
      std::make_unique<NonOwningUciRespondForwarder>(uci_responder_.get());

  // Setting up current position, now that it's known whether it's ponder or
  // not.
  bool is_same_game;
  if (params.ponder && !current_position_.moves.empty()) {
    std::vector<std::string> moves(current_position_.moves);
    std::string ponder_move = moves.back();
    moves.pop_back();
    is_same_game = SetupPosition(current_position_.fen, moves);
    responder = std::make_unique<PonderResponseTransformer>(
        std::move(responder), ponder_move);
  } else {
    is_same_game =
        SetupPosition(current_position_.fen, current_position_.moves);
  }

  auto& options =
      options_.GetSubdict(tree_->IsBlackToMove() ? "black" : "white");

  UpdateFromUciOptions(options);
  if (!is_same_game) CreateFreshTimeManager(options);

  if (!options.Get<bool>(kUciChess960)) {
    // Remap FRC castling to legacy castling.
    responder = std::make_unique<Chess960Transformer>(
        std::move(responder), tree_->HeadPosition().GetBoard());
  }

  if (!options.Get<bool>(kShowWDL)) {
    // Strip WDL information from the response.
    responder = std::make_unique<WDLResponseFilter>(std::move(responder));
  }

  if (!options.Get<bool>(kShowMovesleft)) {
    // Strip movesleft information from the response.
    responder = std::make_unique<MovesLeftResponseFilter>(std::move(responder));
  }
  if (options_.Get<bool>(kValueOnly)) {
    ValueOnlyGo(tree_.get(), network_.get(), options_, std::move(responder));
    return;
  }

  if (options_.Get<Button>(kClearTree).TestAndReset()) {
    tree_->TrimTreeAtHead();
  }

  auto stopper = time_manager_->GetStopper(params, *tree_.get());
  search_ = std::make_unique<Search>(
      *tree_, network_.get(), std::move(responder),
      StringsToMovelist(params.searchmoves, tree_->HeadPosition().GetBoard()),
      *move_start_time_, std::move(stopper), params.infinite, params.ponder,
      options, &cache_, syzygy_tb_.get());

  LOGFILE << "Timer started at "
          << FormatTime(SteadyClockToSystemClock(*move_start_time_));
  search_->StartThreads(options.Get<int>(kThreadsOptionId));
}

void EngineController::PonderHit() {
  ResetMoveTimer();
  go_params_.ponder = false;
  Go(go_params_);
}

void EngineController::Stop() {
  if (search_) search_->Stop();
}

EngineLoop::EngineLoop()
    : engine_(
          std::make_unique<CallbackUciResponder>(
              std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
              std::bind(&UciLoop::SendInfo, this, std::placeholders::_1)),
          options_.GetOptionsDict()) {
  engine_.PopulateOptions(&options_);
  options_.Add<StringOption>(kLogFileId);
}

void EngineLoop::RunLoop() {
  if (!ConfigFile::Init() || !options_.ProcessAllFlags()) return;
  const auto options = options_.GetOptionsDict();
  Logging::Get().SetFilename(options.Get<std::string>(kLogFileId));
  if (options.Get<bool>(kPreload)) engine_.NewGame();
  UciLoop::RunLoop();
}

void EngineLoop::CmdUci() {
  SendId();
  for (const auto& option : options_.ListOptionsUci()) {
    SendResponse(option);
  }
  SendResponse("uciok");
}

void EngineLoop::CmdIsReady() {
  engine_.EnsureReady();
  SendResponse("readyok");
}

void EngineLoop::CmdSetOption(const std::string& name, const std::string& value,
                              const std::string& context) {
  options_.SetUciOption(name, value, context);
  if (context.empty()) {
    options_.SetUciOption(name, value, "white");
    options_.SetUciOption(name, value, "black");
  }
  // Set the log filename for the case it was set in UCI option.
  Logging::Get().SetFilename(
      options_.GetOptionsDict().Get<std::string>(kLogFileId));
}

void EngineLoop::CmdUciNewGame() { engine_.NewGame(); }

void EngineLoop::CmdPosition(const std::string& position,
                             const std::vector<std::string>& moves) {
  std::string fen = position;
  if (fen.empty()) {
    fen = ChessBoard::kStartposFen;
  }
  engine_.SetPosition(fen, moves);
}

void EngineLoop::CmdFen() {
  std::string fen = GetFen(engine_.ApplyPositionMoves());
  return SendResponse(fen);
}
void EngineLoop::CmdGo(const GoParams& params) { engine_.Go(params); }

void EngineLoop::CmdPonderHit() { engine_.PonderHit(); }

void EngineLoop::CmdStop() { engine_.Stop(); }

}  // namespace lczero

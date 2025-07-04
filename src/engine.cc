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
#include <chrono>
#include <thread>
#include "utils/random.h"
#include "syzygy/syzygy.h"

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

class PonderResponseTransformer : public TransformingUciResponder {
 public:
  PonderResponseTransformer(std::unique_ptr<UciResponder> parent,
                            std::string ponder_move)
      : TransformingUciResponder(std::move(parent)),
        ponder_move_(std::move(ponder_move)) {}

  void TransformThinkingInfo(std::vector<ThinkingInfo>* infos) override {
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
void EngineController::UpdateFromUciOptions() {
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
    network_ = NetworkFactory::LoadNetwork(options_);
    network_configuration_ = network_configuration;
  }

  // Cache size.
  cache_.SetCapacity(options_.Get<int>(kNNCacheSizeId));

  // Check whether we can update the move timer in "Go".
  strict_uci_timing_ = options_.Get<bool>(kStrictUciTiming);
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
  CreateFreshTimeManager();
  current_position_ = {ChessBoard::kStartposFen, {}};
  UpdateFromUciOptions();
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

void EngineController::SetupPosition(
    const std::string& fen, const std::vector<std::string>& moves_str) {
  SharedLock lock(busy_mutex_);
  search_.reset();

  UpdateFromUciOptions();

  if (!tree_) tree_ = std::make_unique<NodeTree>();

  std::vector<Move> moves;
  for (const auto& move : moves_str) moves.emplace_back(move);
  const bool is_same_game = tree_->ResetToPosition(fen, moves);
  if (!is_same_game) CreateFreshTimeManager();
}

void EngineController::CreateFreshTimeManager() {
  time_manager_ = MakeTimeManager(options_);
}

namespace {

struct MoveEvaluation {
  Move move;
  float value;
  float policy;
  float wdl[3];
  bool is_tablebase;
  WDLScore tb_result;
};

class PolicyheadSearch {
public:
  PolicyheadSearch(NodeTree* tree, Network* network, const OptionsDict& options,
                   std::unique_ptr<UciResponder> responder, 
                   SyzygyTablebase* syzygy_tb)
      : tree_(tree),
        network_(network),
        options_(options),
        responder_(std::move(responder)),
        syzygy_tb_(syzygy_tb),
        params_(options),
        input_format_(network->GetCapabilities().input_format),
        move_number_(tree->GetPositionHistory().GetLength() / 2 + 1) {}

  void Go() {
    evaluations_.clear();
    EvaluatePosition();
    
    if (params_.GetPolicyheadThinkTime() > 0) {
      SimulateThinking();
    }
    
    OutputFinalInfo();
    
    Move best_move = SelectBestMove();
    BestMoveInfo info(best_move);
    responder_->OutputBestMove(&info);
  }

private:
  void EvaluatePosition() {
    const auto& board = tree_->GetPositionHistory().Last().GetBoard();
    auto legal_moves = board.GenerateLegalMoves();
    tree_->GetCurrentHead()->CreateEdges(legal_moves);
    
    PositionHistory history = tree_->GetPositionHistory();
    std::vector<InputPlanes> planes;
    std::vector<Move> moves_for_eval;
    std::vector<bool> is_terminal;
    
    for (auto edge : tree_->GetCurrentHead()->Edges()) {
      MoveEvaluation eval;
      eval.move = edge.GetMove(tree_->GetPositionHistory().IsBlackToMove());
      eval.is_tablebase = false;
      eval.tb_result = WDL_DRAW;
      
      history.Append(edge.GetMove());
      auto result = history.ComputeGameResult();
      
      if (result != GameResult::UNDECIDED) {
        if (result == GameResult::DRAW) {
          eval.value = 0.0f;
          eval.wdl[0] = 0.0f; eval.wdl[1] = 1.0f; eval.wdl[2] = 0.0f;
        } else {
          eval.value = 1.0f;
          eval.wdl[0] = 1.0f; eval.wdl[1] = 0.0f; eval.wdl[2] = 0.0f;
        }
        eval.policy = 0.0f;
        is_terminal.push_back(true);
      } else {
        bool found_in_tb = false;
        if (syzygy_tb_) {
          const auto& pos = history.Last();
          if (pos.GetBoard().castlings().no_legal_castle() &&
              (pos.GetBoard().ours() | pos.GetBoard().theirs()).count() <= 
               syzygy_tb_->max_cardinality()) {
            ProbeState state;
            WDLScore wdl = syzygy_tb_->probe_wdl(pos, &state);
            if (state != FAIL) {
              eval.is_tablebase = true;
              eval.tb_result = wdl;
              found_in_tb = true;
              
              if (wdl == WDL_WIN) {
                eval.value = 1.0f;
                eval.wdl[0] = 1.0f; eval.wdl[1] = 0.0f; eval.wdl[2] = 0.0f;
              } else if (wdl == WDL_LOSS) {
                eval.value = -1.0f;
                eval.wdl[0] = 0.0f; eval.wdl[1] = 0.0f; eval.wdl[2] = 1.0f;
              } else {
                eval.value = 0.0f;
                eval.wdl[0] = 0.0f; eval.wdl[1] = 1.0f; eval.wdl[2] = 0.0f;
              }
              eval.policy = 0.0f;
            }
          }
        }
        
        if (!found_in_tb) {
          planes.emplace_back(EncodePositionForNN(
              input_format_, history, 8, FillEmptyHistory::FEN_ONLY, nullptr));
          moves_for_eval.push_back(eval.move);
        }
        is_terminal.push_back(found_in_tb);
      }
      
      evaluations_.push_back(eval);
      history.Pop();
    }
    
    if (!planes.empty()) {
      EvaluateWithNetwork(planes, moves_for_eval, is_terminal);
    }
    
    SortEvaluations();
  }
  
  void EvaluateWithNetwork(const std::vector<InputPlanes>& planes,
                          const std::vector<Move>&,
                          const std::vector<bool>& is_terminal) {
    std::vector<float> comp_q;
    std::vector<float> comp_d;
    std::vector<float> comp_policy;
    
    int batch_size = options_.Get<int>(SearchParams::kMiniBatchSizeId);
    if (batch_size == 0) batch_size = network_->GetMiniBatchSize();
    
    auto root_computation = network_->NewComputation();
    root_computation->AddInput(EncodePositionForNN(
        input_format_, tree_->GetPositionHistory(), 8, 
        FillEmptyHistory::FEN_ONLY, nullptr));
    root_computation->ComputeBlocking();
    
    for (size_t i = 0; i < planes.size(); i += batch_size) {
      auto comp = network_->NewComputation();
      size_t batch_end = std::min(i + batch_size, planes.size());
      
      for (size_t j = i; j < batch_end; j++) {
        comp->AddInput(std::move(const_cast<InputPlanes&>(planes[j])));
      }
      comp->ComputeBlocking();
      
      for (size_t j = 0; j < batch_end - i; j++) {
        float q_val = comp->GetQVal(j);
        float d_val = comp->GetDVal(j);
        comp_q.push_back(-q_val);
        comp_d.push_back(d_val);
      }
    }
    
    int eval_idx = 0;
    auto edges = tree_->GetCurrentHead()->Edges();
    for (size_t i = 0; i < evaluations_.size(); i++) {
      if (!is_terminal[i]) {
        evaluations_[i].value = comp_q[eval_idx];
        evaluations_[i].wdl[0] = (1.0f + evaluations_[i].value - comp_d[eval_idx]) / 2.0f;
        evaluations_[i].wdl[1] = comp_d[eval_idx];
        evaluations_[i].wdl[2] = (1.0f - evaluations_[i].value - comp_d[eval_idx]) / 2.0f;
        
        auto edge_iter = edges.begin();
        for (size_t j = 0; j < i; j++) ++edge_iter;
        evaluations_[i].policy = root_computation->GetPVal(
            0, edge_iter.GetMove().as_nn_index(0));
        eval_idx++;
      }
    }
  }
  
  void SortEvaluations() {
    std::sort(evaluations_.begin(), evaluations_.end(),
              [](const MoveEvaluation& a, const MoveEvaluation& b) {
                return a.value > b.value;
              });
  }
  
  Move SelectBestMove() {
    if (evaluations_.empty()) return Move();
    
    float temperature = params_.GetPolicyheadTemperature();
    if (temperature > 0.0f) {
      float decay = params_.GetPolicyheadTempDecay();
      temperature = std::max(0.0f, temperature - decay * (move_number_ - 1));
    }
    
    if (temperature <= 0.0f) {
      return evaluations_[0].move;
    }
    
    std::vector<float> weights;
    float max_val = evaluations_[0].value;
    for (const auto& eval : evaluations_) {
      weights.push_back(std::exp((eval.value - max_val) / temperature));
    }
    
    float total_weight = 0.0f;
    for (float w : weights) total_weight += w;
    
    float random_val = Random::Get().GetFloat(total_weight);
    float cumulative = 0.0f;
    
    for (size_t i = 0; i < weights.size(); i++) {
      cumulative += weights[i];
      if (random_val <= cumulative) {
        return evaluations_[i].move;
      }
    }
    
    return evaluations_[0].move;
  }
  
  void SimulateThinking() {
    int think_time = params_.GetPolicyheadThinkTime();
    if (think_time <= 0) return;
    
    int updates = std::max(1, think_time / 1000);
    int interval = think_time / updates;
    
    for (int i = 0; i < updates; i++) {
      std::this_thread::sleep_for(std::chrono::milliseconds(interval));
      OutputThinkingInfo(i + 1, updates);
    }
  }
  
  void OutputThinkingInfo(int current_update = 0, int total_updates = 1) {
    std::vector<ThinkingInfo> infos;
    
    int multipv = params_.GetPolicyheadMultiPv();
    bool verbose = params_.GetPolicyheadVerbose();
    
    for (int pv = 0; pv < std::min(multipv, static_cast<int>(evaluations_.size())); pv++) {
      ThinkingInfo info;
      info.depth = 1;
      info.seldepth = 1;
      info.time = current_update * (params_.GetPolicyheadThinkTime() / total_updates);
      info.nodes = evaluations_.size();
      info.multipv = pv + 1;
      
      const auto& eval = evaluations_[pv];
      info.score = static_cast<int>(eval.value * 100);
      info.pv.push_back(eval.move);
      
      if (verbose || current_update == total_updates) {
        info.wdl = std::make_optional<ThinkingInfo::WDL>();
        info.wdl->w = static_cast<int>(eval.wdl[0] * 1000);
        info.wdl->d = static_cast<int>(eval.wdl[1] * 1000);
        info.wdl->l = static_cast<int>(eval.wdl[2] * 1000);
        
        if (eval.is_tablebase) {
          info.comment = "TB";
        }
      }
      
      infos.push_back(info);
    }
    
    responder_->OutputThinkingInfo(&infos);
  }
  
  void OutputFinalInfo() {
    OutputThinkingInfo(1, 1);
  }

  NodeTree* tree_;
  Network* network_;
  const OptionsDict& options_;
  std::unique_ptr<UciResponder> responder_;
  SyzygyTablebase* syzygy_tb_;
  SearchParams params_;
  pblczero::NetworkFormat::InputFormat input_format_;
  int move_number_;
  std::vector<MoveEvaluation> evaluations_;
};

void PolicyheadGo(NodeTree* tree, Network* network, const OptionsDict& options,
                  std::unique_ptr<UciResponder> responder, 
                  SyzygyTablebase* syzygy_tb) {
  PolicyheadSearch search(tree, network, options, std::move(responder), 
                         syzygy_tb);
  search.Go();
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
    PolicyheadGo(tree_.get(), network_.get(), options_, std::move(responder), syzygy_tb_.get());
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
      options_, &cache_, syzygy_tb_.get());

  LOGFILE << "Timer started at "
          << FormatTime(SteadyClockToSystemClock(*move_start_time_));
  search_->StartThreads(options_.Get<int>(kThreadsOptionId));
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

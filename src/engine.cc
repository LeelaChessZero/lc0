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

#include <algorithm>
#include <cmath>
#include <functional>

#include "engine.h"
#include "mcts/search.h"
#include "utils/configfile.h"
#include "utils/logging.h"

namespace lczero {
namespace {
const int kDefaultThreads = 2;

const OptionId kThreadsOptionId{"threads", "Threads",
                                "Number of (CPU) worker threads to use.", 't'};
const OptionId kLogFileId{"logfile", "LogFile",
                          "Write log to that file. Special value <stderr> to "
                          "output the log to the console.",
                          'l'};
const OptionId kNNCacheSizeId{
    "nncache", "NNCacheSize",
    "Number of positions to store in a memory cache. A large cache can speed "
    "up searching, but takes memory."};
const OptionId kSlowMoverId{
    "slowmover", "Slowmover",
    "Budgeted time for a move is multiplied by this value, causing the engine "
    "to spend more time (if value is greater than 1) or less time (if the "
    "value is less than 1)."};
const OptionId kMoveOverheadId{
    "move-overhead", "MoveOverheadMs",
    "Amount of time, in milliseconds, that the engine subtracts from it's "
    "total available time (to compensate for slow connection, interprocess "
    "communication, etc)."};
const OptionId kTimePeakPlyId{"time-peak-halfmove", "TimePeakHalfmove",
                              "For which halfmove the time budgeting algorithm "
                              "should allocate the maximum amount of time."};
const OptionId kTimeLeftWidthId{
    "time-left-width", "TimeLeftWidth",
    "\"Width\" of time budget graph to the left of the peak value. For small "
    "values, moves far from the peak will get little time; for larger values, "
    "they will get almost the same time as the peak move."};
const OptionId kTimeRightWidthId{
    "time-right-width", "TimeRightWidth",
    "\"Width\" of time budget graph to the right of the peak value. For small "
    "values, moves far from the peak will get little time; for larger values, "
    "they will get almost the same time as the peak move."};
const OptionId kSyzygyTablebaseId{
    "syzygy-paths", "SyzygyPath",
    "List of Syzygy tablebase directories, list entries separated by system "
    "separator (\";\" for Windows, \":\" for Linux).",
    's'};
const OptionId kSpendSavedTimeId{
    "immediate-time-use", "ImmediateTimeUse",
    "Fraction of time saved by smart pruning, which is added to the budget to "
    "the next move rather than to the entire game. When 1, all saved time is "
    "added to the next move's budget; when 0, saved time is distributed among "
    "all future moves."};
const OptionId kPonderId{"ponder", "Ponder",
                         "This option is ignored. Here to please chess GUIs."};
// Warning! When changed, also change number 30 in the help below!
const size_t kAvgMovesPerPosition = 30;
const OptionId kRamLimitMbId{
    "ramlimit-mb", "RamLimitMb",
    "Maximum memory usage for the engine, in megabytes. The estimation is very "
    "rough, and can be off by a lot. For example, multiple visits to a "
    "terminal node counted several times, and the estimation assumes that all "
    "positions have 30 possible moves. When set to 0, no RAM limit is "
    "enforced."};

const size_t kAvgNodeSize = sizeof(Node) + kAvgMovesPerPosition * sizeof(Edge);
const size_t kAvgCacheItemSize =
    NNCache::GetItemStructSize() + sizeof(CachedNNRequest) +
    sizeof(CachedNNRequest::IdxAndProb) * kAvgMovesPerPosition;

float ComputeMoveWeight(int ply, float peak, float left_width,
                        float right_width) {
  // Inflection points of the function are at ply = peak +/- width.
  // At these points the function is at 2/3 of its max value.
  const float width = ply > peak ? right_width : left_width;
  constexpr float width_scaler = 1.518651485f;  // 2 / log(2 + sqrt(3))
  return std::pow(std::cosh((ply - peak) / width / width_scaler), -2.0f);
}

}  // namespace

EngineController::EngineController(BestMoveInfo::Callback best_move_callback,
                                   ThinkingInfo::Callback info_callback,
                                   const OptionsDict& options)
    : options_(options),
      best_move_callback_(best_move_callback),
      info_callback_(info_callback),
      move_start_time_(std::chrono::steady_clock::now()) {}

void EngineController::PopulateOptions(OptionsParser* options) {
  using namespace std::placeholders;

  options->Add<IntOption>(kThreadsOptionId, 1, 128) = kDefaultThreads;
  options->Add<IntOption>(kNNCacheSizeId, 0, 999999999) = 200000;
  options->Add<FloatOption>(kSlowMoverId, 0.0f, 100.0f) = 1.0f;
  options->Add<IntOption>(kMoveOverheadId, 0, 100000000) = 200;
  options->Add<FloatOption>(kTimePeakPlyId, -1000.0f, 1000.0f) = 26.2f;
  options->Add<FloatOption>(kTimeLeftWidthId, 0.0f, 1000.0f) = 82.0f;
  options->Add<FloatOption>(kTimeRightWidthId, 0.0f, 1000.0f) = 74.0f;
  options->Add<StringOption>(kSyzygyTablebaseId);
  // Add "Ponder" option to signal to GUIs that we support pondering.
  // This option is currently not used by lc0 in any way.
  options->Add<BoolOption>(kPonderId) = true;
  options->Add<FloatOption>(kSpendSavedTimeId, 0.0f, 1.0f) = 0.6f;
  options->Add<IntOption>(kRamLimitMbId, 0, 100000000) = 0;

  // Hide time curve options.
  options->HideOption(kTimePeakPlyId);
  options->HideOption(kTimeLeftWidthId);
  options->HideOption(kTimeRightWidthId);

  NetworkFactory::PopulateOptions(options);
  SearchParams::Populate(options);
  ConfigFile::PopulateOptions(options);

  auto defaults = options->GetMutableDefaultsOptions();

  defaults->Set<int>(SearchParams::kMiniBatchSizeId.GetId(), 256);
  defaults->Set<float>(SearchParams::kFpuReductionId.GetId(), 1.2f);
  defaults->Set<float>(SearchParams::kCpuctId.GetId(), 3.4f);
  defaults->Set<float>(SearchParams::kTradePenaltyId.GetId(), 0.0f);
  defaults->Set<float>(SearchParams::kTradePenalty2Id.GetId(), 0.0f);
  defaults->Set<float>(SearchParams::kContemptId.GetId(), 0.0f);
  defaults->Set<float>(SearchParams::kPolicySoftmaxTempId.GetId(), 2.2f);
  defaults->Set<int>(SearchParams::kMaxCollisionVisitsId.GetId(), 9999);
  defaults->Set<int>(SearchParams::kMaxCollisionEventsId.GetId(), 32);
  defaults->Set<int>(SearchParams::kCacheHistoryLengthId.GetId(), 0);
  defaults->Set<bool>(SearchParams::kOutOfOrderEvalId.GetId(), true);
}

SearchLimits EngineController::PopulateSearchLimits(
    int ply, bool is_black, const GoParams& params,
    std::chrono::steady_clock::time_point start_time) {
  SearchLimits limits;
  int64_t move_overhead = options_.Get<int>(kMoveOverheadId.GetId());
  if (params.movetime >= 0) {
    limits.search_deadline =
        start_time + std::chrono::milliseconds(params.movetime - move_overhead);
  }

  const optional<int64_t>& time = (is_black ? params.btime : params.wtime);
  if (!params.searchmoves.empty()) {
    limits.searchmoves.reserve(params.searchmoves.size());
    for (const auto& move : params.searchmoves) {
      limits.searchmoves.emplace_back(move, is_black);
    }
  }
  limits.infinite = params.infinite || params.ponder;
  if (params.nodes >= 0) limits.visits = params.nodes;
  int ram_limit = options_.Get<int>(kRamLimitMbId.GetId());
  if (ram_limit) {
    const auto cache_size =
        options_.Get<int>(kNNCacheSizeId.GetId()) * kAvgCacheItemSize;
    int64_t limit = (ram_limit * 1000000LL - cache_size) / kAvgNodeSize;
    LOGFILE << "RAM limit " << ram_limit << "MB. Cache takes "
            << cache_size / 1000000 << "MB. Remaining memory is enough for "
            << limit << " nodes.";
    if (limit < 0) limit = 0;
    if (limit < limits.visits) limits.visits = limit;
  }
  limits.depth = params.depth;
  if (limits.infinite || !time) return limits;
  int increment = std::max(int64_t(0), is_black ? params.binc : params.winc);

  int movestogo = params.movestogo < 0 ? 50 : params.movestogo;
  // Fix non-standard uci command.
  if (movestogo == 0) movestogo = 1;

  // How to scale moves time.
  float slowmover = options_.Get<float>(kSlowMoverId.GetId());
  float time_curve_peak = options_.Get<float>(kTimePeakPlyId.GetId());
  float time_curve_left_width = options_.Get<float>(kTimeLeftWidthId.GetId());
  float time_curve_right_width = options_.Get<float>(kTimeRightWidthId.GetId());

  // Total time till control including increments.
  auto total_moves_time =
      std::max(int64_t{0}, *time + increment * (movestogo - 1) - move_overhead);

  // If there is time spared from previous searches, the `time_to_squander` part
  // of it will be used immediately, remove that from planning.
  int time_to_squander = 0;
  if (time_spared_ms_ > 0) {
    time_to_squander =
        time_spared_ms_ * options_.Get<float>(kSpendSavedTimeId.GetId());
    time_spared_ms_ -= time_to_squander;
    total_moves_time -= time_to_squander;
  }

  constexpr int kSmartPruningToleranceMs = 200;
  float this_move_weight = ComputeMoveWeight(
      ply, time_curve_peak, time_curve_left_width, time_curve_right_width);
  float other_move_weights = 0.0f;
  for (int i = 1; i < movestogo; ++i)
    other_move_weights +=
        ComputeMoveWeight(ply + 2 * i, time_curve_peak, time_curve_left_width,
                          time_curve_right_width);
  // Compute the move time without slowmover.
  float this_move_time = total_moves_time * this_move_weight /
                         (this_move_weight + other_move_weights);

  // Only extend thinking time with slowmover if smart pruning can potentially
  // reduce it.
  if (slowmover < 1.0 ||
      this_move_time * slowmover > kSmartPruningToleranceMs) {
    this_move_time *= slowmover;
    // If time is planned to be overused because of slowmover, remove excess
    // of that time from spared time.
    time_spared_ms_ -= this_move_time * (slowmover - 1);
  }

  LOGFILE << "Budgeted time for the move: " << this_move_time << "ms(+"
          << time_to_squander << "ms to squander -"
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start_time)
                 .count()
          << "ms already passed). Remaining time " << *time << "ms(-"
          << move_overhead << "ms overhead)";
  // Use `time_to_squander` time immediately.
  this_move_time += time_to_squander;

  // Make sure we don't exceed current time limit with what we calculated.
  limits.search_deadline =
      start_time +
      std::chrono::milliseconds(std::min(static_cast<int64_t>(this_move_time),
                                         *time - move_overhead));
  return limits;
}

// Updates values from Uci options.
void EngineController::UpdateFromUciOptions() {
  SharedLock lock(busy_mutex_);

  // Syzygy tablebases.
  std::string tb_paths = options_.Get<std::string>(kSyzygyTablebaseId.GetId());
  if (!tb_paths.empty() && tb_paths != tb_paths_) {
    syzygy_tb_ = std::make_unique<SyzygyTablebase>();
    CERR << "Loading Syzygy tablebases from " << tb_paths;
    if (!syzygy_tb_->init(tb_paths)) {
      CERR << "Failed to load Syzygy tablebases!";
      syzygy_tb_ = nullptr;
    } else {
      tb_paths_ = tb_paths;
    }
  }

  // Network.
  auto network_configuration = NetworkFactory::BackendConfiguration(options_);
  if (network_configuration_ != network_configuration) {
    network_ = NetworkFactory::LoadNetwork(options_);
    network_configuration_ = network_configuration;
  }

  // Cache size.
  cache_.SetCapacity(options_.Get<int>(kNNCacheSizeId.GetId()));
}

void EngineController::EnsureReady() {
  UpdateFromUciOptions();
  std::unique_lock<RpSharedMutex> lock(busy_mutex_);
  // If a UCI host is waiting for our ready response, we can consider the move
  // not started until we're done ensuring ready.
  move_start_time_ = std::chrono::steady_clock::now();
}

void EngineController::NewGame() {
  // In case anything relies upon defaulting to default position and just calls
  // newgame and goes straight into go.
  move_start_time_ = std::chrono::steady_clock::now();
  SharedLock lock(busy_mutex_);
  cache_.Clear();
  search_.reset();
  tree_.reset();
  time_spared_ms_ = 0;
  current_position_.reset();
  UpdateFromUciOptions();
}

void EngineController::SetPosition(const std::string& fen,
                                   const std::vector<std::string>& moves_str) {
  // Some UCI hosts just call position then immediately call go, while starting
  // the clock on calling 'position'.
  move_start_time_ = std::chrono::steady_clock::now();
  SharedLock lock(busy_mutex_);
  current_position_ = CurrentPosition{fen, moves_str};
  search_.reset();
}

void EngineController::SetupPosition(
    const std::string& fen, const std::vector<std::string>& moves_str) {
  SharedLock lock(busy_mutex_);
  search_.reset();

  if (!tree_) tree_ = std::make_unique<NodeTree>();

  std::vector<Move> moves;
  for (const auto& move : moves_str) moves.emplace_back(move);
  bool is_same_game = tree_->ResetToPosition(fen, moves);
  if (!is_same_game) time_spared_ms_ = 0;
  UpdateFromUciOptions();
}

void EngineController::Go(const GoParams& params) {
  // TODO: should consecutive calls to go be considered to be a continuation and
  // hence have the same start time like this behaves, or should we check start
  // time hasn't changed since last call to go and capture the new start time
  // now?
  auto start_time = move_start_time_;
  go_params_ = params;

  ThinkingInfo::Callback info_callback(info_callback_);
  BestMoveInfo::Callback best_move_callback(best_move_callback_);

  // Setting up current position, now that it's known whether it's ponder or
  // not.
  if (current_position_) {
    if (params.ponder && !current_position_->moves.empty()) {
      std::vector<std::string> moves(current_position_->moves);
      std::string ponder_move = moves.back();
      moves.pop_back();
      SetupPosition(current_position_->fen, moves);

      info_callback = [this,
                       ponder_move](const std::vector<ThinkingInfo>& infos) {
        ThinkingInfo ponder_info;
        // Output all stats from main variation (not necessary the ponder move)
        // but PV only from ponder move.
        for (const auto& info : infos) {
          if (info.multipv <= 1) {
            ponder_info = info;
            if (ponder_info.score) ponder_info.score = -*ponder_info.score;
            if (ponder_info.depth > 1) ponder_info.depth--;
            if (ponder_info.seldepth > 1) ponder_info.seldepth--;
            ponder_info.pv.clear();
          }
          if (!info.pv.empty() && info.pv[0].as_string() == ponder_move) {
            ponder_info.pv.assign(info.pv.begin() + 1, info.pv.end());
          }
        }
        info_callback_({ponder_info});
      };
    } else {
      SetupPosition(current_position_->fen, current_position_->moves);
    }
  } else if (!tree_) {
    SetupPosition(ChessBoard::kStartposFen, {});
  }

  auto limits = PopulateSearchLimits(
      tree_->GetPlyCount(), tree_->IsBlackToMove(), params, start_time);
  LOGFILE << "Limits: " << limits.DebugString();

  // If there is a time limit, also store amount of time saved.
  if (limits.search_deadline) {
    best_move_callback = [this, limits](const BestMoveInfo& info) {
      best_move_callback_(info);
      if (limits.search_deadline) {
        time_spared_ms_ +=
            std::chrono::duration_cast<std::chrono::milliseconds>(
                *limits.search_deadline - std::chrono::steady_clock::now())
                .count();
      }
    };
  }

  search_ = std::make_unique<Search>(*tree_, network_.get(), best_move_callback,
                                     info_callback, limits, options_, &cache_,
                                     syzygy_tb_.get());

  if (limits.search_deadline) {
    LOGFILE << "Timer started at "
            << FormatTime(SteadyClockToSystemClock(move_start_time_));
  }
  search_->StartThreads(options_.Get<int>(kThreadsOptionId.GetId()));
}

void EngineController::PonderHit() {
  move_start_time_ = std::chrono::steady_clock::now();
  go_params_.ponder = false;
  Go(go_params_);
}

void EngineController::Stop() {
  if (search_) search_->Stop();
}

EngineLoop::EngineLoop()
    : engine_(std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
              std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
              options_.GetOptionsDict()) {
  engine_.PopulateOptions(&options_);
  options_.Add<StringOption>(kLogFileId);
}

void EngineLoop::RunLoop() {
  if (!ConfigFile::Init(&options_) || !options_.ProcessAllFlags()) return;
  Logging::Get().SetFilename(
      options_.GetOptionsDict().Get<std::string>(kLogFileId.GetId()));
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
      options_.GetOptionsDict().Get<std::string>(kLogFileId.GetId()));
}

void EngineLoop::CmdUciNewGame() { engine_.NewGame(); }

void EngineLoop::CmdPosition(const std::string& position,
                             const std::vector<std::string>& moves) {
  std::string fen = position;
  if (fen.empty()) fen = ChessBoard::kStartposFen;
  engine_.SetPosition(fen, moves);
}

void EngineLoop::CmdGo(const GoParams& params) { engine_.Go(params); }

void EngineLoop::CmdPonderHit() { engine_.PonderHit(); }

void EngineLoop::CmdStop() { engine_.Stop(); }

}  // namespace lczero

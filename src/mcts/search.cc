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

#include "mcts/search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>

#include "mcts/node.h"
#include "neural/cache.h"
#include "neural/encoder.h"
#include "utils/fastmath.h"
#include "utils/random.h"

namespace lczero {

namespace {
const int kSmartPruningToleranceNodes = 300;
const int kSmartPruningToleranceMs = 200;
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 5000;
}  // namespace

std::string SearchLimits::DebugString() const {
  std::ostringstream ss;
  ss << "visits:" << visits << " playouts:" << playouts << " depth:" << depth
     << " infinite:" << infinite;
  if (search_deadline) {
    ss << " search_deadline:"
       << FormatTime(SteadyClockToSystemClock(*search_deadline));
  }
  return ss.str();
}

Search::Search(const NodeTree& tree, Network* network,
               BestMoveInfo::Callback best_move_callback,
               ThinkingInfo::Callback info_callback, const SearchLimits& limits,
               const OptionsDict& options, NNCache* cache,
               SyzygyTablebase* syzygy_tb)
    : ok_to_respond_bestmove_(!limits.infinite),
      root_node_(tree.GetCurrentHead()),
      cache_(cache),
      syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      network_(network),
      limits_(limits),
      start_time_(std::chrono::steady_clock::now()),
      initial_visits_(root_node_->GetN()),
      best_move_callback_(best_move_callback),
      info_callback_(info_callback),
      params_(options) {}

namespace {
void ApplyDirichletNoise(Node* node, float eps, double alpha) {
  float total = 0;
  std::vector<float> noise;

  for (int i = 0; i < node->GetNumEdges(); ++i) {
    float eta = Random::Get().GetGamma(alpha, 1.0);
    noise.emplace_back(eta);
    total += eta;
  }

  if (total < std::numeric_limits<float>::min()) return;

  int noise_idx = 0;
  for (const auto& child : node->Edges()) {
    auto* edge = child.edge();
    edge->SetP(edge->GetP() * (1 - eps) + eps * noise[noise_idx++] / total);
  }
}
}  // namespace

void Search::SendUciInfo() REQUIRES(nodes_mutex_) {
  auto edges = GetBestChildrenNoTemperature(root_node_, params_.GetMultiPv());
  const auto score_type = params_.GetScoreType();

  std::vector<ThinkingInfo> uci_infos;

  // Info common for all multipv variants.
  ThinkingInfo common_info;
  common_info.depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  common_info.seldepth = max_depth_;
  common_info.time = GetTimeSinceStart();
  common_info.nodes = total_playouts_ + initial_visits_;
  common_info.hashfull =
      cache_->GetSize() * 1000LL / std::max(cache_->GetCapacity(), 1);
  common_info.nps =
      common_info.time ? (total_playouts_ * 1000 / common_info.time) : 0;
  common_info.tb_hits = tb_hits_.load(std::memory_order_acquire);

  int multipv = 0;
  const auto default_q = -root_node_->GetQ();
  for (const auto& edge : edges) {
    ++multipv;
    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();
    if (score_type == "centipawn") {
      uci_info.score = 295 * edge.GetQ(default_q) /
                       (1 - 0.976953126 * std::pow(edge.GetQ(default_q), 14));
    } else if (score_type == "centipawn_2018") {
      uci_info.score = 290.680623072 * tan(1.548090806 * edge.GetQ(default_q));
    } else if (score_type == "win_percentage") {
      uci_info.score = edge.GetQ(default_q) * 5000 + 5000;
    } else if (score_type == "Q") {
      uci_info.score = edge.GetQ(default_q) * 10000;
    }
    if (params_.GetMultiPv() > 1) uci_info.multipv = multipv;
    bool flip = played_history_.IsBlackToMove();
    for (auto iter = edge; iter;
         iter = GetBestChildNoTemperature(iter.node()), flip = !flip) {
      uci_info.pv.push_back(iter.GetMove(flip));
      if (!iter.node()) break;  // Last edge was dangling, cannot continue.
    }
  }

  if (!uci_infos.empty()) last_outputted_uci_info_ = uci_infos.front();
  if (current_best_edge_ && !edges.empty()) {
    last_outputted_info_edge_ = current_best_edge_.edge();
  }

  info_callback_(uci_infos);
}

// Decides whether anything important changed in stats and new info should be
// shown to a user.
void Search::MaybeOutputInfo() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  if (!bestmove_is_sent_ && current_best_edge_ &&
      (current_best_edge_.edge() != last_outputted_info_edge_ ||
       last_outputted_uci_info_.depth !=
           static_cast<int>(cum_depth_ /
                            (total_playouts_ ? total_playouts_ : 1)) ||
       last_outputted_uci_info_.seldepth != max_depth_ ||
       last_outputted_uci_info_.time + kUciInfoMinimumFrequencyMs <
           GetTimeSinceStart())) {
    SendUciInfo();
    if (params_.GetLogLiveStats()) {
      SendMovesStats();
    }
    if (stop_.load(std::memory_order_acquire) && !ok_to_respond_bestmove_) {
      ThinkingInfo info;
      info.comment =
          "WARNING: Search has reached limit and does not make any progress.";
      info_callback_({info});
    }
  }
}

int64_t Search::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

int64_t Search::GetTimeToDeadline() const {
  if (!limits_.search_deadline) return 0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             *limits_.search_deadline - std::chrono::steady_clock::now())
      .count();
}

namespace {
inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node) {
  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetQ() - value * std::sqrt(node->GetVisitedPolicy());
}

inline float ComputeCpuct(const SearchParams& params, uint32_t N) {
  const float init = params.GetCpuct();
  const float k = params.GetCpuctFactor();
  const float base = params.GetCpuctBase();
  return init + (k ? k * FastLog((N + base) / base) : 0.0f);
}
}  // namespace

std::vector<std::string> Search::GetVerboseStats(Node* node,
                                                 bool is_black_to_move) const {
  const float fpu = GetFpu(params_, node, node == root_node_);
  const float cpuct = ComputeCpuct(params_, node->GetN());
  const float U_coeff =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));

  std::vector<EdgeAndNode> edges;
  for (const auto& edge : node->Edges()) edges.push_back(edge);

  std::sort(
      edges.begin(), edges.end(),
      [&fpu, &U_coeff](EdgeAndNode a, EdgeAndNode b) {
        return std::forward_as_tuple(a.GetN(), a.GetQ(fpu) + a.GetU(U_coeff)) <
               std::forward_as_tuple(b.GetN(), b.GetQ(fpu) + b.GetU(U_coeff));
      });

  std::vector<std::string> infos;
  for (const auto& edge : edges) {
    std::ostringstream oss;
    oss << std::fixed;

    oss << std::left << std::setw(5)
        << edge.GetMove(is_black_to_move).as_string();

    oss << " (" << std::setw(4) << edge.GetMove().as_nn_index() << ")";

    oss << " N: " << std::right << std::setw(7) << edge.GetN() << " (+"
        << std::setw(2) << edge.GetNInFlight() << ") ";

    oss << "(P: " << std::setw(5) << std::setprecision(2) << edge.GetP() * 100
        << "%) ";

    oss << "(Q: " << std::setw(8) << std::setprecision(5) << edge.GetQ(fpu)
        << ") ";

    oss << "(D: " << std::setw(6) << std::setprecision(3) << edge.GetD()
        << ") ";

    oss << "(U: " << std::setw(6) << std::setprecision(5) << edge.GetU(U_coeff)
        << ") ";

    oss << "(Q+U: " << std::setw(8) << std::setprecision(5)
        << edge.GetQ(fpu) + edge.GetU(U_coeff) << ") ";

    oss << "(V: ";
    optional<float> v;
    if (edge.IsTerminal()) {
      v = edge.node()->GetQ();
    } else {
      NNCacheLock nneval = GetCachedNNEval(edge.node());
      if (nneval) v = -nneval->q;
    }
    if (v) {
      oss << std::setw(7) << std::setprecision(4) << *v;
    } else {
      oss << " -.----";
    }
    oss << ") ";

    if (edge.IsTerminal()) oss << "(T) ";
    infos.emplace_back(oss.str());
  }
  return infos;
}

void Search::SendMovesStats() const REQUIRES(counters_mutex_) {
  const bool is_black_to_move = played_history_.IsBlackToMove();
  auto move_stats = GetVerboseStats(root_node_, is_black_to_move);

  if (params_.GetVerboseStats()) {
    std::vector<ThinkingInfo> infos;
    std::transform(move_stats.begin(), move_stats.end(),
                   std::back_inserter(infos), [](const std::string& line) {
                     ThinkingInfo info;
                     info.comment = line;
                     return info;
                   });
    info_callback_(infos);
  } else {
    LOGFILE << "=== Move stats:";
    for (const auto& line : move_stats) LOGFILE << line;
  }
  if (final_bestmove_.HasNode()) {
    LOGFILE
        << "--- Opponent moves after: "
        << final_bestmove_.GetMove(played_history_.IsBlackToMove()).as_string();
    for (const auto& line :
         GetVerboseStats(final_bestmove_.node(), !is_black_to_move)) {
      LOGFILE << line;
    }
  }
}

NNCacheLock Search::GetCachedNNEval(Node* node) const {
  if (!node) return {};

  std::vector<Move> moves;
  for (; node != root_node_; node = node->GetParent()) {
    moves.push_back(node->GetOwnEdge()->GetMove());
  }
  PositionHistory history(played_history_);
  for (auto iter = moves.rbegin(), end = moves.rend(); iter != end; ++iter) {
    history.Append(*iter);
  }
  const auto hash = history.HashLast(params_.GetCacheHistoryLength() + 1);
  NNCacheLock nneval(cache_, hash);
  return nneval;
}

void Search::UpdateKLDGain() {
  if (params_.GetMinimumKLDGainPerNode() <= 0) return;

  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  if (total_playouts_ + initial_visits_ >=
      prev_dist_visits_total_ + params_.GetKLDGainAverageInterval()) {
    std::vector<uint32_t> new_visits;
    for (auto edge : root_node_->Edges()) {
      new_visits.push_back(edge.GetN());
    }
    if (prev_dist_.size() != 0) {
      double sum1 = 0.0;
      double sum2 = 0.0;
      for (decltype(new_visits)::size_type i = 0; i < new_visits.size(); i++) {
        sum1 += prev_dist_[i];
        sum2 += new_visits[i];
      }
      double kldgain = 0.0;
      for (decltype(new_visits)::size_type i = 0; i < new_visits.size(); i++) {
        double o_p = prev_dist_[i] / sum1;
        double n_p = new_visits[i] / sum2;
        if (prev_dist_[i] != 0) {
          kldgain += o_p * log(o_p / n_p);
        }
      }
      if (kldgain / (sum2 - sum1) < params_.GetMinimumKLDGainPerNode()) {
        kldgain_too_small_ = true;
      }
    }
    prev_dist_.swap(new_visits);
    prev_dist_visits_total_ = total_playouts_ + initial_visits_;
  }
}

void Search::MaybeTriggerStop() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  // Already responded bestmove, nothing to do here.
  if (bestmove_is_sent_) return;
  // Don't stop when the root node is not yet expanded.
  if (total_playouts_ == 0) return;

  // If not yet stopped, try to stop for different reasons.
  if (!stop_.load(std::memory_order_acquire)) {
    if (kldgain_too_small_) {
      FireStopInternal();
      LOGFILE << "Stopped search: KLDGain per node too small.";
    }
    // If smart pruning tells to stop (best move found), stop.
    if (only_one_possible_move_left_) {
      FireStopInternal();
      LOGFILE << "Stopped search: Only one move candidate left.";
    }
    // Stop if reached playouts limit.
    if (limits_.playouts >= 0 && total_playouts_ >= limits_.playouts) {
      FireStopInternal();
      LOGFILE << "Stopped search: Reached playouts limit: " << total_playouts_
              << ">=" << limits_.playouts;
    }
    // Stop if reached visits limit.
    if (limits_.visits >= 0 &&
        total_playouts_ + initial_visits_ >= limits_.visits) {
      FireStopInternal();
      LOGFILE << "Stopped search: Reached visits limit: "
              << total_playouts_ + initial_visits_ << ">=" << limits_.visits;
    }
    // Stop if reached time limit.
    if (limits_.search_deadline && GetTimeToDeadline() <= 0) {
      LOGFILE << "Stopped search: Ran out of time.";
      FireStopInternal();
    }
    // Stop if average depth reached requested depth.
    if (limits_.depth >= 0 &&
        cum_depth_ / (total_playouts_ ? total_playouts_ : 1) >=
            static_cast<unsigned int>(limits_.depth)) {
      FireStopInternal();
      LOGFILE << "Stopped search: Reached depth.";
    }
  }
  // If we are the first to see that stop is needed.
  if (stop_.load(std::memory_order_acquire) && ok_to_respond_bestmove_ &&
      !bestmove_is_sent_) {
    SendUciInfo();
    EnsureBestMoveKnown();
    SendMovesStats();
    best_move_callback_(
        {final_bestmove_.GetMove(played_history_.IsBlackToMove()),
         final_pondermove_.GetMove(!played_history_.IsBlackToMove())});
    bestmove_is_sent_ = true;
    current_best_edge_ = EdgeAndNode();
  }
}

void Search::UpdateRemainingMoves() {
  if (params_.GetSmartPruningFactor() <= 0.0f) return;
  SharedMutex::Lock lock(nodes_mutex_);
  remaining_playouts_ = std::numeric_limits<int>::max();
  // Check for how many playouts there is time remaining.
  if (limits_.search_deadline && !nps_start_time_) {
    nps_start_time_ = std::chrono::steady_clock::now();
  } else if (limits_.search_deadline) {
    const auto time_since_start =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - *nps_start_time_)
            .count();
    if (time_since_start > kSmartPruningToleranceMs) {
      const auto nps = 1000LL *
                           (total_playouts_ + kSmartPruningToleranceNodes) /
                           time_since_start +
                       1;
      const int64_t remaining_time = GetTimeToDeadline();
      // Put early_exit scaler here so calculation doesn't have to be done on
      // every node.
      const int64_t remaining_playouts =
          remaining_time * nps / params_.GetSmartPruningFactor() / 1000;
      // Don't assign directly to remaining_playouts_ as overflow is possible.
      if (remaining_playouts < remaining_playouts_)
        remaining_playouts_ = remaining_playouts;
    }
  }
  // Check how many visits are left.
  if (limits_.visits >= 0) {
    // Add kMiniBatchSize, as it's possible to exceed visits limit by that
    // number.
    const auto remaining_visits = limits_.visits - total_playouts_ -
                                  initial_visits_ + params_.GetMiniBatchSize() -
                                  1;

    if (remaining_visits < remaining_playouts_)
      remaining_playouts_ = remaining_visits;
  }
  if (limits_.playouts >= 0) {
    // Add kMiniBatchSize, as it's possible to exceed visits limit by that
    // number.
    const auto remaining_playouts =
        limits_.visits - total_playouts_ + params_.GetMiniBatchSize() + 1;
    if (remaining_playouts < remaining_playouts_)
      remaining_playouts_ = remaining_playouts;
  }
  // Even if we exceeded limits, don't go crazy by not allowing any playouts.
  if (remaining_playouts_ <= 1) remaining_playouts_ = 1;
  // Since remaining_playouts_ has changed, the logic for selecting visited root
  // nodes may also change. Use a 0 visit cancel score update to clear out any
  // cached best edge.
  root_node_->CancelScoreUpdate(0);
}

// Return the evaluation of the actual best child, regardless of temperature
// settings. This differs from GetBestMove, which does obey any temperature
// settings. So, somethimes, they may return results of different moves.
std::pair<float, float> Search::GetBestEval() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  float parent_q = -root_node_->GetQ();
  float parent_d = root_node_->GetD();
  if (!root_node_->HasChildren()) return {parent_q, parent_d};
  EdgeAndNode best_edge = GetBestChildNoTemperature(root_node_);
  return {best_edge.GetQ(parent_q), best_edge.GetD()};
}

std::pair<Move, Move> Search::GetBestMove() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  EnsureBestMoveKnown();
  return {final_bestmove_.GetMove(played_history_.IsBlackToMove()),
          final_pondermove_.GetMove(!played_history_.IsBlackToMove())};
}

std::int64_t Search::GetTotalPlayouts() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  return total_playouts_;
}

bool Search::PopulateRootMoveLimit(MoveList* root_moves) const {
  // Search moves overrides tablebase.
  if (!limits_.searchmoves.empty()) {
    *root_moves = limits_.searchmoves;
    return false;
  }
  auto board = played_history_.Last().GetBoard();
  if (!syzygy_tb_ || !board.castlings().no_legal_castle() ||
      (board.ours() | board.theirs()).count() > syzygy_tb_->max_cardinality()) {
    return false;
  }
  return syzygy_tb_->root_probe(
             played_history_.Last(),
             params_.GetSyzygyFastPlay() ||
                 played_history_.DidRepeatSinceLastZeroingMove(),
             root_moves) ||
         syzygy_tb_->root_probe_wdl(played_history_.Last(), root_moves);
}

// Computes the best move, maybe with temperature (according to the settings).
void Search::EnsureBestMoveKnown() REQUIRES(nodes_mutex_)
    REQUIRES(counters_mutex_) {
  if (bestmove_is_sent_) return;
  if (!root_node_->HasChildren()) return;

  float temperature = params_.GetTemperature();
  const int cutoff_move = params_.GetTemperatureCutoffMove();
  const int moves = played_history_.Last().GetGamePly() / 2;
  if (cutoff_move && (moves + 1) >= cutoff_move) {
    temperature = params_.GetTemperatureEndgame();
  } else if (temperature && params_.GetTempDecayMoves()) {
    if (moves >= params_.GetTempDecayMoves()) {
      temperature = 0.0;
    } else {
      temperature *= static_cast<float>(params_.GetTempDecayMoves() - moves) /
                     params_.GetTempDecayMoves();
    }
  }

  final_bestmove_ = temperature
                        ? GetBestChildWithTemperature(root_node_, temperature)
                        : GetBestChildNoTemperature(root_node_);

  if (final_bestmove_.HasNode() && final_bestmove_.node()->HasChildren()) {
    final_pondermove_ = GetBestChildNoTemperature(final_bestmove_.node());
  }
}

// Returns @count children with most visits.
std::vector<EdgeAndNode> Search::GetBestChildrenNoTemperature(Node* parent,
                                                              int count) const {
  MoveList root_limit;
  if (parent == root_node_) {
    PopulateRootMoveLimit(&root_limit);
  }
  // Best child is selected using the following criteria:
  // * Largest number of playouts.
  // * If two nodes have equal number:
  //   * If that number is 0, the one with larger prior wins.
  //   * If that number is larger than 0, the one with larger eval wins.
  using El = std::tuple<uint64_t, float, float, EdgeAndNode>;
  std::vector<El> edges;
  for (auto edge : parent->Edges()) {
    if (parent == root_node_ && !root_limit.empty() &&
        std::find(root_limit.begin(), root_limit.end(), edge.GetMove()) ==
            root_limit.end()) {
      continue;
    }
    edges.emplace_back(edge.GetN(), edge.GetQ(0), edge.GetP(), edge);
  }
  const auto middle = (static_cast<int>(edges.size()) > count)
                          ? edges.begin() + count
                          : edges.end();
  std::partial_sort(edges.begin(), middle, edges.end(), std::greater<El>());

  std::vector<EdgeAndNode> res;
  std::transform(edges.begin(), middle, std::back_inserter(res),
                 [](const El& x) { return std::get<3>(x); });
  return res;
}

// Returns a child with most visits.
EdgeAndNode Search::GetBestChildNoTemperature(Node* parent) const {
  auto res = GetBestChildrenNoTemperature(parent, 1);
  return res.empty() ? EdgeAndNode() : res.front();
}

// Returns a child chosen according to weighted-by-temperature visit count.
EdgeAndNode Search::GetBestChildWithTemperature(Node* parent,
                                                float temperature) const {
  MoveList root_limit;
  if (parent == root_node_) {
    PopulateRootMoveLimit(&root_limit);
  }

  std::vector<float> cumulative_sums;
  float sum = 0.0;
  float max_n = 0.0;
  const float offset = params_.GetTemperatureVisitOffset();
  float max_eval = -1.0f;
  const float fpu = GetFpu(params_, parent, parent == root_node_);

  for (auto edge : parent->Edges()) {
    if (parent == root_node_ && !root_limit.empty() &&
        std::find(root_limit.begin(), root_limit.end(), edge.GetMove()) ==
            root_limit.end()) {
      continue;
    }
    if (edge.GetN() + offset > max_n) {
      max_n = edge.GetN() + offset;
      max_eval = edge.GetQ(fpu);
    }
  }

  // No move had enough visits for temperature, so use default child criteria
  if (max_n <= 0.0f) return GetBestChildNoTemperature(parent);

  // TODO(crem) Simplify this code when samplers.h is merged.
  const float min_eval =
      max_eval - params_.GetTemperatureWinpctCutoff() / 50.0f;
  for (auto edge : parent->Edges()) {
    if (parent == root_node_ && !root_limit.empty() &&
        std::find(root_limit.begin(), root_limit.end(), edge.GetMove()) ==
            root_limit.end()) {
      continue;
    }
    if (edge.GetQ(fpu) < min_eval) continue;
    sum += std::pow(
        std::max(0.0f, (static_cast<float>(edge.GetN()) + offset) / max_n),
        1 / temperature);
    cumulative_sums.push_back(sum);
  }
  assert(sum);

  const float toss = Random::Get().GetFloat(cumulative_sums.back());
  int idx =
      std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) -
      cumulative_sums.begin();

  for (auto edge : parent->Edges()) {
    if (parent == root_node_ && !root_limit.empty() &&
        std::find(root_limit.begin(), root_limit.end(), edge.GetMove()) ==
            root_limit.end()) {
      continue;
    }
    if (edge.GetQ(fpu) < min_eval) continue;
    if (idx-- == 0) return edge;
  }
  assert(false);
  return {};
}

void Search::StartThreads(size_t how_many) {
  Mutex::Lock lock(threads_mutex_);
  // First thread is a watchdog thread.
  if (threads_.size() == 0) {
    threads_.emplace_back([this]() { WatchdogThread(); });
  }
  // Start working threads.
  while (threads_.size() <= how_many) {
    threads_.emplace_back([this]() {
      SearchWorker worker(this, params_);
      worker.RunBlocking();
    });
  }
}

void Search::RunBlocking(size_t threads) {
  StartThreads(threads);
  Wait();
}

bool Search::IsSearchActive() const {
  return !stop_.load(std::memory_order_acquire);
}

void Search::WatchdogThread() {
  LOGFILE << "Start a watchdog thread.";
  while (true) {
    MaybeTriggerStop();
    MaybeOutputInfo();

    using namespace std::chrono_literals;
    constexpr auto kMaxWaitTime = std::chrono::milliseconds(100);
    constexpr auto kMinWaitTime = std::chrono::milliseconds(1);

    Mutex::Lock lock(counters_mutex_);
    // Only exit when bestmove is responded. It may happen that search threads
    // already all exited, and we need at least one thread that can do that.
    if (bestmove_is_sent_) break;

    auto remaining_time = limits_.search_deadline
                              ? std::chrono::milliseconds(GetTimeToDeadline())
                              : kMaxWaitTime;
    if (remaining_time > kMaxWaitTime) remaining_time = kMaxWaitTime;
    if (remaining_time < kMinWaitTime) remaining_time = kMinWaitTime;
    // There is no real need to have max wait time, and sometimes it's fine
    // to wait without timeout at all (e.g. in `go nodes` mode), but we
    // still limit wait time for exotic cases like when pc goes to sleep
    // mode during thinking.
    // Minimum wait time is there to prevent busy wait and other threads
    // starvation.
    watchdog_cv_.wait_for(lock.get_raw(), remaining_time, [this]() {
      return stop_.load(std::memory_order_acquire);
    });
  }
  LOGFILE << "End a watchdog thread.";
}

void Search::FireStopInternal() {
  stop_.store(true, std::memory_order_release);
  watchdog_cv_.notify_all();
}

void Search::Stop() {
  Mutex::Lock lock(counters_mutex_);
  ok_to_respond_bestmove_ = true;
  FireStopInternal();
  LOGFILE << "Stopping search due to `stop` uci command.";
}

void Search::Abort() {
  Mutex::Lock lock(counters_mutex_);
  if (!stop_.load(std::memory_order_acquire) ||
      (!bestmove_is_sent_ && !ok_to_respond_bestmove_)) {
    bestmove_is_sent_ = true;
    FireStopInternal();
  }
  LOGFILE << "Aborting search, if it is still active.";
}

void Search::Wait() {
  Mutex::Lock lock(threads_mutex_);
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
}

Search::~Search() {
  Abort();
  Wait();
  LOGFILE << "Search destroyed.";
}

//////////////////////////////////////////////////////////////////////////////
// SearchWorker
//////////////////////////////////////////////////////////////////////////////

void SearchWorker::ExecuteOneIteration() {
  // 1. Initialize internal structures.
  InitializeIteration(search_->network_->NewComputation());

  // 2. Gather minibatch.
  GatherMinibatch();

  // 3. Prefetch into cache.
  MaybePrefetchIntoCache();

  // 4. Run NN computation.
  RunNNComputation();

  // 5. Retrieve NN computations (and terminal values) into nodes.
  FetchMinibatchResults();

  // 6. Propagate the new nodes' information to all their parents in the tree.
  DoBackupUpdate();

  // 7. Update the Search's status and progress information.
  UpdateCounters();
}

// 1. Initialize internal structures.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::InitializeIteration(
    std::unique_ptr<NetworkComputation> computation) {
  computation_ = std::make_unique<CachingComputation>(std::move(computation),
                                                      search_->cache_);
  minibatch_.clear();

  if (!root_move_filter_populated_) {
    root_move_filter_populated_ = true;
    if (search_->PopulateRootMoveLimit(&root_move_filter_)) {
      search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
    }
  }
}

// 2. Gather minibatch.
// ~~~~~~~~~~~~~~~~~~~~
void SearchWorker::GatherMinibatch() {
  // Total number of nodes to process.
  int minibatch_size = 0;
  int collision_events_left = params_.GetMaxCollisionEvents();
  int collisions_left = params_.GetMaxCollisionVisitsId();

  // Number of nodes processed out of order.
  number_out_of_order_ = 0;

  // Gather nodes to process in the current batch.
  // If we had too many (kMiniBatchSize) nodes out of order, also interrupt the
  // iteration so that search can exit.
  while (minibatch_size < params_.GetMiniBatchSize() &&
         number_out_of_order_ < params_.GetMiniBatchSize()) {
    // If there's something to process without touching slow neural net, do it.
    if (minibatch_size > 0 && computation_->GetCacheMisses() == 0) return;
    // Pick next node to extend.
    minibatch_.emplace_back(PickNodeToExtend(collisions_left));
    auto& picked_node = minibatch_.back();
    auto* node = picked_node.node;

    // There was a collision. If limit has been reached, return, otherwise
    // just start search of another node.
    if (picked_node.IsCollision()) {
      if (--collision_events_left <= 0) return;
      if ((collisions_left -= picked_node.multivisit) <= 0) return;
      if (search_->stop_.load(std::memory_order_acquire)) return;
      continue;
    }
    ++minibatch_size;

    // If node is already known as terminal (win/loss/draw according to rules
    // of the game), it means that we already visited this node before.
    if (picked_node.IsExtendable()) {
      // Node was never visited, extend it.
      ExtendNode(node);

      // Only send non-terminal nodes to a neural network.
      if (!node->IsTerminal()) {
        picked_node.nn_queried = true;
        picked_node.is_cache_hit = AddNodeToComputation(node, true);
      }
    }

    // If out of order eval is enabled and the node to compute we added last
    // doesn't require NN eval (i.e. it's a cache hit or terminal node), do
    // out of order eval for it.
    if (params_.GetOutOfOrderEval() && picked_node.CanEvalOutOfOrder()) {
      // Perform out of order eval for the last entry in minibatch_.
      FetchSingleNodeResult(&picked_node, computation_->GetBatchSize() - 1);
      {
        // Nodes mutex for doing node updates.
        SharedMutex::Lock lock(search_->nodes_mutex_);
        DoBackupUpdateSingleNode(picked_node);
      }

      // Remove last entry in minibatch_, as it has just been
      // processed.
      // If NN eval was already processed out of order, remove it.
      if (picked_node.nn_queried) computation_->PopCacheHit();
      minibatch_.pop_back();
      --minibatch_size;
      ++number_out_of_order_;
    }
    // Check for stop at the end so we have at least one node.
    if (search_->stop_.load(std::memory_order_acquire)) return;
  }
}

namespace {
void IncrementNInFlight(Node* node, Node* root, int amount) {
  if (amount == 0) return;
  while (true) {
    node->IncrementNInFlight(amount);
    if (node == root) break;
    node = node->GetParent();
  }
}
}  // namespace

// Returns node and whether there's been a search collision on the node.
SearchWorker::NodeToProcess SearchWorker::PickNodeToExtend(
    int collision_limit) {
  // Starting from search_->root_node_, generate a playout, choosing a
  // node at each level according to the MCTS formula. n_in_flight_ is
  // incremented for each node in the playout (via TryStartScoreUpdate()).

  Node* node = search_->root_node_;
  Node::Iterator best_edge;
  Node::Iterator second_best_edge;

  // Precache a newly constructed node to avoid memory allocations being
  // performed while the mutex is held.
  if (!precached_node_) {
    precached_node_ = std::make_unique<Node>(nullptr, 0);
  }

  SharedMutex::Lock lock(search_->nodes_mutex_);

  // Fetch the current best root node visits for possible smart pruning.
  const int64_t best_node_n = search_->current_best_edge_.GetN();

  // True on first iteration, false as we dive deeper.
  bool is_root_node = true;
  uint16_t depth = 0;
  bool node_already_updated = true;

  while (true) {
    // First, terminate if we find collisions or leaf nodes.
    // Set 'node' to point to the node that was picked on previous iteration,
    // possibly spawning it.
    // TODO(crem) This statement has to be in the end of the loop rather than
    //            in the beginning (and there would be no need for "if
    //            (!is_root_node)"), but that would mean extra mutex lock.
    //            Will revisit that after rethinking locking strategy.
    if (!node_already_updated) {
      node = best_edge.GetOrSpawnNode(/* parent */ node, &precached_node_);
    }
    best_edge.Reset();
    depth++;
    // n_in_flight_ is incremented. If the method returns false, then there is
    // a search collision, and this node is already being expanded.
    if (!node->TryStartScoreUpdate()) {
      if (!is_root_node) {
        IncrementNInFlight(node->GetParent(), search_->root_node_,
                           collision_limit - 1);
      }
      return NodeToProcess::Collision(node, depth, collision_limit);
    }
    // Either terminal or unexamined leaf node -- the end of this playout.
    if (node->IsTerminal() || !node->HasChildren()) {
      return NodeToProcess::Visit(node, depth);
    }
    Node* possible_shortcut_child = node->GetCachedBestChild();
    if (possible_shortcut_child) {
      // Add two here to reverse the conservatism that goes into calculating the
      // remaining cache visits.
      collision_limit =
          std::min(collision_limit, node->GetRemainingCacheVisits() + 2);
      is_root_node = false;
      node = possible_shortcut_child;
      node_already_updated = true;
      continue;
    }
    node_already_updated = false;

    // If we fall through, then n_in_flight_ has been incremented but this
    // playout remains incomplete; we must go deeper.
    const float cpuct = ComputeCpuct(params_, node->GetN());
    const float puct_mult =
        cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
    float best = std::numeric_limits<float>::lowest();
    float second_best = std::numeric_limits<float>::lowest();
    int possible_moves = 0;
    const float fpu = GetFpu(params_, node, is_root_node);
    for (auto child : node->Edges()) {
      if (is_root_node) {
        // If there's no chance to catch up to the current best node with
        // remaining playouts, don't consider it.
        // best_move_node_ could have changed since best_node_n was retrieved.
        // To ensure we have at least one node to expand, always include
        // current best node.
        if (child != search_->current_best_edge_ &&
            search_->remaining_playouts_ < best_node_n - child.GetN()) {
          continue;
        }
        // If root move filter exists, make sure move is in the list.
        if (!root_move_filter_.empty() &&
            std::find(root_move_filter_.begin(), root_move_filter_.end(),
                      child.GetMove()) == root_move_filter_.end()) {
          continue;
        }
        ++possible_moves;
      }
      const float Q = child.GetQ(fpu);
      const float score = child.GetU(puct_mult) + Q;
      if (score > best) {
        second_best = best;
        second_best_edge = best_edge;
        best = score;
        best_edge = child;
      } else if (score > second_best) {
        second_best = score;
        second_best_edge = child;
      }
    }

    if (second_best_edge) {
      int estimated_visits_to_change_best =
          best_edge.GetVisitsToReachU(second_best, puct_mult, fpu);
      // Only cache for n-2 steps as the estimate created by GetVisitsToReachU
      // has potential rounding errors and some conservative logic that can push
      // it up to 2 away from the real value.
      node->UpdateBestChild(best_edge,
                            std::max(0, estimated_visits_to_change_best - 2));
      collision_limit =
          std::min(collision_limit, estimated_visits_to_change_best);
      assert(collision_limit >= 1);
      second_best_edge.Reset();
    }

    if (is_root_node && possible_moves <= 1 && !search_->limits_.infinite) {
      // If there is only one move theoretically possible within remaining time,
      // output it.
      Mutex::Lock counters_lock(search_->counters_mutex_);
      search_->only_one_possible_move_left_ = true;
    }
    is_root_node = false;
  }
}

void SearchWorker::ExtendNode(Node* node) {
  // Initialize position sequence with pre-move position.
  history_.Trim(search_->played_history_.GetLength());
  std::vector<Move> to_add;
  // Could instead reserve one more than the difference between history_.size()
  // and history_.capacity().
  to_add.reserve(60);
  Node* cur = node;
  while (cur != search_->root_node_) {
    Node* prev = cur->GetParent();
    to_add.push_back(prev->GetEdgeToNode(cur)->GetMove());
    cur = prev;
  }
  for (int i = to_add.size() - 1; i >= 0; i--) {
    history_.Append(to_add[i]);
  }

  // We don't need the mutex because other threads will see that N=0 and
  // N-in-flight=1 and will not touch this node.
  const auto& board = history_.Last().GetBoard();
  auto legal_moves = board.GenerateLegalMoves();

  // Check whether it's a draw/lose by position. Importantly, we must check
  // these before doing the by-rule checks below.
  if (legal_moves.empty()) {
    // Could be a checkmate or a stalemate
    if (board.IsUnderCheck()) {
      node->MakeTerminal(GameResult::WHITE_WON);
    } else {
      node->MakeTerminal(GameResult::DRAW);
    }
    return;
  }

  // We can shortcircuit these draws-by-rule only if they aren't root;
  // if they are root, then thinking about them is the point.
  if (node != search_->root_node_) {
    if (!board.HasMatingMaterial()) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    if (history_.Last().GetNoCaptureNoPawnPly() >= 100) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    if (history_.Last().GetRepetitions() >= 2) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    // Neither by-position or by-rule termination, but maybe it's a TB position.
    if (search_->syzygy_tb_ && board.castlings().no_legal_castle() &&
        history_.Last().GetNoCaptureNoPawnPly() == 0 &&
        (board.ours() | board.theirs()).count() <=
            search_->syzygy_tb_->max_cardinality()) {
      ProbeState state;
      const WDLScore wdl =
          search_->syzygy_tb_->probe_wdl(history_.Last(), &state);
      // Only fail state means the WDL is wrong, probe_wdl may produce correct
      // result with a stat other than OK.
      if (state != FAIL) {
        // If the colors seem backwards, check the checkmate check above.
        if (wdl == WDL_WIN) {
          node->MakeTerminal(GameResult::BLACK_WON);
        } else if (wdl == WDL_LOSS) {
          node->MakeTerminal(GameResult::WHITE_WON);
        } else {  // Cursed wins and blessed losses count as draws.
          node->MakeTerminal(GameResult::DRAW);
        }
        search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
        return;
      }
    }
  }

  // Add legal moves as edges of this node.
  node->CreateEdges(legal_moves);
}

// Returns whether node was already in cache.
bool SearchWorker::AddNodeToComputation(Node* node, bool add_if_cached) {
  const auto hash = history_.HashLast(params_.GetCacheHistoryLength() + 1);
  // If already in cache, no need to do anything.
  if (add_if_cached) {
    if (computation_->AddInputByHash(hash)) return true;
  } else {
    if (search_->cache_->ContainsKey(hash)) return true;
  }
  auto planes = EncodePositionForNN(history_, 8, params_.GetHistoryFill());

  std::vector<uint16_t> moves;

  if (node && node->HasChildren()) {
    // Legal moves are known, use them.
    moves.reserve(node->GetNumEdges());
    for (const auto& edge : node->Edges()) {
      moves.emplace_back(edge.GetMove().as_nn_index());
    }
  } else {
    // Cache pseudolegal moves. A bit of a waste, but faster.
    const auto& pseudolegal_moves =
        history_.Last().GetBoard().GeneratePseudolegalMoves();
    moves.reserve(pseudolegal_moves.size());
    for (auto iter = pseudolegal_moves.begin(), end = pseudolegal_moves.end();
         iter != end; ++iter) {
      moves.emplace_back(iter->as_nn_index());
    }
  }

  computation_->AddInput(hash, std::move(planes), std::move(moves));
  return false;
}

// 3. Prefetch into cache.
// ~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::MaybePrefetchIntoCache() {
  // TODO(mooskagh) Remove prefetch into cache if node collisions work well.
  // If there are requests to NN, but the batch is not full, try to prefetch
  // nodes which are likely useful in future.
  if (search_->stop_.load(std::memory_order_acquire)) return;
  if (computation_->GetCacheMisses() > 0 &&
      computation_->GetCacheMisses() < params_.GetMaxPrefetchBatch()) {
    history_.Trim(search_->played_history_.GetLength());
    SharedMutex::SharedLock lock(search_->nodes_mutex_);
    PrefetchIntoCache(search_->root_node_, params_.GetMaxPrefetchBatch() -
                                               computation_->GetCacheMisses());
  }
}

// Prefetches up to @budget nodes into cache. Returns number of nodes
// prefetched.
int SearchWorker::PrefetchIntoCache(Node* node, int budget) {
  if (budget <= 0) return 0;

  // We are in a leaf, which is not yet being processed.
  if (!node || node->GetNStarted() == 0) {
    if (AddNodeToComputation(node, false)) {
      // Make it return 0 to make it not use the slot, so that the function
      // tries hard to find something to cache even among unpopular moves.
      // In practice that slows things down a lot though, as it's not always
      // easy to find what to cache.
      return 1;
    }
    return 1;
  }

  assert(node);
  // n = 0 and n_in_flight_ > 0, that means the node is being extended.
  if (node->GetN() == 0) return 0;
  // The node is terminal; don't prefetch it.
  if (node->IsTerminal()) return 0;

  // Populate all subnodes and their scores.
  typedef std::pair<float, EdgeAndNode> ScoredEdge;
  std::vector<ScoredEdge> scores;
  const float cpuct = ComputeCpuct(params_, node->GetN());
  const float puct_mult =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  const float fpu = GetFpu(params_, node, node == search_->root_node_);
  for (auto edge : node->Edges()) {
    if (edge.GetP() == 0.0f) continue;
    // Flip the sign of a score to be able to easily sort.
    scores.emplace_back(-edge.GetU(puct_mult) - edge.GetQ(fpu), edge);
  }

  size_t first_unsorted_index = 0;
  int total_budget_spent = 0;
  int budget_to_spend = budget;  // Initialize for the case where there's only
                                 // one child.
  for (size_t i = 0; i < scores.size(); ++i) {
    if (search_->stop_.load(std::memory_order_acquire)) break;
    if (budget <= 0) break;

    // Sort next chunk of a vector. 3 at a time. Most of the time it's fine.
    if (first_unsorted_index != scores.size() &&
        i + 2 >= first_unsorted_index) {
      const int new_unsorted_index =
          std::min(scores.size(), budget < 2 ? first_unsorted_index + 2
                                             : first_unsorted_index + 3);
      std::partial_sort(scores.begin() + first_unsorted_index,
                        scores.begin() + new_unsorted_index, scores.end(),
                        [](const ScoredEdge& a, const ScoredEdge& b) {
                          return a.first < b.first;
                        });
      first_unsorted_index = new_unsorted_index;
    }

    auto edge = scores[i].second;
    // Last node gets the same budget as prev-to-last node.
    if (i != scores.size() - 1) {
      // Sign of the score was flipped for sorting, so flip it back.
      const float next_score = -scores[i + 1].first;
      const float q = edge.GetQ(-fpu);
      if (next_score > q) {
        budget_to_spend =
            std::min(budget, int(edge.GetP() * puct_mult / (next_score - q) -
                                 edge.GetNStarted()) +
                                 1);
      } else {
        budget_to_spend = budget;
      }
    }
    history_.Append(edge.GetMove());
    const int budget_spent = PrefetchIntoCache(edge.node(), budget_to_spend);
    history_.Pop();
    budget -= budget_spent;
    total_budget_spent += budget_spent;
  }
  return total_budget_spent;
}

// 4. Run NN computation.
// ~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::RunNNComputation() { computation_->ComputeBlocking(); }

// 5. Retrieve NN computations (and terminal values) into nodes.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::FetchMinibatchResults() {
  // Populate NN/cached results, or terminal results, into nodes.
  int idx_in_computation = 0;
  for (auto& node_to_process : minibatch_) {
    FetchSingleNodeResult(&node_to_process, idx_in_computation);
    if (node_to_process.nn_queried) ++idx_in_computation;
  }
}

void SearchWorker::FetchSingleNodeResult(NodeToProcess* node_to_process,
                                         int idx_in_computation) {
  Node* node = node_to_process->node;
  if (!node_to_process->nn_queried) {
    // Terminal nodes don't involve the neural NetworkComputation, nor do
    // they require any further processing after value retrieval.
    node_to_process->v = node->GetQ();
    node_to_process->d = node->GetD();
    return;
  }
  // For NN results, we need to populate policy as well as value.
  // First the value...
  node_to_process->v = -computation_->GetQVal(idx_in_computation);
  node_to_process->d = computation_->GetDVal(idx_in_computation);
  // ...and secondly, the policy data.
  // Calculate maximum first.
  float max_p = -std::numeric_limits<float>::infinity();
  for (auto edge : node->Edges()) {
    max_p =
        std::max(max_p, computation_->GetPVal(idx_in_computation,
                                              edge.GetMove().as_nn_index()));
  }
  float total = 0.0;
  for (auto edge : node->Edges()) {
    float p =
        computation_->GetPVal(idx_in_computation, edge.GetMove().as_nn_index());
    // Perform softmax and take into account policy softmax temperature T.
    // Note that we want to calculate (exp(p-max_p))^(1/T) = exp((p-max_p)/T).
    p = FastExp((p - max_p) / params_.GetPolicySoftmaxTemp());

    // Note that p now lies in [0, 1], so it is safe to store it in compressed
    // format. Normalization happens later.
    edge.edge()->SetP(p);
    // Edge::SetP does some rounding, so only add to the total after rounding.
    total += edge.edge()->GetP();
  }
  // Normalize P values to add up to 1.0.
  if (total > 0.0f) {
    const float scale = 1.0f / total;
    for (auto edge : node->Edges()) edge.edge()->SetP(edge.GetP() * scale);
  }
  // Add Dirichlet noise if enabled and at root.
  if (params_.GetNoise() && node == search_->root_node_) {
    ApplyDirichletNoise(node, 0.25, 0.3);
  }
}

// 6. Propagate the new nodes' information to all their parents in the tree.
// ~~~~~~~~~~~~~~
void SearchWorker::DoBackupUpdate() {
  // Nodes mutex for doing node updates.
  SharedMutex::Lock lock(search_->nodes_mutex_);

  for (const NodeToProcess& node_to_process : minibatch_) {
    DoBackupUpdateSingleNode(node_to_process);
  }
}

void SearchWorker::DoBackupUpdateSingleNode(
    const NodeToProcess& node_to_process) REQUIRES(search_->nodes_mutex_) {
  Node* node = node_to_process.node;
  if (node_to_process.IsCollision()) {
    // If it was a collision, just undo counters.
    for (node = node->GetParent(); node != search_->root_node_->GetParent();
         node = node->GetParent()) {
      node->CancelScoreUpdate(node_to_process.multivisit);
    }
    return;
  }

  // For the first visit to a terminal, maybe convert ancestors to terminal too.
  auto can_convert =
      params_.GetStickyEndgames() && node->IsTerminal() && !node->GetN();

  // Backup V value up to a root. After 1 visit, V = Q.
  float v = node_to_process.v;
  float d = node_to_process.d;
  for (Node *n = node, *p; n != search_->root_node_->GetParent(); n = p) {
    p = n->GetParent();

    // Current node might have become terminal from some other descendant, so
    // backup the rest of the way with more accurate values.
    if (n->IsTerminal()) {
      v = n->GetQ();
      d = n->GetD();
    }
    n->FinalizeScoreUpdate(v, d, node_to_process.multivisit);

    // Nothing left to do without ancestors to update.
    if (!p) break;

    // Convert parents to terminals except the root or those already converted.
    can_convert = can_convert && p != search_->root_node_ && !p->IsTerminal();

    // A non-winning terminal move needs all other moves to have the same value.
    if (can_convert && v != 1.0f) {
      for (const auto& edge : p->Edges()) {
        can_convert = can_convert && edge.IsTerminal() && edge.GetQ(0.0f) == v;
      }
    }

    // Convert the parent to a terminal loss if at least one move is winning or
    // to a terminal win or draw if all moves are loss or draw respectively.
    if (can_convert) {
      p->MakeTerminal(v == 1.0f ? GameResult::BLACK_WON
                                : v == -1.0f ? GameResult::WHITE_WON
                                             : GameResult::DRAW);
    }

    // Q will be flipped for opponent.
    v = -v;

    // Update the stats.
    // Best move.
    if (p == search_->root_node_ &&
        search_->current_best_edge_.GetN() <= n->GetN()) {
      search_->current_best_edge_ =
          search_->GetBestChildNoTemperature(search_->root_node_);
    }
  }
  search_->total_playouts_ += node_to_process.multivisit;
  search_->cum_depth_ += node_to_process.depth * node_to_process.multivisit;
  search_->max_depth_ = std::max(search_->max_depth_, node_to_process.depth);
}  // namespace lczero

// 7. Update the Search's status and progress information.
//~~~~~~~~~~~~~~~~~~~~
void SearchWorker::UpdateCounters() {
  search_->UpdateRemainingMoves();  // Updates smart pruning counters.
  search_->UpdateKLDGain();
  search_->MaybeTriggerStop();
  search_->MaybeOutputInfo();

  // If this thread had no work, not even out of order, then sleep for some
  // milliseconds. Collisions don't count as work, so have to enumerate to find
  // out if there was anything done.
  bool work_done = number_out_of_order_ > 0;
  if (!work_done) {
    for (NodeToProcess& node_to_process : minibatch_) {
      if (!node_to_process.IsCollision()) {
        work_done = true;
        break;
      }
    }
  }
  if (!work_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace lczero

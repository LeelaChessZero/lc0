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

#include "mcts/search.h"

#include <algorithm>
#include <bitset>
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
#include "utils/random.h"

namespace lczero {

namespace {
const int kSmartPruningToleranceNodes = 300;
const int kSmartPruningToleranceMs = 200;
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 2500;
}  // namespace

Search::Search(const NodeTree& tree, Network* network,
               BestMoveInfo::Callback best_move_callback,
               ThinkingInfo::Callback info_callback, const SearchLimits& limits,
               const OptionsDict& options, NNCache* cache,
               SyzygyTablebase* syzygy_tb)
    : root_node_(tree.GetCurrentHead()),
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
  if (!best_move_edge_) return;

  auto edges = GetBestChildrenNoTemperature(root_node_, params_.GetMultiPv());

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
  common_info.certain = certain_.load(std::memory_order_acquire);
  common_info.bounds = bounds_.load(std::memory_order_acquire);

  int multipv = 0;
  for (const auto& edge : edges) {
    ++multipv;
    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();
    uci_info.score = 290.680623072 * tan(1.548090806 * edge.GetQ(0));
    if (params_.GetMultiPv() > 1) uci_info.multipv = multipv;
    bool flip = played_history_.IsBlackToMove();
    for (auto iter = edge; iter;
         iter = GetBestChildNoTemperature(iter.node()), flip = !flip) {
      uci_info.pv.push_back(iter.GetMove(flip));
      if (!iter.node()) break;  // Last edge was dangling, cannot continue.
    }
    // For mate display offset certain scores by 20000 + length of pv (average
    // mate). Length of mate is +1000 if winning certain score is based on a
    // propagated TBHit. For root filtered TB search, use best_rank (mate +500)
    // TODO: include proper mate scores in uci_info structure
    if (params_.GetCertaintyPropagation() > 0) {
      if (edge.IsCertain() && edge.GetEQ() != 0.0f)
        uci_info.score =
            edge.GetEQ() * (20000 + ((uci_info.pv.size() + 1) / 2) + 1 +
                            (edge.IsPropagatedTBHit() ? 1000 : 0));
      else if (root_syzygy_rank_) {
        int sign = (root_syzygy_rank_ - 1 > 0) - (root_syzygy_rank_ - 1 < 0);
        uci_info.score = sign * (19500 + abs(root_syzygy_rank_));
      }
    }
  }

  if (!uci_infos.empty()) last_outputted_uci_info_ = uci_infos.front();
  if (!edges.empty()) last_outputted_best_move_edge_ = best_move_edge_.edge();

  info_callback_(uci_infos);
}

// Decides whether anything important changed in stats and new info should be
// shown to a user.
void Search::MaybeOutputInfo() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  if (!responded_bestmove_ && best_move_edge_ &&
      (best_move_edge_.edge() != last_outputted_best_move_edge_ ||
       last_outputted_uci_info_.depth !=
           static_cast<int>(cum_depth_ /
                            (total_playouts_ ? total_playouts_ : 1)) ||
       last_outputted_uci_info_.seldepth != max_depth_ ||
       last_outputted_uci_info_.time + kUciInfoMinimumFrequencyMs <
           GetTimeSinceStart())) {
    SendUciInfo();
  }
}

int64_t Search::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

void Search::SendMovesStats() const {
  const float parent_q =
      -root_node_->GetQ() -
      params_.GetFpuReduction() * std::sqrt(root_node_->GetVisitedPolicy());
  const float U_coeff =
      params_.GetCpuct() *
      std::sqrt(std::max(root_node_->GetChildrenVisits(), 1u));

  std::vector<EdgeAndNode> edges;
  for (const auto& edge : root_node_->Edges()) edges.push_back(edge);

  std::sort(edges.begin(), edges.end(),
            [&parent_q, &U_coeff](EdgeAndNode a, EdgeAndNode b) {
              return std::forward_as_tuple(a.GetN(),
                                           a.GetQ(parent_q) + a.GetU(U_coeff)) <
                     std::forward_as_tuple(b.GetN(),
                                           b.GetQ(parent_q) + b.GetU(U_coeff));
            });

  const bool is_black_to_move = played_history_.IsBlackToMove();
  std::vector<ThinkingInfo> infos;

  // Root info
  infos.emplace_back();
  ThinkingInfo& info = infos.back();
  std::ostringstream oss;
  oss << std::fixed;
  oss << "Root         N: ";
  oss << std::right << std::setw(7) << root_node_->GetN() << " (+"
      << std::setw(3) << root_node_->GetNInFlight() << ") ";
  oss << "            (Q: " << std::setw(8) << std::setprecision(5)
      << -root_node_->GetQ() << ") ";
  if (root_node_->GetEdgeToMe())
    oss << " C:"
        << std::bitset<8>(root_node_->GetEdgeToMe()->GetCertaintyStatus());
  else
    oss << " C: No Edge";

  info.comment = oss.str();
  // Move Infos
  for (const auto& edge : edges) {
    infos.emplace_back();
    ThinkingInfo& info = infos.back();
    std::ostringstream oss;
    oss << std::fixed;

    oss << std::left << std::setw(5)
        << edge.GetMove(is_black_to_move).as_string();

    oss << " (" << std::setw(4) << edge.GetMove().as_nn_index() << ")";

    oss << " N: " << std::right << std::setw(7) << edge.GetN() << " (+"
        << std::setw(3) << edge.GetNInFlight() << ") ";

    oss << "(P: " << std::setw(5) << std::setprecision(2) << edge.GetP() * 100
        << "%) ";

    oss << "(Q: " << std::setw(8) << std::setprecision(5) << edge.GetQ(parent_q)
        << ") ";

    oss << "(U: " << std::setw(6) << std::setprecision(5) << edge.GetU(U_coeff)
        << ") ";

    oss << "(Q+U: " << std::setw(8) << std::setprecision(5)
        << edge.GetQ(parent_q) + edge.GetU(U_coeff) << ") ";

    oss << "(V: ";
    optional<float> v;
    if (edge.IsCertain()) {
      v = edge.edge()->GetEQ();
    } else {
      NNCacheLock nneval = GetCachedFirstPlyResult(edge);
      if (nneval) v = -nneval->q;
    }
    if (v) {
      oss << std::setw(7) << std::setprecision(4) << *v;
    } else {
      oss << " -.----";
    }
    oss << ") ";

    oss << " C:" << std::bitset<8>(edge.edge()->GetCertaintyStatus());

    info.comment = oss.str();
  }
  info_callback_(infos);
}

NNCacheLock Search::GetCachedFirstPlyResult(EdgeAndNode edge) const {
  if (!edge.HasNode()) return {};
  assert(edge.node()->GetParent() == root_node_);
  // It would be relatively straightforward to generalize this to fetch NN
  // results for an abitrary move.
  optional<float> retval;
  PositionHistory history(played_history_);  // Is it worth it to move this
  // initialization to SendMoveStats, reducing n memcpys to 1? Probably not.
  history.Append(edge.GetMove());
  auto hash = history.HashLast(params_.GetCacheHistoryLength() + 1);
  NNCacheLock nneval(cache_, hash);
  return nneval;
}

void Search::MaybeTriggerStop() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  // Already responded bestmove, nothing to do here.
  if (responded_bestmove_) return;
  // Don't stop when the root node is not yet expanded.
  if (total_playouts_ == 0) return;
  // If smart pruning tells to stop (best move found), stop.
  if (found_best_move_) {
    FireStopInternal();
  }
  // Stop if reached playouts limit.
  if (limits_.playouts >= 0 && total_playouts_ >= limits_.playouts) {
    FireStopInternal();
  }
  // Stop if reached visits limit.
  if (limits_.visits >= 0 &&
      total_playouts_ + initial_visits_ >= limits_.visits) {
    FireStopInternal();
  }
  // Stop if reached time limit.
  if (limits_.time_ms >= 0 && GetTimeSinceStart() >= limits_.time_ms) {
    FireStopInternal();
  }
  // Stop if average depth reached requested depth.
  if (limits_.depth >= 0 &&
      cum_depth_ / (total_playouts_ ? total_playouts_ : 1) >=
          static_cast<unsigned int>(limits_.depth)) {
    FireStopInternal();
  }
  // If we are the first to see that stop is needed.
  if (stop_ && !responded_bestmove_) {
    SendUciInfo();
    if (params_.GetVerboseStats()) SendMovesStats();
    best_move_ = GetBestMoveInternal();
    best_move_callback_({best_move_.first, best_move_.second});
    responded_bestmove_ = true;
    best_move_edge_ = EdgeAndNode();
  }
}

void Search::UpdateRemainingMoves() {
  if (params_.GetAggressiveTimePruning() <= 0.0f) return;
  SharedMutex::Lock lock(nodes_mutex_);
  remaining_playouts_ = std::numeric_limits<int>::max();
  // Check for how many playouts there is time remaining.
  if (limits_.time_ms >= 0) {
    auto time_since_start = GetTimeSinceStart();
    if (time_since_start > kSmartPruningToleranceMs * 2) {
      auto nps = 1000LL * (total_playouts_ + kSmartPruningToleranceNodes) /
                     (time_since_start - kSmartPruningToleranceMs) +
                 1;
      int64_t remaining_time = limits_.time_ms - time_since_start;
      // Put early_exit scaler here so calculation doesn't have to be done on
      // every node.
      int64_t remaining_playouts =
          remaining_time * nps / params_.GetAggressiveTimePruning() / 1000;
      // Don't assign directly to remaining_playouts_ as overflow is possible.
      if (remaining_playouts < remaining_playouts_)
        remaining_playouts_ = remaining_playouts;
    }
  }
  // Check how many visits are left.
  if (limits_.visits >= 0) {
    // Add kMiniBatchSize, as it's possible to exceed visits limit by that
    // number.
    auto remaining_visits = limits_.visits - total_playouts_ - initial_visits_ +
                            params_.GetMiniBatchSize() - 1;

    if (remaining_visits < remaining_playouts_)
      remaining_playouts_ = remaining_visits;
  }
  if (limits_.playouts >= 0) {
    // Add kMiniBatchSize, as it's possible to exceed visits limit by that
    // number.
    auto remaining_playouts =
        limits_.visits - total_playouts_ + params_.GetMiniBatchSize() + 1;
    if (remaining_playouts < remaining_playouts_)
      remaining_playouts_ = remaining_playouts;
  }
  // Even if we exceeded limits, don't go crazy by not allowing any playouts.
  if (remaining_playouts_ <= 1) remaining_playouts_ = 1;
}

// Return the evaluation of the actual best child, regardless of temperature
// settings. This differs from GetBestMove, which does obey any temperature
// settings. So, somethimes, they may return results of different moves.
float Search::GetBestEval() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  float parent_q = -root_node_->GetQ();
  if (!root_node_->HasChildren()) return parent_q;
  EdgeAndNode best_edge = GetBestChildNoTemperature(root_node_);
  return best_edge.GetQ(parent_q);
}

std::pair<Move, Move> Search::GetBestMove() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  return GetBestMoveInternal();
}

int Search::PopulateRootMoveLimit(MoveList* root_moves) const {
  // Search moves overrides tablebase.
  if (!limits_.searchmoves.empty()) {
    *root_moves = limits_.searchmoves;
    return 0;
  }

  // Syzygy root_probe returns best_rank for proper eval if
  // moves are syzygy root filtered.
  auto board = played_history_.Last().GetBoard();
  if (!syzygy_tb_ || !board.castlings().no_legal_castle() ||
      (board.ours() + board.theirs()).count() > syzygy_tb_->max_cardinality()) {
    return 0;
  }

  int best_rank = syzygy_tb_->root_probe(
      played_history_.Last(), played_history_.DidRepeatSinceLastZeroingMove(),
      root_moves);
  if (!best_rank)
    best_rank = syzygy_tb_->root_probe_wdl(played_history_.Last(), root_moves);
  return best_rank;
}

// Returns the best move, maybe with temperature (according to the settings).
std::pair<Move, Move> Search::GetBestMoveInternal() const
    REQUIRES_SHARED(nodes_mutex_) REQUIRES_SHARED(counters_mutex_) {
  if (responded_bestmove_) return best_move_;
  if (!root_node_->HasChildren()) return {};

  float temperature = params_.GetTemperature();
  if (temperature && params_.GetTempDecayMoves()) {
    int moves = played_history_.Last().GetGamePly() / 2;
    if (moves >= params_.GetTempDecayMoves()) {
      temperature = 0.0;
    } else {
      temperature *= static_cast<float>(params_.GetTempDecayMoves() - moves) /
                     params_.GetTempDecayMoves();
    }
  }

  auto best_node = temperature
                       ? GetBestChildWithTemperature(root_node_, temperature)
                       : GetBestChildNoTemperature(root_node_);

  Move ponder_move;  // Default is "null move" which means "don't display
                     // anything".
  if (best_node.HasNode() && best_node.node()->HasChildren()) {
    ponder_move = GetBestChildNoTemperature(best_node.node())
                      .GetMove(!played_history_.IsBlackToMove());
  }
  return {best_node.GetMove(played_history_.IsBlackToMove()), ponder_move};
}

// Returns @count children with most visits.
std::vector<EdgeAndNode> Search::GetBestChildrenNoTemperature(Node* parent,
                                                              int count) const {
  MoveList root_limit;
  if (parent == root_node_) {
    PopulateRootMoveLimit(&root_limit);
  }
  // Best child is selected using the following criteria:
  // with Certainty Propagation >= 2:
  // * Prefer certain wins, avoid certain losses
  // * certain draws are not searched and
  // * inserted by Q over N
  // Otherwise:
  // * Largest number of playouts.
  // * If two nodes have equal number:
  //   * If that number is 0, the one with larger prior wins.
  //   * If that number is larger than 0, the one with larger eval wins.
  using El = std::tuple<float, uint64_t, float, float, EdgeAndNode>;
  std::vector<El> edges;
  for (auto edge : parent->Edges()) {
    if (parent == root_node_ && !root_limit.empty() &&
        std::find(root_limit.begin(), root_limit.end(), edge.GetMove()) ==
            root_limit.end()) {
      continue;
    }
    edges.emplace_back(
        (params_.GetCertaintyPropagation() > 1 || parent != root_node_)
            ? edge.edge()->GetEQ()
            : 0.0f,
        edge.GetN(), edge.GetQ(0), edge.GetP(), edge);
  }

  // In case of certain draws (that are no longer searched CP>=2), insert
  // these draws at first N (descending) where Q<=0
  if (params_.GetCertaintyPropagation() >  1) {
    std::partial_sort(edges.begin(), edges.end(), edges.end(),
                      std::greater<El>());
    // largest N with Q >= 0
    uint64_t largest_N = 0;
    for (auto it = edges.begin(); it != edges.end(); ++it) {
      if (std::get<2>(*it) <= 0.0f && largest_N == 0)
        largest_N = std::get<1>(*it);
      if (std::get<4>(*it).edge()->IsCertainDraw() && largest_N > 0)
        std::get<1>(*it) = largest_N;
    }
  }
  // Final sort pass over adjusted certain draw Ns
  auto middle = (static_cast<int>(edges.size()) > count) ? edges.begin() + count
                                                         : edges.end();
  std::partial_sort(edges.begin(), middle, edges.end(), std::greater<El>());

  std::vector<EdgeAndNode> res;
  std::transform(edges.begin(), middle, std::back_inserter(res),
                 [](const El& x) { return std::get<4>(x); });
  return res;
}

// Returns best child.
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
  float offset = params_.GetTemperatureVisitOffset();

  for (auto edge : parent->Edges()) {
    if (parent == root_node_ && !root_limit.empty() &&
        std::find(root_limit.begin(), root_limit.end(), edge.GetMove()) ==
            root_limit.end()) {
      continue;
    }
    if (edge.GetN() + offset > max_n) {
      max_n = edge.GetN() + offset;
    }
  }

  // No move had enough visits for temperature, so use default child criteria
  if (max_n <= 0.0f) return GetBestChildNoTemperature(parent);

  for (auto edge : parent->Edges()) {
    if (parent == root_node_ && !root_limit.empty() &&
        std::find(root_limit.begin(), root_limit.end(), edge.GetMove()) ==
            root_limit.end()) {
      continue;
    }
    sum += std::pow(
        std::max(0.0f, (static_cast<float>(edge.GetN()) + offset) / max_n),
        1 / temperature);
    cumulative_sums.push_back(sum);
  }
  assert(sum);

  float toss = Random::Get().GetFloat(cumulative_sums.back());
  int idx =
      std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) -
      cumulative_sums.begin();

  for (auto edge : parent->Edges()) {
    if (parent == root_node_ && !root_limit.empty() &&
        std::find(root_limit.begin(), root_limit.end(), edge.GetMove()) ==
            root_limit.end()) {
      continue;
    }
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
  Mutex::Lock lock(counters_mutex_);
  return !stop_;
}

void Search::WatchdogThread() {
  LOGFILE << "Starting watchdog thread.";
  while (IsSearchActive()) {
    {
      using namespace std::chrono_literals;
      constexpr auto kMaxWaitTime = 100ms;
      constexpr auto kMinWaitTime = 1ms;
      Mutex::Lock lock(counters_mutex_);
      auto remaining_time = limits_.time_ms >= 0
                                ? (limits_.time_ms - GetTimeSinceStart()) * 1ms
                                : kMaxWaitTime;
      if (remaining_time > kMaxWaitTime) remaining_time = kMaxWaitTime;
      if (remaining_time < kMinWaitTime) remaining_time = kMinWaitTime;
      // There is no real need to have max wait time, and sometimes it's fine
      // to wait without timeout at all (e.g. in `go nodes` mode), but we
      // still limit wait time for exotic cases like when pc goes to sleep
      // mode during thinking.
      // Minimum wait time is there to prevent busy wait and other thread
      // starvation.
      watchdog_cv_.wait_for(lock.get_raw(), remaining_time,
                            [this]()
                                NO_THREAD_SAFETY_ANALYSIS { return stop_; });
    }
    MaybeTriggerStop();
  }
  MaybeTriggerStop();
}

void Search::FireStopInternal() REQUIRES(counters_mutex_) {
  stop_ = true;
  watchdog_cv_.notify_all();
  LOGFILE << "Stopping search.";
}

void Search::Stop() {
  Mutex::Lock lock(counters_mutex_);
  FireStopInternal();
}

void Search::Abort() {
  Mutex::Lock lock(counters_mutex_);
  responded_bestmove_ = true;
  FireStopInternal();
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
    int best_rank = search_->PopulateRootMoveLimit(&root_move_filter_);
    if (best_rank) {
      search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
      search_->root_syzygy_rank_ = best_rank;
    }
  }
}

// 2. Gather minibatch.
// ~~~~~~~~~~~~~~~~~~~~
void SearchWorker::GatherMinibatch() {
  // Total number of nodes to process.
  int minibatch_size = 0;
  int collision_events_left = params_.GetAllowedNodeCollisionEvents();
  int collisions_left = params_.GetAllowedTotalNodeCollisions();

  // Number of nodes processed out of order.
  int number_out_of_order = 0;

  // Gather nodes to process in the current batch.
  // If we had too many (kMiniBatchSize) nodes out of order, also interrupt the
  // iteration so that search can exit.
  // TODO(crem) change that to checking search_->stop_ when bestmove reporting
  // is in a separate thread.
  while (minibatch_size < params_.GetMiniBatchSize() &&
         number_out_of_order < params_.GetMiniBatchSize()) {
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
      continue;
    }
    ++minibatch_size;

    // If node is already known as terminal (win/loss/draw according to rules
    // of the game), it means that we already visited this node before.
    if (picked_node.IsExtendable()) {
      // Node was never visited, extend it.
      ExtendNode(node);

      // Only send non-terminal nodes to a neural network.
      if (!node->IsCertain()) {
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
      ++number_out_of_order;
    }
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
  Node::Iterator certain_draw_edge;
  // Initialize position sequence with pre-move position.
  history_.Trim(search_->played_history_.GetLength());

  SharedMutex::Lock lock(search_->nodes_mutex_);

  // Fetch the current best root node visits for possible smart pruning.
  int best_node_n = search_->best_move_edge_.GetN();

  // True on first iteration, false as we dive deeper.
  bool is_root_node = true;
  uint16_t depth = 0;

  while (true) {
    // First, terminate if we find collisions or leaf nodes.
    // Set 'node' to point to the node that was picked on previous iteration,
    // possibly spawning it.
    // TODO(crem) This statement has to be in the end of the loop rather than
    //            in the beginning (and there would be no need for "if
    //            (!is_root_node)"), but that would mean extra mutex lock.
    //            Will revisit that after rethinking locking strategy.
    if (!is_root_node) node = best_edge.GetOrSpawnNode(/* parent */ node);
    best_edge.Reset();
    depth++;
    // n_in_flight_ is incremented. If the method returns false, then there is
    // a search collision, and this node is already being expanded.
    if (!node->TryStartScoreUpdate()) {
      IncrementNInFlight(node, search_->root_node_, collision_limit - 1);
      return NodeToProcess::Collision(node, depth, collision_limit);
    }
    // Either terminal or unexamined leaf node -- the end of this playout.
    if (node->IsCertain()) {
      int multivisit =
          (params_.GetCertaintyPropagation() && params_.GetOutOfOrderEval())
              ? 1
              : collision_limit;
      IncrementNInFlight(node, search_->root_node_, multivisit - 1);
      return NodeToProcess::TerminalHit(node, depth, multivisit);
    } else if (!node->HasChildren()) {
      return NodeToProcess::Extension(node, depth);
    }

    // If we fall through, then n_in_flight_ has been incremented but this
    // playout remains incomplete; we must go deeper.
    float puct_mult =
        params_.GetCpuct() * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
    float best = std::numeric_limits<float>::lowest();
    float second_best = std::numeric_limits<float>::lowest();
    int possible_moves = 0;
    float parent_q =
        ((is_root_node && params_.GetNoise()) || !params_.GetFpuReduction())
            ? -node->GetQ()
            : -node->GetQ() - params_.GetFpuReduction() *
                                  std::sqrt(node->GetVisitedPolicy());
    for (auto child : node->Edges()) {
      if (is_root_node) {
        // If there's no chance to catch up to the current best node with
        // remaining playouts, don't consider it.
        // best_move_node_ could have changed since best_node_n was retrieved.
        // To ensure we have at least one node to expand, always include
        // current best node.
        if (child != search_->best_move_edge_ &&
            search_->remaining_playouts_ <
                best_node_n - static_cast<int>(child.GetN())) {
          continue;
        }
        // If CertaintyPropagation >= 2 play certain win and don't search other
        // moves at root. If search limit infinite continue searching other moves.
        if (params_.GetCertaintyPropagation() > 1 &&
            child.edge()->IsCertainWin()) {
          if (!search_->limits_.infinite) {
            best_edge = child;
            possible_moves = 1;
            break;
          } else if (search_->best_move_edge_ &&
                     search_->best_move_edge_.IsCertainWin())
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
      // Don't search certain (draw) branches (CP >= 2).
      // Certain wins we will not encounter more than once as
      // they are propagated to certain losses one ply up.
      // Certain losses we still visit because if policy hits them
      // the line might be bad already much earlier (signal).
      // TODO:
      // Check tighter bounds: if only Bounds(=,1) and (-1,=)
      // never visit (-1,=) -> needs enhanced PV selection rule
      if (params_.GetCertaintyPropagation() > 1 && node->GetNumEdges() > 1 && child.edge()->IsCertainDraw()) {
        certain_draw_edge = child;
        continue;
      }

      float Q = child.GetQ(parent_q);
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
      collision_limit = std::min(
          collision_limit,
          best_edge.GetVisitsToReachU(second_best, puct_mult, parent_q));
      assert(collision_limit >= 1);
      second_best_edge.Reset();
    }

    // SAFEGUARD: If all moves are certain draws. 
    // Should not happen as then certainty would have propagated upward.
    if (!best_edge && certain_draw_edge) best_edge = certain_draw_edge;

    history_.Append(best_edge.GetMove());
    if (is_root_node && possible_moves <= 1 && !search_->limits_.infinite) {
      // If there is only one move theoretically possible within remaining time,
      // output it.
      Mutex::Lock counters_lock(search_->counters_mutex_);
      search_->found_best_move_ = true;
    }
    is_root_node = false;
  }
}

void SearchWorker::EvalPosition(Node* node, MoveList& legal_moves,
                                const ChessBoard& board, GameResult& result,
                                CertaintyTrigger& trigger) {
  // Check whether it's a draw/lose by position. Importantly, we must check
  // these before doing the by-rule checks below.
  if (legal_moves.empty()) {
    // Could be a checkmate or a stalemate
    if (board.IsUnderCheck()) {
      result = GameResult::WHITE_WON;
      trigger = CertaintyTrigger::TERMINAL;
      // node->MakeTerminal(GameResult::WHITE_WON);
    } else {
      result = GameResult::DRAW;
      trigger = CertaintyTrigger::TERMINAL;
      // node->MakeTerminal(GameResult::DRAW);
    }
    return;
  }

  // We can shortcircuit these draws-by-rule only if they aren't root;
  // if they are root, then thinking about them is the point.
  if (node != search_->root_node_) {
    if (!board.HasMatingMaterial()) {
      result = GameResult::DRAW;
      trigger = CertaintyTrigger::TERMINAL;
      // node->MakeTerminal(GameResult::DRAW);
      return;
    }

    if (history_.Last().GetNoCaptureNoPawnPly() >= 100) {
      result = GameResult::DRAW;
      trigger = CertaintyTrigger::TERMINAL;
      // node->MakeTerminal(GameResult::DRAW);
      return;
    }

    if (history_.Last().GetRepetitions() >= 2) {
      result = GameResult::DRAW;
      trigger = CertaintyTrigger::TERMINAL;
      // node->MakeTerminal(GameResult::DRAW);
      return;
    }

    if ((history_.Last().GetRepetitions() >= 1) &&
        params_.GetCertaintyPropagation()) {
      result = GameResult::DRAW;
      trigger = CertaintyTrigger::TWO_FOLD;
      // node->MakeCertain(0.0f);
      return;
    }

    // Neither by-position or by-rule termination, but maybe it's a TB position.
    if (!search_->root_syzygy_rank_ && search_->syzygy_tb_ &&
        board.castlings().no_legal_castle() &&
        history_.Last().GetNoCaptureNoPawnPly() == 0 &&
        (board.ours() + board.theirs()).count() <=
            search_->syzygy_tb_->max_cardinality()) {
      ProbeState state;
      WDLScore wdl = search_->syzygy_tb_->probe_wdl(history_.Last(), &state);
      // Only fail state means the WDL is wrong, probe_wdl may produce correct
      // result with a stat other than OK.
      if (state != FAIL) {
        // If the colors seem backwards, check the checkmate check above.
        if (wdl == WDL_WIN) {
          result = GameResult::BLACK_WON;
          trigger = CertaintyTrigger::TB_HIT;
          // node->MakeTerminal(GameResult::BLACK_WON);
        } else if (wdl == WDL_LOSS) {
          result = GameResult::WHITE_WON;
          trigger = CertaintyTrigger::TB_HIT;
          // node->MakeTerminal(GameResult::WHITE_WON);
        } else {  // Cursed wins and blessed losses count as draws.
          result = GameResult::DRAW;
          trigger = CertaintyTrigger::NORMAL;
          // node->MakeTerminal(GameResult::DRAW);
        }
        search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
        return;
      }
    }
  }
}

void SearchWorker::ExtendNode(Node* node) {
  // We don't need the mutex because other threads will see that N=0 and
  // N-in-flight=1 and will not touch this node.
  const auto& board = history_.Last().GetBoard();
  auto legal_moves = board.GenerateLegalMoves();
  GameResult result = GameResult::UNDECIDED;
  CertaintyTrigger trigger = CertaintyTrigger::NONE;

  EvalPosition(node, legal_moves, board, result, trigger);

  if (trigger != CertaintyTrigger::NONE) {
    if (trigger == CertaintyTrigger::TERMINAL)
      node->MakeTerminal(result);
    else
      node->MakeCertain(result, trigger);
    return;
  }
  // Add legal moves as edges of this node.
  node->CreateEdges(legal_moves);

  // Certainty Propagation
  // Look-ahead-search to see whether these moves already
  // bound the edges or the node, or even make them certain.
  // TODO: Hash positions.
  // TODO: Move ordering.
  // TODO: Optimize for speed (move generator) to search
  // three plys deep to catch all Q+,K,Q+,K two-folds.
  if (params_.GetCertaintyPropagation()) {
    int node_lowerbound = -1;
    int node_upperbound = -1;
    bool based_on_propagated_tbhit = false;
    for (auto iter : node->Edges()) {
      // Eval each edge:
      // Append -> Search -> Pop
      history_.Append(iter.GetMove());
      // Search with depth, lower bound and upper bound.
      // Currently depth = 1, as move generator is slow.
      struct Bounds bounds = SearchWorker::NegaBoundSearch(0, -1, 1);
      history_.Pop();
      if (bounds.lowerbound == bounds.upperbound) {
        iter.edge()->MakeCertain(-bounds.lowerbound,
                                 bounds.based_on_tbhit
                                     ? CertaintyTrigger::TB_HIT
                                     : CertaintyTrigger::NORMAL);
        based_on_propagated_tbhit |= bounds.based_on_tbhit;
        search_->certain_.fetch_add(1, std::memory_order_acq_rel);
      } else {
        if (bounds.lowerbound > -1)
          iter.edge()->UBound((float)-bounds.lowerbound);
        if (bounds.upperbound < 1)
          iter.edge()->LBound((float)-bounds.upperbound);
        if (bounds.lowerbound > -1 || bounds.upperbound < 1)
          search_->bounds_.fetch_add(1, std::memory_order_acq_rel);
      }
      if (-bounds.upperbound > node_lowerbound)
        node_lowerbound = -bounds.upperbound;
      if (-bounds.lowerbound > node_upperbound)
        node_upperbound = -bounds.lowerbound;
    }
    if (node != search_->root_node_) {
      if (node_lowerbound == node_upperbound) {
        node->MakeCertain((float)-node_lowerbound,
                          based_on_propagated_tbhit ? CertaintyTrigger::TB_HIT
                                                    : CertaintyTrigger::NORMAL);
        search_->certain_.fetch_add(1, std::memory_order_acq_rel);
      } else {
        if (node_lowerbound > -1) node->UBound((float)-node_lowerbound);
        if (node_upperbound < 1) node->LBound((float)-node_upperbound);
        if (node_lowerbound > -1 || node_upperbound < 1)
          search_->bounds_.fetch_add(1, std::memory_order_acq_rel);
      }
    }
  }
}

struct SearchWorker::Bounds SearchWorker::NegaBoundSearch(int depth,
                                                          int lowerbound,
                                                          int upperbound) {
  struct Bounds returnbound;
  const auto& board = history_.Last().GetBoard();
  auto legal_moves_child = board.GenerateLegalMoves();
  GameResult result = GameResult::UNDECIDED;
  CertaintyTrigger trigger = CertaintyTrigger::NONE;
  EvalPosition(nullptr, legal_moves_child, board, result, trigger);

  if (trigger != CertaintyTrigger::NONE) {
    returnbound.based_on_tbhit |= (trigger == CertaintyTrigger::TB_HIT);
    int score = (result == GameResult::WHITE_WON)
                    ? -1
                    : (result == GameResult::BLACK_WON ? 1 : 0);
    returnbound.lowerbound = score;
    returnbound.upperbound = score;
    return returnbound;
  }
  // Singular-extend
  // positions with only one move.
  if ((depth == 0 && legal_moves_child.size() != 1) || depth < -1) {
    returnbound.lowerbound = -1;
    returnbound.upperbound = 1;
    return returnbound;
  }

  int myupperbound = -1;
  for (auto iter : legal_moves_child) {
    history_.Append(iter);
    // Call recursive  NegaBoundSearch with bounds reversed for opponent.
    struct Bounds bound = NegaBoundSearch(depth - 1, -upperbound, -lowerbound);
    int rlower = -bound.upperbound;
    int rupper = -bound.lowerbound;
    returnbound.based_on_tbhit |= bound.based_on_tbhit;
    history_.Pop();
    if (rlower >= lowerbound) lowerbound = rlower;
    if (rupper >= myupperbound) myupperbound = rupper;
    // Alpha-Bound cutoff.
    if (lowerbound >= upperbound) break;
  }
  returnbound.lowerbound = lowerbound;
  returnbound.upperbound = myupperbound;
  return returnbound;
}

// Returns whether node was already in cache.
bool SearchWorker::AddNodeToComputation(Node* node, bool add_if_cached) {
  auto hash = history_.HashLast(params_.GetCacheHistoryLength() + 1);
  // If already in cache, no need to do anything.
  if (add_if_cached) {
    if (computation_->AddInputByHash(hash)) return true;
  } else {
    if (search_->cache_->ContainsKey(hash)) return true;
  }
  auto planes = EncodePositionForNN(history_, 8);

  std::vector<uint16_t> moves;

  if (node && node->HasChildren()) {
    // Legal moves are known, use them.
    for (auto edge : node->Edges()) {
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
  // TODO(Videodr0me) Maybe use bounds here to more efficiently select nodes.
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
  // The node is certain; don't prefetch it.
  if (node->IsCertain()) return 0;

  // Populate all subnodes and their scores.
  typedef std::pair<float, EdgeAndNode> ScoredEdge;
  std::vector<ScoredEdge> scores;
  float puct_mult =
      params_.GetCpuct() * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  // FPU reduction is not taken into account.
  const float parent_q = -node->GetQ();
  for (auto edge : node->Edges()) {
    if (edge.GetP() == 0.0f) continue;
    // Flip the sign of a score to be able to easily sort.
    scores.emplace_back(-edge.GetU(puct_mult) - edge.GetQ(parent_q), edge);
  }

  size_t first_unsorted_index = 0;
  int total_budget_spent = 0;
  int budget_to_spend = budget;  // Initialize for the case where there's only
                                 // one child.
  for (size_t i = 0; i < scores.size(); ++i) {
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
      const float q = edge.GetQ(-parent_q);
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
    // Terminal or certain nodes don't involve the neural NetworkComputation,
    // nor do they require any further processing after value retrieval.
    node_to_process->v = node->GetQ();
    return;
  }
  // For NN results, we need to populate policy as well as value.
  // First the value...
  node_to_process->v = -computation_->GetQVal(idx_in_computation);
  // ...and secondly, the policy data.
  float total = 0.0;
  for (auto edge : node->Edges()) {
    float p =
        computation_->GetPVal(idx_in_computation, edge.GetMove().as_nn_index());
    if (params_.GetPolicySoftmaxTemp() != 1.0f) {
      p = pow(p, 1 / params_.GetPolicySoftmaxTemp());
    }
    edge.edge()->SetP(p);
    // Edge::SetP does some rounding, so only add to the total after rounding.
    total += edge.edge()->GetP();
  }
  // Normalize P values to add up to 1.0.
  if (total > 0.0f) {
    float scale = 1.0f / total;
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

  // Backup V value up to a root. After 1 visit, V = Q.
  float v = node_to_process.v;
  bool origin_bounded = node->IsBounded();
  for (Node* n = node; n != search_->root_node_->GetParent();
       n = n->GetParent()) {
    // Certainty Propagation:
    // If update could affect bounds (origin_bounded),
    // check all childs, and update bounds/certainty.
    if (n != search_->root_node_ && params_.GetCertaintyPropagation() &&
        n != node && (origin_bounded) && !n->IsCertain()) {
      bool based_on_propagated_tbhit = false;
      float lower_bound = -1.0f;
      float upper_bound = -1.0f;
      for (auto iter : n->Edges()) {
        if (iter.IsLBounded() && iter.GetEQ() > lower_bound)
          lower_bound = iter.GetEQ();
        if (iter.IsUBounded() && iter.GetEQ() > upper_bound)
          upper_bound = iter.GetEQ();
        // Only checking !UBounded so that lower bounded
        // edges, also get the correct upper_bound
        if (!iter.IsUBounded()) upper_bound = 1.0f;
        if (lower_bound == upper_bound && lower_bound == 1.0f) {
          based_on_propagated_tbhit = iter.IsPropagatedTBHit();
          break;
        }
        based_on_propagated_tbhit |= iter.IsPropagatedTBHit();
      }
      // Exact scores are certain and propagate certainty.
      // Inexact scores propagate their bounds.
      if (lower_bound == upper_bound) {
        v = -lower_bound;
        n->MakeCertain(v, based_on_propagated_tbhit ? CertaintyTrigger::TB_HIT
                                                    : CertaintyTrigger::NORMAL);
        search_->certain_.fetch_add(1, std::memory_order_acq_rel);
      } else {
        if (lower_bound > -1.0f) n->UBound(-lower_bound);
        if (upper_bound < 1.0f) n->LBound(-upper_bound);
        if (lower_bound > -1.0f || upper_bound < 1.0f)
          search_->bounds_.fetch_add(1, std::memory_order_acq_rel);
      }
    }

    // Certainty propagation: reduce error by keeping score in proven bounds.
    if (params_.GetCertaintyPropagation() > 1 && n != search_->root_node_ &&
        !n->IsCertain()) {
      if (n->GetEdgeToMe()->IsUBounded() && v > 0.0f) v = 0.0f;
      if (n->GetEdgeToMe()->IsLBounded() && v < 0.0f) v = 0.0f;
    }

    n->FinalizeScoreUpdate(v, node_to_process.multivisit);
    // Q will be flipped for opponent.
    v = -v;

    // Update best move if new N > best N or
    // if the node is a certain child of root
    if (n->GetParent() == search_->root_node_ &&
        (search_->best_move_edge_.GetN() <= n->GetN() || n->IsCertain())) {
      search_->best_move_edge_ =
          search_->GetBestChildNoTemperature(search_->root_node_);
    }
  }
  search_->total_playouts_ += node_to_process.multivisit;
  search_->cum_depth_ += node_to_process.depth;
  search_->max_depth_ = std::max(search_->max_depth_, node_to_process.depth);
}  // namespace lczero

// 7. Update the Search's status and progress information.
//~~~~~~~~~~~~~~~~~~~~
void SearchWorker::UpdateCounters() {
  search_->UpdateRemainingMoves();  // Updates smart pruning counters.
  search_->MaybeOutputInfo();
  search_->MaybeTriggerStop();

  // If this thread had no work, sleep for some milliseconds.
  // Collisions don't count as work, so have to enumerate to find out if there
  // was anything done.
  bool work_done = false;
  for (NodeToProcess& node_to_process : minibatch_) {
    if (!node_to_process.IsCollision()) {
      work_done = true;
      break;
    }
  }
  if (!work_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace lczero

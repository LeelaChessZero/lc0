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
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

#include "mcts/node.h"
#include "neural/cache.h"
#include "neural/encoder.h"
#include "neural/network_st_batch.h"
#include "utils/random.h"

namespace lczero {

const char* SearchParams::kMiniBatchSizeStr = "Minibatch size for NN inference";
const char* SearchParams::kCpuctStr = "Cpuct MCTS option";
const char* SearchParams::kTemperatureStr = "Initial temperature";
const char* SearchParams::kTempDecayMovesStr = "Moves with temperature decay";
const char* SearchParams::kNoiseStr = "Add Dirichlet noise at root node";
const char* SearchParams::kVerboseStatsStr = "Display verbose move stats";
const char* SearchParams::kAggressiveTimePruningStr =
    "Aversion to search if change unlikely";
const char* SearchParams::kFpuReductionStr = "First Play Urgency Reduction";
const char* SearchParams::kCacheHistoryLengthStr =
    "Length of history to include in cache";
const char* SearchParams::kPolicySoftmaxTempStr = "Policy softmax temperature";
const char* SearchParams::kAllowedNodeCollisionsStr =
    "Allowed node collisions, per batch";
const char* SearchParams::kOutOfOrderEvalStr =
    "Out-of-order cache backpropagation";

namespace {
const int kSmartPruningToleranceNodes = 100;
const int kSmartPruningToleranceMs = 200;
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 5000;
}  // namespace

//////////////////////////////////////////////////////////////////////////////
// SearchParams
//////////////////////////////////////////////////////////////////////////////

SearchParams::SearchParams(const OptionsDict& options)
    : kMiniBatchSize(options.Get<int>(kMiniBatchSizeStr)),
      kCpuct(options.Get<float>(kCpuctStr)),
      kTemperature(options.Get<float>(kTemperatureStr)),
      kTempDecayMoves(options.Get<int>(kTempDecayMovesStr)),
      kNoise(options.Get<bool>(kNoiseStr)),
      kVerboseStats(options.Get<bool>(kVerboseStatsStr)),
      kAggressiveTimePruning(options.Get<float>(kAggressiveTimePruningStr)),
      kFpuReduction(options.Get<float>(kFpuReductionStr)),
      kCacheHistoryLength(options.Get<int>(kCacheHistoryLengthStr)),
      kPolicySoftmaxTemp(options.Get<float>(kPolicySoftmaxTempStr)),
      kAllowedNodeCollisions(options.Get<int>(kAllowedNodeCollisionsStr)),
      kOutOfOrderEval(options.Get<bool>(kOutOfOrderEvalStr)) {}

void SearchParams::PopulateUciParams(OptionsParser* options) {
  // Here the "safe defaults" are listed.
  // Many of them are overridden with optimized defaults in engine.cc and
  // tournament.cc

  options->Add<IntOption>(kMiniBatchSizeStr, 1, 1024, "minibatch-size") = 1;
  options->Add<FloatOption>(kCpuctStr, 0.0f, 100.0f, "cpuct") = 1.2f;
  options->Add<FloatOption>(kTemperatureStr, 0.0f, 100.0f, "temperature") =
      0.0f;
  options->Add<IntOption>(kTempDecayMovesStr, 0, 100, "tempdecay-moves") = 0;
  options->Add<BoolOption>(kNoiseStr, "noise", 'n') = false;
  options->Add<BoolOption>(kVerboseStatsStr, "verbose-move-stats") = false;
  options->Add<FloatOption>(kAggressiveTimePruningStr, 0.0f, 10.0f,
                            "futile-search-aversion") = 1.33f;
  options->Add<FloatOption>(kFpuReductionStr, -100.0f, 100.0f,
                            "fpu-reduction") = 0.0f;
  options->Add<IntOption>(kCacheHistoryLengthStr, 0, 7,
                          "cache-history-length") = 7;
  options->Add<FloatOption>(kPolicySoftmaxTempStr, 0.1f, 10.0f,
                            "policy-softmax-temp") = 1.0f;
  options->Add<IntOption>(kAllowedNodeCollisionsStr, 0, 1024,
                          "allowed-node-collisions") = 0;
  options->Add<BoolOption>(kOutOfOrderEvalStr, "out-of-order-eval") = true;
}

//////////////////////////////////////////////////////////////////////////////
// Search
//////////////////////////////////////////////////////////////////////////////

Search::Search(const NodeTree& tree, Network* network,
               BestMoveInfo::Callback best_move_callback,
               ThinkingInfo::Callback info_callback, const SearchLimits& limits,
               const OptionsDict& options, NNCache* cache,
               SyzygyTablebase* syzygy_tb)
    : worker_overlord_(options, cache),
      cache_(cache),
      syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      network_(network),
      limits_(limits),
      start_time_(std::chrono::steady_clock::now()),
      initial_visits_(tree.GetCurrentHeadNode()->GetN()),
      best_move_callback_(best_move_callback),
      info_callback_(info_callback),
      params_(options) {
  worker_overlord_.SpawnNewWorker(true, tree.GetTreeAtCurrentMove(),
                                  played_history_);
}

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

Node* Search::GetRootNode() const {
  return worker_overlord_.GetRootWorker()->GetRootNode();
}

void Search::SendUciInfo() REQUIRES(nodes_mutex_) {
  const auto& best_move_edge =
      worker_overlord_.GetRootWorker()->GetBestMoveEdge();
  if (!best_move_edge) return;
  SearchWorker* root_worker = worker_overlord_.GetRootWorker();
  auto total_playouts = root_worker->GetTotalPlayouts();
  last_outputted_best_move_edge_ = best_move_edge.edge();
  uci_info_.depth = cum_depth_ / (total_playouts ? total_playouts : 1);
  uci_info_.seldepth = max_depth_;
  uci_info_.time = GetTimeSinceStart();
  uci_info_.nodes = total_playouts + initial_visits_;
  uci_info_.hashfull =
      cache_->GetSize() * 1000LL / std::max(cache_->GetCapacity(), 1);
  uci_info_.nps = uci_info_.time ? (total_playouts * 1000 / uci_info_.time) : 0;
  uci_info_.score = 290.680623072 * tan(1.548090806 * best_move_edge.GetQ(0));
  uci_info_.tb_hits = tb_hits_.load(std::memory_order_acquire);
  uci_info_.pv.clear();

  bool flip = played_history_.IsBlackToMove();
  for (auto iter = best_move_edge; iter;
       iter = GetBestChildNoTemperature(iter.node()), flip = !flip) {
    uci_info_.pv.push_back(iter.GetMove(flip));
    if (!iter.node()) break;  // Last edge was dangling, cannot continue.
  }
  uci_info_.comment.clear();
  info_callback_(uci_info_);
}

// Decides whether anything important changed in stats and new info should be
// shown to a user.
void Search::MaybeOutputInfo() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  const auto& best_move_edge =
      worker_overlord_.GetRootWorker()->GetBestMoveEdge();
  SearchWorker* root_worker = worker_overlord_.GetRootWorker();
  auto total_playouts = root_worker->GetTotalPlayouts();

  if (!responded_bestmove_ && best_move_edge &&
      (best_move_edge.edge() != last_outputted_best_move_edge_ ||
       uci_info_.depth !=
           static_cast<int>(cum_depth_ /
                            (total_playouts ? total_playouts : 1)) ||
       uci_info_.seldepth != max_depth_ ||
       uci_info_.time + kUciInfoMinimumFrequencyMs < GetTimeSinceStart())) {
    SendUciInfo();
  }
}

int64_t Search::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

void Search::SendMovesStats() const {
  Node* root_node = GetRootNode();
  const float parent_q =
      -root_node->GetQ() -
      params_.kFpuReduction * std::sqrt(root_node->GetVisitedPolicy());
  const float U_coeff =
      params_.kCpuct * std::sqrt(std::max(root_node->GetChildrenVisits(), 1u));

  std::vector<EdgeAndNode> edges;
  for (const auto& edge : root_node->Edges()) edges.push_back(edge);

  std::sort(edges.begin(), edges.end(),
            [&parent_q, &U_coeff](EdgeAndNode a, EdgeAndNode b) {
              return std::forward_as_tuple(a.GetN(),
                                           a.GetQ(parent_q) + a.GetU(U_coeff)) <
                     std::forward_as_tuple(b.GetN(),
                                           b.GetQ(parent_q) + b.GetU(U_coeff));
            });

  const bool is_black_to_move = played_history_.IsBlackToMove();
  ThinkingInfo info;
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

    oss << "(Q: " << std::setw(8) << std::setprecision(5) << edge.GetQ(parent_q)
        << ") ";

    oss << "(U: " << std::setw(6) << std::setprecision(5) << edge.GetU(U_coeff)
        << ") ";

    oss << "(Q+U: " << std::setw(8) << std::setprecision(5)
        << edge.GetQ(parent_q) + edge.GetU(U_coeff) << ") ";

    oss << "(V: ";
    optional<float> v;
    if (edge.IsTerminal()) {
      v = edge.node()->GetQ();
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

    if (edge.IsTerminal()) oss << "(T) ";

    info.comment = oss.str();
    info_callback_(info);
  }
}

NNCacheLock Search::GetCachedFirstPlyResult(EdgeAndNode edge) const {
  if (!edge.HasNode()) return {};
  assert(edge.node()->GetParent() == GetRootNode());
  // It would be relatively straightforward to generalize this to fetch NN
  // results for an abitrary move.
  optional<float> retval;
  PositionHistory history(played_history_);  // Is it worth it to move this
  // initialization to SendMoveStats, reducing n memcpys to 1? Probably not.
  history.Append(edge.GetMove());
  auto hash = history.HashLast(params_.kCacheHistoryLength + 1);
  NNCacheLock nneval(cache_, hash);
  return nneval;
}

void Search::MaybeTriggerStop() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  SearchWorker* root_worker = worker_overlord_.GetRootWorker();
  auto total_playouts = root_worker->GetTotalPlayouts();

  // Already responded bestmove, nothing to do here.
  if (responded_bestmove_) return;
  // Don't stop when the root node is not yet expanded.
  if (total_playouts == 0) return;
  // If smart pruning tells to stop (best move found), stop.
  if (found_best_move_) {
    FireStopInternal();
  }
  // Stop if reached playouts limit.
  if (limits_.playouts >= 0 && total_playouts >= limits_.playouts) {
    FireStopInternal();
  }
  // Stop if reached visits limit.
  if (limits_.visits >= 0 &&
      total_playouts + initial_visits_ >= limits_.visits) {
    FireStopInternal();
  }
  // Stop if reached time limit.
  if (limits_.time_ms >= 0 && GetTimeSinceStart() >= limits_.time_ms) {
    FireStopInternal();
  }
  // If we are the first to see that stop is needed.
  if (stop_ && !responded_bestmove_) {
    SendUciInfo();
    if (params_.kVerboseStats) SendMovesStats();
    best_move_ = GetBestMoveInternal();
    best_move_callback_({best_move_.first, best_move_.second});
    responded_bestmove_ = true;
  }
}

void Search::UpdateRemainingMoves() {
  if (params_.kAggressiveTimePruning <= 0.0f) return;
  SharedMutex::Lock lock(nodes_mutex_);
  SearchWorker* root_worker = worker_overlord_.GetRootWorker();
  auto total_playouts = root_worker->GetTotalPlayouts();

  remaining_playouts_ = std::numeric_limits<int>::max();
  // Check for how many playouts there is time remaining.
  if (limits_.time_ms >= 0) {
    auto time_since_start = GetTimeSinceStart();
    if (time_since_start > kSmartPruningToleranceMs) {
      auto nps = (1000LL * total_playouts + kSmartPruningToleranceNodes) /
                     (time_since_start - kSmartPruningToleranceMs) +
                 1;
      int64_t remaining_time = limits_.time_ms - time_since_start;
      // Put early_exit scaler here so calculation doesn't have to be done on
      // every node.
      int64_t remaining_playouts =
          remaining_time * nps / params_.kAggressiveTimePruning / 1000;
      // Don't assign directly to remaining_playouts_ as overflow is possible.
      if (remaining_playouts < remaining_playouts_)
        remaining_playouts_ = remaining_playouts;
    }
  }
  // Check how many visits are left.
  if (limits_.visits >= 0) {
    // Add kMiniBatchSize, as it's possible to exceed visits limit by that
    // number.
    auto remaining_visits = limits_.visits - total_playouts - initial_visits_ +
                            params_.kMiniBatchSize - 1;

    if (remaining_visits < remaining_playouts_)
      remaining_playouts_ = remaining_visits;
  }
  if (limits_.playouts >= 0) {
    // Add kMiniBatchSize, as it's possible to exceed visits limit by that
    // number.
    auto remaining_playouts =
        limits_.visits - total_playouts + params_.kMiniBatchSize + 1;
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
  float parent_q = -GetRootNode()->GetQ();
  if (!GetRootNode()->HasChildren()) return parent_q;
  EdgeAndNode best_edge = GetBestChildNoTemperature(GetRootNode());
  return best_edge.GetQ(parent_q);
}

std::pair<Move, Move> Search::GetBestMove() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  return GetBestMoveInternal();
}

bool Search::PopulateRootMoveLimit(MoveList* root_moves) const {
  // Search moves overrides tablebase.
  if (!limits_.searchmoves.empty()) {
    *root_moves = limits_.searchmoves;
    return false;
  }
  auto board = played_history_.Last().GetBoard();
  if (!syzygy_tb_ || !board.castlings().no_legal_castle() ||
      (board.ours() + board.theirs()).count() > syzygy_tb_->max_cardinality()) {
    return false;
  }
  return syzygy_tb_->root_probe(played_history_.Last(), root_moves) ||
         syzygy_tb_->root_probe_wdl(played_history_.Last(), root_moves);
}

// Returns the best move, maybe with temperature (according to the settings).
std::pair<Move, Move> Search::GetBestMoveInternal() const
    REQUIRES_SHARED(nodes_mutex_) REQUIRES_SHARED(counters_mutex_) {
  Node* root_node = GetRootNode();
  if (responded_bestmove_) return best_move_;
  if (!root_node->HasChildren()) return {};

  float temperature = params_.kTemperature;
  if (temperature && params_.kTempDecayMoves) {
    int moves = played_history_.Last().GetGamePly() / 2;
    if (moves >= params_.kTempDecayMoves) {
      temperature = 0.0;
    } else {
      temperature *= static_cast<float>(params_.kTempDecayMoves - moves) /
                     params_.kTempDecayMoves;
    }
  }

  auto best_node = temperature && root_node->GetChildrenVisits() > 0
                       ? GetBestChildWithTemperature(root_node, temperature)
                       : GetBestChildNoTemperature(root_node);

  Move ponder_move;  // Default is "null move" which means "don't display
                     // anything".
  if (best_node.HasNode() && best_node.node()->HasChildren()) {
    ponder_move = GetBestChildNoTemperature(best_node.node())
                      .GetMove(!played_history_.IsBlackToMove());
  }
  return {best_node.GetMove(played_history_.IsBlackToMove()), ponder_move};
}

// Returns a child with most visits.
EdgeAndNode Search::GetBestChildNoTemperature(Node* parent) {
  /*  Node* root_node = GetRootNode();
    MoveList root_limit;
    if (parent == root_node) {
      PopulateRootMoveLimit(&root_limit);
    } */
  EdgeAndNode best_edge;
  // Best child is selected using the following criteria:
  // * Largest number of playouts.
  // * If two nodes have equal number:
  //   * If that number is 0, the one with larger prior wins.
  //   * If that number is larger than 0, the one with larger eval wins.
  std::tuple<int, float, float> best(-1, 0.0, 0.0);
  for (auto edge : parent->Edges()) {
    /*    if (parent == root_node && !root_limit.empty() &&
            std::find(root_limit.begin(), root_limit.end(), edge.GetMove()) ==
                root_limit.end()) {
          continue;
        } */
    std::tuple<int, float, float> val(edge.GetN(), edge.GetQ(-10.0),
                                      edge.GetP());
    if (val > best) {
      best = val;
      best_edge = edge;
    }
  }
  return best_edge;
}

// Returns a child chosen according to weighted-by-temperature visit count.
EdgeAndNode Search::GetBestChildWithTemperature(Node* parent,
                                                float temperature) const {
  Node* root_node = GetRootNode();
  MoveList root_limit;
  if (parent == root_node) {
    PopulateRootMoveLimit(&root_limit);
  }

  assert(parent->GetChildrenVisits() > 0);
  std::vector<float> cumulative_sums;
  float sum = 0.0;
  const float n_parent = parent->GetN();

  for (auto edge : parent->Edges()) {
    if (parent == root_node && !root_limit.empty() &&
        std::find(root_limit.begin(), root_limit.end(), edge.GetMove()) ==
            root_limit.end()) {
      continue;
    }
    sum += std::pow(edge.GetN() / n_parent, 1 / temperature);
    cumulative_sums.push_back(sum);
  }

  float toss = Random::Get().GetFloat(cumulative_sums.back());
  int idx =
      std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) -
      cumulative_sums.begin();

  for (auto edge : parent->Edges()) {
    if (parent == root_node && !root_limit.empty() &&
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
    int thread_id = threads_.size() - 1;
    threads_.emplace_back([this, thread_id]() { WorkerThread(thread_id); });
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

// A main search thread. There may be several of those.
void Search::WorkerThread(int thread_num) {
  // The requests from multiple detached subtrees will be computed in parallel
  // through SingleThreadBatchingNetwork adapter.
  SingleThreadBatchingNetwork network(network_);

  while (IsSearchActive()) {
    const int epoch = epoch_.fetch_add(1, std::memory_order_acq_rel);
    std::cerr << "Epoch " << epoch << std::endl;
    // Make batching network collect a new batch.
    network.Reset();

    std::vector<WorkerOverlord::LeasedWorker> workers;

    worker_overlord_.DemoteIdleWorkers(epoch);

    if (thread_num == 0 ||
        worker_overlord_.GetTotalIdleBatchSize() >= params_.kMiniBatchSize) {
      // While there is a space in a minibatch, get more workers.
      while (network.GetTotalBatchSize() < params_.kMiniBatchSize) {
        // Get free worker with a highest priority.
        auto lease = worker_overlord_.AcquireWorker(
            params_.kMiniBatchSize - network.GetTotalBatchSize());

        // No workers availabe, break.
        if (!lease.worker) break;

        // 1. Initialize internal structures.
        // @computation is the computation to use on this iteration.
        lease.worker->InitializeIteration(network.NewComputation());

        // 2. Gather minibatch.
        lease.worker->GatherMinibatch(
            std::min(lease.recommended_batch_size,
                     params_.kMiniBatchSize - network.GetTotalBatchSize()));

        workers.push_back(std::move(lease));
      }
    } else {
      worker_overlord_.ReportEmptyBatch(params_.kMiniBatchSize);
    }

    if (network.GetTotalBatchSize())
      std::cerr << "Batch size: " << network.GetTotalBatchSize() << std::endl;
    // 3. Run NN computation.
    // In fact batching network adapter only actually will run computation once.
    for (auto& lease : workers) lease.worker->RunNNComputation();

    std::vector<WorkerOverlord::DetachCandidate> candidates;
    for (auto& lease : workers) {
      // 4. Retrieve NN computations (and terminal values) into nodes.
      lease.worker->FetchMinibatchResults();

      // 5. Propagate the new nodes' information to all their parents in the
      // tree.
      lease.worker->DoBackupUpdate();

      // 6. Transfer information from the root of the subtree into the subtree
      // stub.
      lease.worker->TransferCountersToStub(&candidates);

      lease.worker->last_epoch_ = epoch;
    }

    // Pass candidates for detaching to overlord, maybe it will detach some.
    worker_overlord_.MaybeDetach(candidates);

    // Release workers back to a pool for other threads to use.
    for (auto& lease : workers) {
      lease.worker->tree_->debux2_ = epoch;
      worker_overlord_.ReleaseWorker(std::move(lease));
    }

    UpdateRemainingMoves();  // Updates smart pruning counters.
    MaybeOutputInfo();
    MaybeTriggerStop();

    if (workers.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void Search::FireStopInternal() REQUIRES(counters_mutex_) {
  stop_ = true;
  watchdog_cv_.notify_all();
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
}

//////////////////////////////////////////////////////////////////////////////
// SearchWorker
//////////////////////////////////////////////////////////////////////////////

SearchWorker::SearchWorker(const SearchParams& params,
                           const PositionHistory& history, SubTree* tree,
                           NNCache* cache, WorkerOverlord* overlord)
    : tree_(tree),
      history_length_(history.GetLength()),
      history_(history),
      cache_(cache),
      params_(params),
      overlord_(overlord) {
  tree_->SetHasAssignedWorker();
}

SearchWorker::~SearchWorker() { tree_->ResetHasAssignedWorker(); }

int64_t SearchWorker::GetTotalPlayouts() const {
  return total_playouts_.load(std::memory_order_acquire);
}

// 1. Initialize internal structures.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::InitializeIteration(
    std::unique_ptr<NetworkComputation> computation) {
  computation_ =
      std::make_unique<CachingComputation>(std::move(computation), cache_);
  minibatch_.clear();

  int target_ahead = tree_->GetTargetAheadNodes();

  if (reached_target_ahead_ && tree_->IsBehind()) {
    tree_->SetTargetAheadNodes(std::min(15 + target_ahead + target_ahead / 16,
                                        params_.kMiniBatchSize));
    reached_target_ahead_ = false;
  } /* else if (target_ahead > 21) {
     tree_->SetTargetAheadNodes(target_ahead - 20);
   } else {
     tree_->SetTargetAheadNodes(1);
   }*/

  /* if (!root_move_filter_populated_) {
    root_move_filter_populated_ = true;
    if (search_->PopulateRootMoveLimit(&root_move_filter_)) {
      search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
    }
  } */
}

// 2. Gather minibatch.
// ~~~~~~~~~~~~~~~~~~~~
void SearchWorker::GatherMinibatch(int max_batch_size) {
  // Total number of nodes to process.
  int minibatch_size = 0;
  int collisions_found = 0;
  // Number of nodes processed out of order.
  int number_out_of_order = 0;

  // Gather nodes to process in the current batch.
  // If we had too many (kMiniBatchSize) nodes out of order, also interrupt
  // the iteration so that search can exit.
  // TODO(crem) change that to checking search_->stop_ when bestmove reporting
  // is in a separate thread.
  while (minibatch_size < max_batch_size /*&&
         number_out_of_order < max_batch_size*/) {
    // If there's something to process without touching slow neural net, do
    // it.
    if (minibatch_size > 0 && computation_->GetCacheMisses() == 0) return;
    // Pick next node to extend.
    minibatch_.emplace_back(PickNodeToExtend());
    auto& picked_node = minibatch_.back();
    auto* node = picked_node.node;

    // There was a collision. If limit has been reached, return, otherwise
    // just start search of another node.
    if (picked_node.is_collision) {
      if (++collisions_found > params_.kAllowedNodeCollisions) return;
      continue;
    }
    ++minibatch_size;

    // If node is already known as terminal (win/loss/draw according to rules
    // of the game), or it has a detached subtree, it means that we already
    // visited this node before and have no need to extend it.
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
    if (params_.kOutOfOrderEval && picked_node.CanEvalOutOfOrder()) {
      // Perform out of order eval for the last entry in minibatch_.
      FetchSingleNodeResult(&picked_node, computation_->GetBatchSize() - 1);
      DoBackupUpdateSingleNode(picked_node);

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

// Returns node and whether there's been a search collision on the node.
SearchWorker::NodeToProcess SearchWorker::PickNodeToExtend() {
  // Starting from search_->root_node_, generate a playout, choosing a
  // node at each level according to the MCTS formula. n_in_flight_ is
  // incremented for each node in the playout (via TryStartScoreUpdate()).

  Node* node = tree_->GetRootNode();
  Node::Iterator best_edge;
  // Initialize position sequence with pre-move position.
  history_.Trim(history_length_);

  // Fetch the current best root node visits for possible smart pruning.
  // int best_node_n = search_->best_move_edge_.GetN();

  // True on first iteration, false as we dive deeper.
  bool is_root_node = true;
  uint16_t depth = 0;
  Node* node_at_root = nullptr;

  while (true) {
    // First, terminate if we find collisions or leaf nodes.
    // Set 'node' to point to the node that was picked on previous iteration,
    // possibly spawning it.
    // TODO(crem) This statement has to be in the end of the loop rather than
    //            in the beginning (and there would be no need for "if
    //            (!is_root_node)"), but that would mean extra mutex lock.
    //            Will revisit that after rethinking locking strategy.
    if (!is_root_node) {
      node = best_edge.GetOrSpawnNode(/* parent */ node);
      if (!node_at_root) node_at_root = node;
    }
    depth++;
    // If node has detached subtree, that may have many reasons.
    if (node->HasDetachedSubtree()) {
      SubTree* subtree = node->GetDetachedSubtree();
      // If the subtree doesn't have worker allocated, add it.
      if (!subtree->HasWorker()) {
        overlord_->SpawnNewWorker(false, node->GetDetachedSubtree(), history_);
      }
      // DO NOT SUBMIT write comment
      if (node->TryStartUpdateFromSubtree()) {
        return NodeToProcess::Subtree(node, depth, node_at_root);
      } else {
        return NodeToProcess::Collision(node, depth, node_at_root);
      }
    }
    // n_in_flight_ is incremented. If the method returns false, then there is
    // a search collision, and this node is already being expanded.
    if (!node->TryStartScoreUpdate()) {
      return NodeToProcess::Collision(node, depth, node_at_root);
    }
    // Either terminal or unexamined leaf node -- the end of this playout.
    if (!node->HasChildren()) {
      return NodeToProcess::Extension(node, depth, node_at_root);
    }
    // If we fall through, then n_in_flight_ has been incremented but this
    // playout remains incomplete; we must go deeper.
    float puct_mult =
        params_.kCpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
    float best = -100.0f;
    int possible_moves = 0;
    float parent_q =
        ((is_root_node && params_.kNoise) || !params_.kFpuReduction)
            ? -node->GetQ()
            : -node->GetQ() -
                  params_.kFpuReduction * std::sqrt(node->GetVisitedPolicy());
    for (auto child : node->Edges()) {
      if (is_root_node) {
        // If there's no chance to catch up to the current best node with
        // remaining playouts, don't consider it.
        // best_move_node_ could have changed since best_node_n was retrieved.
        // To ensure we have at least one node to expand, always include
        // current best node.
        // DO NOT SUBMIT   Fix smart pruning!
        // if (IsRootWorker() && child != search_->best_move_edge_ &&
        //     search_->remaining_playouts_ <
        //         best_node_n - static_cast<int>(child.GetN())) {
        //   continue;
        // }
        /*
        // If root move filter exists, make sure move is in the list.
        if (!root_move_filter_.empty() &&
            std::find(root_move_filter_.begin(), root_move_filter_.end(),
                      child.GetMove()) == root_move_filter_.end()) {
          continue;
        }*/
        ++possible_moves;
      }
      float Q = child.GetQ(parent_q);
      const float score = child.GetU(puct_mult) + Q;
      if (score > best) {
        best = score;
        best_edge = child;
      }
    }

    history_.Append(best_edge.GetMove());
    /*
    DO NOT SUBMIT
    if (is_root_node && possible_moves <= 1 && !search_->limits_.infinite) {
      // If there is only one move theoretically possible within remaining
    time,
      // output it.
      Mutex::Lock counters_lock(search_->counters_mutex_);
      search_->found_best_move_ = true;
    } */
    is_root_node = false;
  }
}

void SearchWorker::ExtendNode(Node* node) {
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
  if (node != tree_->GetRootNode()) {
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

    // Neither by-position or by-rule termination, but maybe it's a TB
    // position.
    /*if (search_->syzygy_tb_ && board.castlings().no_legal_castle() &&
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
          node->MakeTerminal(GameResult::BLACK_WON);
        } else if (wdl == WDL_LOSS) {
          node->MakeTerminal(GameResult::WHITE_WON);
        } else {  // Cursed wins and blessed losses count as draws.
          node->MakeTerminal(GameResult::DRAW);
        }
        search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
        return;
      }
    }*/
  }

  // Add legal moves as edges of this node.
  node->CreateEdges(legal_moves);
}

// Returns whether node was already in cache.
bool SearchWorker::AddNodeToComputation(Node* node, bool add_if_cached) {
  auto hash = history_.HashLast(params_.kCacheHistoryLength + 1);
  // If already in cache, no need to do anything.
  if (add_if_cached) {
    if (computation_->AddInputByHash(hash)) return true;
  } else {
    if (cache_->ContainsKey(hash)) return true;
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

// 3. Run NN computation.
// ~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::RunNNComputation() { computation_->ComputeBlocking(); }

// 4. Retrieve NN computations (and terminal values) into nodes.
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
  if (node_to_process->is_subtree) {
    // If detached subtree, there is a magic formuala for V which correctly
    // updates Q all the way to the root.
    auto new_q = node->GetDetachedSubtree()->GetQ();
    node_to_process->v = new_q + (new_q - node->GetQ()) * node->GetN();
    return;
  }
  if (!node_to_process->nn_queried) {
    // Terminal nodes don't involve the neural NetworkComputation, nor do
    // they require any further processing after value retrieval.
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
    if (params_.kPolicySoftmaxTemp != 1.0f) {
      p = pow(p, 1 / params_.kPolicySoftmaxTemp);
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
  if (params_.kNoise && node == tree_->GetRootNode() && IsRootWorker()) {
    ApplyDirichletNoise(node, 0.25, 0.3);
  }
}

// 5. Propagate the new nodes' information to all their parents in the tree.
// ~~~~~~~~~~~~~~
void SearchWorker::DoBackupUpdate() {
  for (const NodeToProcess& node_to_process : minibatch_) {
    DoBackupUpdateSingleNode(node_to_process);
  }
}

void SearchWorker::DoBackupUpdateSingleNode(
    const NodeToProcess& node_to_process) {
  Node* node = node_to_process.node;
  if (node_to_process.is_collision) {
    // If it was a collision, just undo counters.
    for (node = node->GetParent(); node; node = node->GetParent()) {
      node->CancelScoreUpdate();
    }
    return;
  }

  // Backup V value up to a root. After 1 visit, V = Q.
  float v = node_to_process.v;
  for (Node* n = node; n; n = n->GetParent()) {
    n->FinalizeScoreUpdate(v);
    // Q will be flipped for opponent.
    v = -v;

    // Update best move.
    if (n->GetParent() == nullptr && best_move_edge_.GetN() <= n->GetN()) {
      best_move_edge_ = Search::GetBestChildNoTemperature(GetRootNode());
    }
  }
  node = node_to_process.node;
  if (node->HasDetachedSubtree()) {
    node->GetDetachedSubtree()->PullStatsFromParent();
  }
  // DO NOT SUBMIT
  /* search_->cum_depth_ += node_to_process.depth;
  search_->max_depth_ = std::max(search_->max_depth_, node_to_process.depth);
  */
  total_playouts_.fetch_add(1, std::memory_order_release);
}

// 6. Transfer information from the root of the subtree into the subtree stub.
// ~~~~~~~~~~~~~~
void SearchWorker::TransferCountersToStub(
    std::vector<WorkerOverlord::DetachCandidate>* candidates) {
  Node* root = tree_->GetRootNode();
  tree_->UpdateNQ(root->GetN(), root->GetQ());
  if (tree_->IsAhead()) {
    reached_target_ahead_ = true;
  }

  int evaled_nodes = 0;

  int counter = 0;
  Node* best = nullptr;
  std::unordered_map<Node*, int> node_to_count;

  for (const NodeToProcess& node_to_process : minibatch_) {
    if (!node_to_process.nn_queried && !node_to_process.is_cache_hit) {
      continue;
    }
    if (!node_to_process.node_at_root) continue;
    ++evaled_nodes;
    auto new_count = ++node_to_count[node_to_process.node_at_root];
    if (new_count > counter) {
      counter = new_count;
      best = node_to_process.node_at_root;
    }
  }

  if (best == nullptr) return;

  counter = 0;
  for (const NodeToProcess& node_to_process : minibatch_) {
    if (!node_to_process.nn_queried && !node_to_process.is_cache_hit) {
      continue;
    }
    if (best == node_to_process.node_at_root) ++counter;
  }

  candidates->emplace_back(this, best, best->GetEdgeToSelf()->GetP(), counter,
                           evaled_nodes, minibatch_.size());
}

const PositionHistory& SearchWorker::GetHistoryToNode(Node* node) {
  std::vector<Move> moves;
  for (; node->GetParent(); node = node->GetParent()) {
    moves.push_back(node->GetEdgeToSelf()->GetMove());
  }
  std::reverse(moves.begin(), moves.end());

  history_.Trim(history_length_);
  for (const auto& m : moves) history_.Append(m);
  return history_;
}

//////////////////////////////////////////////////////////////////////////////
// WorkerOverlord
//////////////////////////////////////////////////////////////////////////////

namespace {
int kMinBatchReserve = 256;
}

void WorkerOverlord::SpawnNewWorker(bool is_root, SubTree* tree,
                                    const PositionHistory& history) {
  assert(static_cast<bool>(root_worker_) != is_root);

  auto new_worker =
      std::make_unique<SearchWorker>(params_, history, tree, cache_, this);
  if (is_root) root_worker_ = new_worker.get();

  ReleaseWorker(std::move(new_worker));
}

int WorkerOverlord::GetTotalIdleBatchSize() const {
  Mutex::Lock lock(queue_mutex_);
  int total_batch_size = 0;
  for (auto& worker : idle_workers_) {
    total_batch_size += worker->GetRecommendedBatch();
  }
  return total_batch_size;
}

void WorkerOverlord::DemoteIdleWorkers(int64_t current_epoch) {
  Mutex::Lock lock(queue_mutex_);
  for (auto& worker : idle_workers_) {
    if (worker->GetRecommendedBatch()) continue;
    if (worker->last_epoch_ < current_epoch - 100) {
      worker->last_epoch_ = current_epoch;
      int nodes = worker->tree_->GetTargetAheadNodes();
      if (nodes > 1) nodes /= 2;
      worker->tree_->SetTargetAheadNodes(nodes);
    }
  }
}

void WorkerOverlord::ReportEmptyBatch(int batch_size) {
  Mutex::Lock lock(queue_mutex_);
  nodes_to_add_into_batch_ = kMinBatchReserve + batch_size;
}

WorkerOverlord::LeasedWorker WorkerOverlord::AcquireWorker(int batch_left) {
  Mutex::Lock lock(queue_mutex_);

  int total_available_workers = 0;
  int recommended_batch_size = 0;
  std::unique_ptr<SearchWorker>* best_worker = nullptr;
  int total_batch_size = 0;
  for (auto& worker : idle_workers_) {
    int recommended_batch =
        std::min(worker->GetRecommendedBatch(), batch_left + kMinBatchReserve);
    if (recommended_batch <= 0) continue;
    ++total_available_workers;
    total_batch_size += recommended_batch;
    if (recommended_batch > recommended_batch_size) {
      best_worker = &worker;
      recommended_batch_size = recommended_batch;
    }
  }

  nodes_to_add_into_batch_ =
      std::max(nodes_to_add_into_batch_.load(), kMinBatchReserve + batch_left -
                                                    total_batch_size -
                                                    recommended_batch_size);

  if (!best_worker) return {};

  std::swap(*best_worker, idle_workers_.back());
  auto res = std::move(idle_workers_.back());

  idle_workers_.pop_back();
  return {std::move(res), recommended_batch_size};
}

void WorkerOverlord::ReleaseWorker(std::unique_ptr<SearchWorker> worker) {
  Mutex::Lock lock(queue_mutex_);
  idle_workers_.push_back(std::move(worker));
}

void WorkerOverlord::ReleaseWorker(LeasedWorker lease) {
  ReleaseWorker(std::move(lease.worker));
}

void WorkerOverlord::MaybeDetach(
    const std::vector<DetachCandidate>& candidates) {
  if (nodes_to_add_into_batch_ <= 0) return;
  const DetachCandidate* best_candidate = nullptr;

  for (const auto& candidate : candidates) {
    if (candidate.node_visits < 40) continue;
    if (!best_candidate ||
        best_candidate->node_visits < candidate.node_visits) {
      best_candidate = &candidate;
    }
  }

  if (best_candidate) {
    std::cerr << "Detaching! " << best_candidate->node_visits << '/'
              << best_candidate->total_eval_visits << " ("
              << nodes_to_add_into_batch_ << ")\n";
    auto subtree = best_candidate->node->DetachSubtree();
    subtree->debux_ = best_candidate->node_visits;
    subtree->SetTargetAheadNodes(best_candidate->node_visits);
    SpawnNewWorker(
        false, std::move(subtree),
        best_candidate->worker->GetHistoryToNode(best_candidate->node));
    nodes_to_add_into_batch_ -= best_candidate->node_visits;
  }
}

}  // namespace lczero

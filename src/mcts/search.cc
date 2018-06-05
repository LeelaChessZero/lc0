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
#include "utils/random.h"

namespace lczero {

const char* Search::kMiniBatchSizeStr = "Minibatch size for NN inference";
const char* Search::kMaxPrefetchBatchStr = "Max prefetch nodes, per NN call";
const char* Search::kCpuctStr = "Cpuct MCTS option";
const char* Search::kTemperatureStr = "Initial temperature";
const char* Search::kTempDecayMovesStr = "Moves with temperature decay";
const char* Search::kNoiseStr = "Add Dirichlet noise at root node";
const char* Search::kVerboseStatsStr = "Display verbose move stats";
const char* Search::kSmartPruningStr = "Enable smart pruning";
const char* Search::kVirtualLossBugStr = "Virtual loss bug";
const char* Search::kFpuReductionStr = "First Play Urgency Reduction";
const char* Search::kCacheHistoryLengthStr =
    "Length of history to include in cache";
const char* Search::kExtraVirtualLossStr = "Extra virtual loss";
const char* Search::kPolicySoftmaxTempStr = "Policy softmax temperature";
const char* Search::kAllowedNodeCollisionsStr =
    "Allowed node collisions, per batch";

namespace {
const int kSmartPruningToleranceNodes = 100;
const int kSmartPruningToleranceMs = 200;
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 5000;
}  // namespace

void Search::PopulateUciParams(OptionsParser* options) {
  // Here the "safe defaults" are listed.
  // Many of them are overriden with optimized defaults in engine.cc and
  // tournament.cc

  options->Add<IntOption>(kMiniBatchSizeStr, 1, 1024, "minibatch-size") = 1;
  options->Add<IntOption>(kMaxPrefetchBatchStr, 0, 1024, "max-prefetch") = 32;
  options->Add<FloatOption>(kCpuctStr, 0.0f, 100.0f, "cpuct") = 1.2f;
  options->Add<FloatOption>(kTemperatureStr, 0.0f, 100.0f, "temperature") =
      0.0f;
  options->Add<IntOption>(kTempDecayMovesStr, 0, 100, "tempdecay-moves") = 0;
  options->Add<BoolOption>(kNoiseStr, "noise", 'n') = false;
  options->Add<BoolOption>(kVerboseStatsStr, "verbose-move-stats") = false;
  options->Add<BoolOption>(kSmartPruningStr, "smart-pruning") = true;
  options->Add<FloatOption>(kVirtualLossBugStr, -100.0f, 100.0f,
                            "virtual-loss-bug") = 0.0f;
  options->Add<FloatOption>(kFpuReductionStr, -100.0f, 100.0f,
                            "fpu-reduction") = 0.0f;
  options->Add<IntOption>(kCacheHistoryLengthStr, 0, 7,
                          "cache-history-length") = 7;
  options->Add<FloatOption>(kExtraVirtualLossStr, 0.0f, 100.0f,
                            "extra-virtual-loss") = 0.0f;
  options->Add<FloatOption>(kPolicySoftmaxTempStr, 0.1f, 10.0f,
                            "policy-softmax-temp") = 1.0f;
  options->Add<IntOption>(kAllowedNodeCollisionsStr, 0, 1024,
                          "allowed-node-collisions") = 0;
}

Search::Search(const NodeTree& tree, Network* network,
               BestMoveInfo::Callback best_move_callback,
               ThinkingInfo::Callback info_callback, const SearchLimits& limits,
               const OptionsDict& options, NNCache* cache)
    : root_node_(tree.GetCurrentHead()),
      cache_(cache),
      played_history_(tree.GetPositionHistory()),
      network_(network),
      limits_(limits),
      start_time_(std::chrono::steady_clock::now()),
      initial_visits_(root_node_->GetN()),
      best_move_callback_(best_move_callback),
      info_callback_(info_callback),
      kMiniBatchSize(options.Get<int>(kMiniBatchSizeStr)),
      kMaxPrefetchBatch(options.Get<int>(kMaxPrefetchBatchStr)),
      kCpuct(options.Get<float>(kCpuctStr)),
      kTemperature(options.Get<float>(kTemperatureStr)),
      kTempDecayMoves(options.Get<int>(kTempDecayMovesStr)),
      kNoise(options.Get<bool>(kNoiseStr)),
      kVerboseStats(options.Get<bool>(kVerboseStatsStr)),
      kSmartPruning(options.Get<bool>(kSmartPruningStr)),
      kVirtualLossBug(options.Get<float>(kVirtualLossBugStr)),
      kFpuReduction(options.Get<float>(kFpuReductionStr)),
      kCacheHistoryLength(options.Get<int>(kCacheHistoryLengthStr)),
      kExtraVirtualLoss(options.Get<float>(kExtraVirtualLossStr)),
      kPolicySoftmaxTemp(options.Get<float>(kPolicySoftmaxTempStr)),
      kAllowedNodeCollisions(options.Get<int>(kAllowedNodeCollisionsStr)) {}

// Returns whether node was already in cache.
bool Search::AddNodeToCompute(Node* node, CachingComputation* computation,
                              const PositionHistory& history,
                              bool add_if_cached) {
  auto hash = history.HashLast(kCacheHistoryLength + 1);
  // If already in cache, no need to do anything.
  if (add_if_cached) {
    if (computation->AddInputByHash(hash)) return true;
  } else {
    if (cache_->ContainsKey(hash)) return true;
  }
  auto planes = EncodePositionForNN(history, 8);

  std::vector<uint16_t> moves;

  if (node->HasChildren()) {
    // Legal moves are known, using them.
    for (Node* iter : node->Children()) {
      moves.emplace_back(iter->GetMove().as_nn_index());
    }
  } else {
    // Cache pseudolegal moves. A bit of a waste, but faster.
    const auto& pseudolegal_moves =
        history.Last().GetBoard().GeneratePseudolegalMoves();
    moves.reserve(pseudolegal_moves.size());
    // As an optimization, store moves in reverse order in cache, because
    // that's the order nodes are listed in nodelist.
    for (auto iter = pseudolegal_moves.rbegin(), end = pseudolegal_moves.rend();
         iter != end; ++iter) {
      moves.emplace_back(iter->as_nn_index());
    }
  }

  computation->AddInput(hash, std::move(planes), std::move(moves));
  return false;
}

namespace {
void ApplyDirichletNoise(Node* node, float eps, double alpha) {
  float total = 0;
  std::vector<float> noise;

  // TODO(mooskagh) remove this loop when we store number of children.
  for (Node* iter : node->Children()) {
    (void)iter;  // Silence the unused variable warning.
    float eta = Random::Get().GetGamma(alpha, 1.0);
    noise.emplace_back(eta);
    total += eta;
  }

  if (total < std::numeric_limits<float>::min()) return;

  int noise_idx = 0;
  for (Node* iter : node->Children()) {
    iter->SetP(iter->GetP() * (1 - eps) + eps * noise[noise_idx++] / total);
  }
}
}  // namespace

void Search::Worker() {
  // Nodes to be extended/update counters.
  std::vector<Node*> nodes_to_process;
  // Nodes, for which collision happened. For those only n_in_flight have to
  // be rolled back.
  std::vector<Node*> node_collisions;
  PositionHistory history(played_history_);

  // Exit check is at the end of the loop as at least one iteration is
  // necessary.
  while (true) {
    nodes_to_process.clear();
    node_collisions.clear();
    auto computation = CachingComputation(network_->NewComputation(), cache_);

    // Gather nodes to process in the current batch.
    while (static_cast<int>(nodes_to_process.size()) < kMiniBatchSize) {
      // Initialize position sequence with pre-move position.
      history.Trim(played_history_.GetLength());
      // If there's something to do without touching slow neural net, do it.
      if (!nodes_to_process.empty() && computation.GetCacheMisses() == 0) break;
      // Returns <Node, whether it's computable> pair. The node is not
      // computable if there is collision.
      auto node_and_computable = PickNodeToExtend(root_node_, &history);
      Node* node = node_and_computable.first;
      const bool computable = node_and_computable.second;

      // If there is collision, add to a vector to undo the virtual loss later.
      if (!computable) {
        node_collisions.emplace_back(node);
        if (static_cast<int>(node_collisions.size()) > kAllowedNodeCollisions)
          break;
        continue;
      }

      // If we hit the node that is already processed (by our batch or in
      // another thread) stop gathering and process smaller batch.
      if (!node) break;

      nodes_to_process.push_back(node);
      // If node is already known as terminal (win/lose/draw according to rules
      // of the game), it means that we already visited this node before.
      if (node->IsTerminal()) continue;

      ExtendNode(node, history);

      // If node turned out to be a terminal one, no need to send to NN for
      // evaluation.
      if (!node->IsTerminal()) {
        AddNodeToCompute(node, &computation, history);
      }
    }

    // TODO(mooskagh) Remove prefetch into cache if node collisions work well.
    // If there are requests to NN, but the batch is not full, try to prefetch
    // nodes which are likely useful in future.
    if (computation.GetCacheMisses() > 0 &&
        computation.GetCacheMisses() < kMaxPrefetchBatch) {
      history.Trim(played_history_.GetLength());
      SharedMutex::SharedLock lock(nodes_mutex_);
      PrefetchIntoCache(root_node_,
                        kMaxPrefetchBatch - computation.GetCacheMisses(),
                        &computation, &history);
    }

    // Evaluate nodes through NN.
    if (computation.GetBatchSize() != 0) {
      computation.ComputeBlocking();

      int idx_in_computation = 0;
      for (Node* node : nodes_to_process) {
        if (node->IsTerminal()) continue;
        // Populate Q value.
        node->SetV(-computation.GetQVal(idx_in_computation));
        // Populate P values.
        float total = 0.0;
        for (Node* n : node->Children()) {
          float p = computation.GetPVal(idx_in_computation,
                                        n->GetMove().as_nn_index());
          if (kPolicySoftmaxTemp != 1.0f) {
            p = pow(p, 1 / kPolicySoftmaxTemp);
          }
          total += p;
          n->SetP(p);
        }
        // Scale P values to add up to 1.0.
        if (total > 0.0f) {
          float scale = 1.0f / total;
          for (Node* n : node->Children()) n->SetP(n->GetP() * scale);
        }
        // Add Dirichlet noise if enabled and at root.
        if (kNoise && node == root_node_) {
          ApplyDirichletNoise(node, 0.25, 0.3);
        }
        ++idx_in_computation;
      }
    }

    {
      // Update nodes.
      SharedMutex::Lock lock(nodes_mutex_);
      for (Node* node : nodes_to_process) {
        float v = node->GetV();
        // Maximum depth the node is explored.
        uint16_t depth = 0;
        // If the node is terminal, mark it as fully explored to an infinite
        // depth.
        uint16_t cur_full_depth = node->IsTerminal() ? 999 : 0;
        bool full_depth_updated = true;
        for (Node* n = node; n != root_node_->GetParent(); n = n->GetParent()) {
          ++depth;
          n->FinalizeScoreUpdate(v);
          // Q will be flipped for opponent.
          v = -v;

          // Updating stats.
          // Max depth.
          n->UpdateMaxDepth(depth);
          // Full depth.
          if (full_depth_updated)
            full_depth_updated = n->UpdateFullDepth(&cur_full_depth);
          // Best move.
          if (n->GetParent() == root_node_) {
            if (!best_move_node_ || best_move_node_->GetN() < n->GetN()) {
              best_move_node_ = n;
            }
          }
        }
      }
      total_playouts_ += nodes_to_process.size();

      // Remove virtual loss from node collisions.
      for (Node* node : node_collisions) {
        for (node = node->GetParent(); node != root_node_->GetParent();
             node = node->GetParent()) {
          node->CancelScoreUpdate();
        }
      }
    }
    UpdateRemainingMoves();  // Update remaining moves using smart pruning.
    MaybeOutputInfo();
    MaybeTriggerStop();

    // If required to stop, stop.
    {
      Mutex::Lock lock(counters_mutex_);
      if (stop_) break;
    }
    if (nodes_to_process.empty()) {
      // If this thread had no work, sleep for some milliseconds.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}  // namespace lczero

// Prefetches up to @budget nodes into cache. Returns number of nodes
// prefetched.
int Search::PrefetchIntoCache(Node* node, int budget,
                              CachingComputation* computation,
                              PositionHistory* history) {
  if (budget <= 0) return 0;

  // We are in a leaf, which is not yet being processed.
  if (node->GetNStarted() == 0) {
    if (AddNodeToCompute(node, computation, *history, false)) {
      // Make it return 0 to make it not use the slot, so that the function
      // tries hard to find something to cache even among unpopular moves.
      // In practice that slows things down a lot though, as it's not always
      // easy to find what to cache.
      return 1;
    }
    return 1;
  }

  // If it's a node in progress of expansion or is terminal, not prefetching.
  if (!node->HasChildren()) return 0;

  // Populate all subnodes and their scores.
  typedef std::pair<float, Node*> ScoredNode;
  std::vector<ScoredNode> scores;
  float factor = kCpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  // FPU reduction is not taken into account.
  const float parent_q = -node->GetQ(0, kExtraVirtualLoss);
  for (Node* iter : node->Children()) {
    if (iter->GetP() == 0.0f) continue;
    // Flipping sign of a score to be able to easily sort.
    scores.emplace_back(
        -factor * iter->GetU() - iter->GetQ(parent_q, kExtraVirtualLoss), iter);
  }

  size_t first_unsorted_index = 0;
  int total_budget_spent = 0;
  int budget_to_spend = budget;  // Initializing for the case there's only
                                 // on child.
  for (size_t i = 0; i < scores.size(); ++i) {
    if (budget <= 0) break;

    // Sort next chunk of a vector. 3 of a time. Most of the times it's fine.
    if (first_unsorted_index != scores.size() &&
        i + 2 >= first_unsorted_index) {
      const int new_unsorted_index =
          std::min(scores.size(), budget < 2 ? first_unsorted_index + 2
                                             : first_unsorted_index + 3);
      std::partial_sort(scores.begin() + first_unsorted_index,
                        scores.begin() + new_unsorted_index, scores.end());
      first_unsorted_index = new_unsorted_index;
    }

    Node* n = scores[i].second;
    // Last node gets the same budget as prev-to-last node.
    if (i != scores.size() - 1) {
      // Sign of the score was flipped for sorting, flipping back.
      const float next_score = -scores[i + 1].first;
      const float q = n->GetQ(-parent_q, kExtraVirtualLoss);
      if (next_score > q) {
        budget_to_spend = std::min(
            budget,
            int(n->GetP() * factor / (next_score - q) - n->GetNStarted()) + 1);
      } else {
        budget_to_spend = budget;
      }
    }
    history->Append(n->GetMove());
    const int budget_spent =
        PrefetchIntoCache(n, budget_to_spend, computation, history);
    history->Pop();
    budget -= budget_spent;
    total_budget_spent += budget_spent;
  }
  return total_budget_spent;
}

Node* Search::GetBestChild(Node* parent) const {
  Node* best_node = nullptr;
  // Best child is selected using the following criteria:
  // * Largest number of playouts.
  // * If two nodes have equal number:
  //   * If that number is 0, the one with larger prior wins.
  //   * If that number is larger than 0, the one wil larger eval wins.
  std::tuple<int, float, float> best(-1, 0.0, 0.0);
  for (Node* node : parent->Children()) {
    if (parent == root_node_ && !limits_.searchmoves.empty() &&
        std::find(limits_.searchmoves.begin(), limits_.searchmoves.end(),
                  node->GetMove()) == limits_.searchmoves.end()) {
      continue;
    }
    std::tuple<int, float, float> val(node->GetN(), node->GetQ(-10.0, 0.0),
                                      node->GetP());
    if (val > best) {
      best = val;
      best_node = node;
    }
  }
  return best_node;
}

Node* Search::GetBestChildWithTemperature(Node* parent,
                                          float temperature) const {
  std::vector<float> cumulative_sums;
  float sum = 0.0;
  const float n_parent = parent->GetN();

  for (Node* node : parent->Children()) {
    if (parent == root_node_ && !limits_.searchmoves.empty() &&
        std::find(limits_.searchmoves.begin(), limits_.searchmoves.end(),
                  node->GetMove()) == limits_.searchmoves.end()) {
      continue;
    }
    sum += std::pow(node->GetN() / n_parent, 1 / temperature);
    cumulative_sums.push_back(sum);
  }

  float toss = Random::Get().GetFloat(cumulative_sums.back());
  int idx =
      std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) -
      cumulative_sums.begin();

  for (Node* node : parent->Children()) {
    if (parent == root_node_ && !limits_.searchmoves.empty() &&
        std::find(limits_.searchmoves.begin(), limits_.searchmoves.end(),
                  node->GetMove()) == limits_.searchmoves.end()) {
      continue;
    }
    if (idx-- == 0) return node;
  }
  assert(false);
  return nullptr;
}

void Search::SendUciInfo() REQUIRES(nodes_mutex_) {
  if (!best_move_node_) return;
  last_outputted_best_move_node_ = best_move_node_;
  uci_info_.depth = root_node_->GetFullDepth();
  uci_info_.seldepth = root_node_->GetMaxDepth();
  uci_info_.time = GetTimeSinceStart();
  uci_info_.nodes = total_playouts_ + initial_visits_;
  uci_info_.hashfull =
      cache_->GetSize() * 1000LL / std::max(cache_->GetCapacity(), 1);
  uci_info_.nps =
      uci_info_.time ? (total_playouts_ * 1000 / uci_info_.time) : 0;
  uci_info_.score =
      290.680623072 * tan(1.548090806 * best_move_node_->GetQ(0, 0));
  uci_info_.pv.clear();

  bool flip = played_history_.IsBlackToMove();
  for (Node* iter = best_move_node_; iter;
       iter = GetBestChild(iter), flip = !flip) {
    uci_info_.pv.push_back(iter->GetMove(flip));
  }
  uci_info_.comment.clear();
  info_callback_(uci_info_);
}

// Decides whether anything important changed in stats and new info should be
// shown to a user.
void Search::MaybeOutputInfo() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  if (!responded_bestmove_ && best_move_node_ &&
      (best_move_node_ != last_outputted_best_move_node_ ||
       uci_info_.depth != root_node_->GetFullDepth() ||
       uci_info_.seldepth != root_node_->GetMaxDepth() ||
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
  std::vector<const Node*> nodes;
  const float parent_q =
      -root_node_->GetQ(0, 0) -
      kFpuReduction * std::sqrt(root_node_->GetVisitedPolicy());
  for (Node* iter : root_node_->Children()) {
    nodes.emplace_back(iter);
  }
  std::sort(nodes.begin(), nodes.end(),
            [](const Node* a, const Node* b) { return a->GetN() < b->GetN(); });

  const bool is_black_to_move = played_history_.IsBlackToMove();
  ThinkingInfo info;
  for (const Node* node : nodes) {
    std::ostringstream oss;
    oss << std::fixed;
    oss << std::left << std::setw(5)
        << node->GetMove(is_black_to_move).as_string();
    oss << " (" << std::setw(4) << node->GetMove().as_nn_index() << ")";
    oss << " N: ";
    oss << std::right << std::setw(7) << node->GetN() << " (+" << std::setw(2)
        << node->GetNInFlight() << ") ";
    oss << "(V: " << std::setw(6) << std::setprecision(2) << node->GetV() * 100
        << "%) ";
    oss << "(P: " << std::setw(5) << std::setprecision(2) << node->GetP() * 100
        << "%) ";
    oss << "(Q: " << std::setw(8) << std::setprecision(5)
        << node->GetQ(parent_q, 0) << ") ";
    oss << "(U: " << std::setw(6) << std::setprecision(5)
        << node->GetU() * kCpuct *
               std::sqrt(std::max(node->GetParent()->GetChildrenVisits(), 1u))
        << ") ";

    oss << "(Q+U: " << std::setw(8) << std::setprecision(5)
        << node->GetQ(parent_q, 0) +
               node->GetU() * kCpuct *
                   std::sqrt(
                       std::max(node->GetParent()->GetChildrenVisits(), 1u))
        << ") ";
    info.comment = oss.str();
    info_callback_(info);
  }
}

void Search::MaybeTriggerStop() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  // Don't stop when the root node is not yet expanded.
  if (total_playouts_ == 0) return;
  // If smart pruning tells to stop (best move found), stop.
  if (found_best_move_) {
    stop_ = true;
  }
  // Stop if reached playouts limit.
  if (limits_.playouts >= 0 && total_playouts_ >= limits_.playouts) {
    stop_ = true;
  }
  // Stop if reached visits limit.
  if (limits_.visits >= 0 &&
      total_playouts_ + initial_visits_ >= limits_.visits) {
    stop_ = true;
  }
  // Stop if reached time limit.
  if (limits_.time_ms >= 0 && GetTimeSinceStart() >= limits_.time_ms) {
    stop_ = true;
  }
  // If we are the first to see that stop is needed.
  if (stop_ && !responded_bestmove_) {
    SendUciInfo();
    if (kVerboseStats) SendMovesStats();
    best_move_ = GetBestMoveInternal();
    best_move_callback_({best_move_.first, best_move_.second});
    responded_bestmove_ = true;
    best_move_node_ = nullptr;
  }
}

void Search::UpdateRemainingMoves() {
  if (!kSmartPruning) return;
  SharedMutex::Lock lock(nodes_mutex_);
  remaining_playouts_ = std::numeric_limits<int>::max();
  // Check for how many playouts there is time remaining.
  if (limits_.time_ms >= 0) {
    auto time_since_start = GetTimeSinceStart();
    if (time_since_start > kSmartPruningToleranceMs) {
      auto nps = (1000LL * total_playouts_ + kSmartPruningToleranceNodes) /
                     (time_since_start - kSmartPruningToleranceMs) +
                 1;
      int64_t remaining_time = limits_.time_ms - time_since_start;
      int64_t remaining_playouts = remaining_time * nps / 1000;
      // Don't assign directly to remaining_playouts_ as overflow is possible.
      if (remaining_playouts < remaining_playouts_)
        remaining_playouts_ = remaining_playouts;
    }
  }
  // Check how many visits are left.
  if (limits_.visits >= 0) {
    // Adding kMiniBatchSize, as it's possible to exceed visits limit by that
    // number.
    auto remaining_visits =
        limits_.visits - total_playouts_ - initial_visits_ + kMiniBatchSize - 1;

    if (remaining_visits < remaining_playouts_)
      remaining_playouts_ = remaining_visits;
  }
  if (limits_.playouts >= 0) {
    // Adding kMiniBatchSize, as it's possible to exceed visits limit by that
    // number.
    auto remaining_playouts =
        limits_.visits - total_playouts_ + kMiniBatchSize + 1;
    if (remaining_playouts < remaining_playouts_)
      remaining_playouts_ = remaining_playouts;
  }
  // Even if we exceeded limits, don't go crazy by not allowing any playouts.
  if (remaining_playouts_ <= 1) remaining_playouts_ = 1;
}

void Search::ExtendNode(Node* node, const PositionHistory& history) {
  // Not taking mutex because other threads will see that N=0 and N-in-flight=1
  // and will not touch this node.
  const auto& board = history.Last().GetBoard();
  auto legal_moves = board.GenerateLegalMoves();

  // Check whether it's a draw/lose by rules.
  if (legal_moves.empty()) {
    // Checkmate or stalemate.
    if (board.IsUnderCheck()) {
      // Checkmate.
      node->MakeTerminal(GameResult::WHITE_WON);
    } else {
      // Stalemate.
      node->MakeTerminal(GameResult::DRAW);
    }
    return;
  }

  // If it's root node and we're asked to think, pretend there's no draw.
  if (node != root_node_) {
    if (!board.HasMatingMaterial()) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    if (history.Last().GetNoCapturePly() >= 100) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    if (history.Last().GetRepetitions() >= 2) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }
  }

  // Add legal moves as children to this node.
  for (const auto& move : legal_moves) node->CreateChild(move);
}

// Returns node and whether it should be processed.
// (false if it is a collision).
std::pair<Node*, bool> Search::PickNodeToExtend(Node* node,
                                                PositionHistory* history) {
  // Fetch the current best root node visits for possible smart pruning.
  int best_node_n = 0;
  {
    SharedMutex::Lock lock(nodes_mutex_);
    if (best_move_node_) best_node_n = best_move_node_->GetN();
  }

  // True on first iteration, false as we dive deeper.
  bool is_root_node = true;
  while (true) {
    {
      SharedMutex::Lock lock(nodes_mutex_);
      // Check whether we are in the leave.
      if (!node->TryStartScoreUpdate()) return {node, false};
      // Found leave, and we are the the first to visit it.
      if (!node->HasChildren()) return {node, true};
    }

    // Now we are not in leave, we need to go deeper.
    SharedMutex::SharedLock lock(nodes_mutex_);
    float factor = kCpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
    float best = -100.0f;
    int possible_moves = 0;
    float parent_q =
        (is_root_node && kNoise)
            ? -node->GetQ(0, kExtraVirtualLoss)
            : -node->GetQ(0, kExtraVirtualLoss) -
                  kFpuReduction * std::sqrt(node->GetVisitedPolicy());
    for (Node* iter : node->Children()) {
      if (is_root_node) {
        // If there's no chance to catch up the currently best node with
        // remaining playouts, not consider it.
        // best_move_node_ can change since best_node_n computation.
        // To ensure we have at least one node to expand, always include
        // current best node.
        if (iter != best_move_node_ &&
            remaining_playouts_ <
                best_node_n - static_cast<int>(iter->GetN())) {
          continue;
        }
        // If searchmoves was sent, restrict the search only in that moves
        if (!limits_.searchmoves.empty() &&
            std::find(limits_.searchmoves.begin(),
                      limits_.searchmoves.end(),
                      iter->GetMove()) == limits_.searchmoves.end()) {
            continue;
        }
        ++possible_moves;
      }
      float Q = iter->GetQ(parent_q, kExtraVirtualLoss);
      if (kVirtualLossBug && iter->GetN() == 0) {
        Q = (Q * iter->GetParent()->GetN() - kVirtualLossBug) /
            (iter->GetParent()->GetN() + std::fabs(kVirtualLossBug));
      }
      const float score = factor * iter->GetU() + Q;
      if (score > best) {
        best = score;
        node = iter;
      }
    }
    history->Append(node->GetMove());
    if (is_root_node && possible_moves <= 1 && !limits_.infinite) {
      // If there is only one move theoretically possible within remaining time,
      // output it.
      Mutex::Lock counters_lock(counters_mutex_);
      found_best_move_ = true;
    }
    is_root_node = false;
  }
}

std::pair<Move, Move> Search::GetBestMove() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  return GetBestMoveInternal();
}

std::pair<Move, Move> Search::GetBestMoveInternal() const
    REQUIRES_SHARED(nodes_mutex_) REQUIRES_SHARED(counters_mutex_) {
  if (responded_bestmove_) return best_move_;
  if (!root_node_->HasChildren()) return {};

  float temperature = kTemperature;
  if (temperature && kTempDecayMoves) {
    int moves = played_history_.Last().GetGamePly() / 2;
    if (moves >= kTempDecayMoves) {
      temperature = 0.0;
    } else {
      temperature *=
          static_cast<float>(kTempDecayMoves - moves) / kTempDecayMoves;
    }
  }

  Node* best_node = temperature && root_node_->GetN() > 1
                        ? GetBestChildWithTemperature(root_node_, temperature)
                        : GetBestChild(root_node_);

  Move ponder_move;
  /*  // Doesn't seem to work for now, so disabling.
  if (best_node->HasChildren()) {
    ponder_move =
        GetBestChild(best_node)->GetMove(!played_history_.IsBlackToMove());
  }
  */
  return {best_node->GetMove(played_history_.IsBlackToMove()), ponder_move};
}

void Search::StartThreads(size_t how_many) {
  Mutex::Lock lock(threads_mutex_);
  while (threads_.size() < how_many) {
    threads_.emplace_back([&]() { Worker(); });
  }
}

void Search::RunSingleThreaded() { Worker(); }

void Search::RunBlocking(size_t threads) {
  if (threads == 1) {
    Worker();
  } else {
    StartThreads(threads);
    Wait();
  }
}

void Search::Stop() {
  Mutex::Lock lock(counters_mutex_);
  stop_ = true;
}

void Search::Abort() {
  Mutex::Lock lock(counters_mutex_);
  responded_bestmove_ = true;
  stop_ = true;
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

}  // namespace lczero

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
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "mcts/node.h"
#include "neural/cache.h"
#include "utils/random.h"

namespace lczero {

namespace {
const char* kMiniBatchSizeStr = "Minibatch size for NN inference";
const char* kMiniPrefetchBatchStr = "Max prefetch nodes, per NN call";
const char* kAggresiveCachingStr = "Try hard to find what to cache";
const char* kCpuctStr = "Cpuct MCTS option";
const char* kTemperatureStr = "Initial temperature";
const char* kTempDecayStr = "Per move temperature decay";
const char* kNoiseStr = "Add Dirichlet noise at root node";
const char* kVerboseStatsStr = "Display verbose move stats";
}  // namespace

void Search::PopulateUciParams(OptionsParser* options) {
  options->Add<IntOption>(kMiniBatchSizeStr, 1, 1024, "minibatch-size") = 128;
  options->Add<IntOption>(kMiniPrefetchBatchStr, 0, 1024, "max-prefetch") = 32;
  options->Add<BoolOption>(kAggresiveCachingStr, "aggressive-caching") = false;
  options->Add<FloatOption>(kCpuctStr, 0, 100, "cpuct") = 1.7;
  options->Add<FloatOption>(kTemperatureStr, 0, 100, "temperature", 'm') = 0.0;
  options->Add<FloatOption>(kTempDecayStr, 0, 1.00, "tempdecay") = 0.0;
  options->Add<BoolOption>(kNoiseStr, "noise", 'n') = false;
  options->Add<BoolOption>(kVerboseStatsStr, "verbose-move-stats") = false;
}

Search::Search(Node* root_node, NodePool* node_pool, Network* network,
               BestMoveInfo::Callback best_move_callback,
               ThinkingInfo::Callback info_callback, const SearchLimits& limits,
               const OptionsDict& options, NNCache* cache)
    : root_node_(root_node),
      node_pool_(node_pool),
      cache_(cache),
      network_(network),
      limits_(limits),
      start_time_(std::chrono::steady_clock::now()),
      initial_visits_(root_node->n),
      best_move_callback_(best_move_callback),
      info_callback_(info_callback),
      kMiniBatchSize(options.Get<int>(kMiniBatchSizeStr)),
      kMiniPrefetchBatch(options.Get<int>(kMiniPrefetchBatchStr)),
      kAggresiveCaching(options.Get<bool>(kAggresiveCachingStr)),
      kCpuct(options.Get<float>(kCpuctStr)),
      kTemperature(options.Get<float>(kTemperatureStr)),
      kTempDecay(options.Get<float>(kTempDecayStr)),
      kNoise(options.Get<bool>(kNoiseStr)),
      kVerboseStats(options.Get<bool>(kVerboseStatsStr)) {}

// Returns whether node was already in cache.
bool Search::AddNodeToCompute(Node* node, CachingComputation* computation,
                              bool add_if_cached) {
  auto hash = node->BoardHash();
  // If already in cache, no need to do anything.
  if (add_if_cached) {
    if (computation->AddInputByHash(hash)) return true;
  } else {
    if (cache_->ContainsKey(hash)) return true;
  }
  auto planes = node->EncodeForNN();

  std::vector<uint16_t> moves;

  if (node->child) {
    // Valid moves are known, using them.
    for (Node* iter = node->child; iter; iter = iter->sibling) {
      moves.emplace_back(iter->move.as_nn_index());
    }
  } else {
    // Cache pseudovalid moves. A bit of a waste, but faster.
    const auto& pseudovalid_moves = node->board.GeneratePseudovalidMoves();
    moves.reserve(pseudovalid_moves.size());
    for (const Move& m : pseudovalid_moves) {
      moves.emplace_back(m.as_nn_index());
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
  for (Node* iter = node->child; iter; iter = iter->sibling) {
    float eta = Random::Get().GetGamma(alpha, 1.0);
    noise.emplace_back(eta);
    total += eta;
  }

  if (total < std::numeric_limits<float>::min()) return;

  int noise_idx = 0;
  for (Node* iter = node->child; iter; iter = iter->sibling) {
    iter->p = iter->p * (1 - eps) + noise[noise_idx++] / total * eps;
  }
}
}  // namespace

void Search::Worker() {
  std::vector<Node*> nodes_to_process;

  // Exit check is at the end of the loop as at least one iteration is
  // necessary.
  while (true) {
    nodes_to_process.clear();
    auto computation = CachingComputation(network_->NewComputation(), cache_);

    // Gather nodes to process in the current batch.
    for (int i = 0; i < kMiniBatchSize; ++i) {
      // If there's something to do without touching slow neural net, do it.
      if (i > 0 && computation.GetCacheMisses() == 0) break;
      Node* node = PickNodeToExtend(root_node_);
      // If we hit the node that is already processed (by our batch or in
      // another thread) stop gathering and process smaller batch.
      if (!node) break;

      nodes_to_process.push_back(node);
      // If node is already known as terminal (win/lose/draw according to rules
      // of the game), it means that we already visited this node before.
      if (node->is_terminal) continue;

      ExtendNode(node);

      // If node turned out to be a terminal one, no need to send to NN for
      // evaluation.
      if (!node->is_terminal) {
        AddNodeToCompute(node, &computation);
      }
    }

    // If there are requests to NN, but the batch is not full, try to prefetch
    // nodes which are likely useful in future.
    if (computation.GetCacheMisses() > 0 &&
        computation.GetCacheMisses() < kMiniPrefetchBatch) {
      SharedMutex::SharedLock lock(nodes_mutex_);
      PrefetchIntoCache(root_node_,
                        kMiniPrefetchBatch - computation.GetCacheMisses(),
                        &computation);
    }

    // Evaluate nodes through NN.
    if (computation.GetBatchSize() != 0) {
      computation.ComputeBlocking();

      int idx_in_computation = 0;
      for (Node* node : nodes_to_process) {
        if (node->is_terminal) continue;
        // Populate Q value.
        node->v = -computation.GetQVal(idx_in_computation);
        // Populate P values.
        float total = 0.0;
        for (Node* n = node->child; n; n = n->sibling) {
          float p =
              computation.GetPVal(idx_in_computation, n->move.as_nn_index());
          total += p;
          n->p = p;
        }
        // Scale P values to add up to 1.0.
        if (total > 0.0f) {
          for (Node* n = node->child; n; n = n->sibling) {
            n->p /= total;
          }
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
        float v = node->v;
        // Maximum depth the node is explored.
        uint16_t depth = 0;
        // If the node is terminal, mark it as fully explored to an infinite
        // depth.
        uint16_t cur_full_depth = node->is_terminal ? 999 : 0;
        bool full_depth_updated = true;
        for (Node* n = node; n != root_node_->parent; n = n->parent) {
          ++depth;
          // Add new value to W.
          n->w += v;
          // Increment N.
          ++n->n;
          // Decrement virtual loss.
          --n->n_in_flight;
          // Recompute Q.
          n->q = n->w / n->n;
          // Q will be flipped for opponent.
          v = -v;

          // Updating stats.
          // Max depth.
          if (depth > n->max_depth) {
            n->max_depth = depth;
          }
          // Full depth.
          if (full_depth_updated && n->full_depth <= cur_full_depth) {
            for (Node* iter = n->child; iter; iter = iter->sibling) {
              if (cur_full_depth > iter->full_depth) {
                cur_full_depth = iter->full_depth;
              }
            }
            if (cur_full_depth >= n->full_depth) {
              n->full_depth = ++cur_full_depth;
            } else {
              full_depth_updated = false;
            }
          }
          // Best move.
          if (n->parent == root_node_) {
            if (!best_move_node_ || best_move_node_->n < n->n) {
              best_move_node_ = n;
            }
          }
        }
      }
      total_playouts_ += nodes_to_process.size();
    }
    MaybeOutputInfo();
    MaybeTriggerStop();

    // If required to stop, stop.
    {
      Mutex::Lock lock(counters_mutex_);
      if (stop_) break;
    }
  }
}

// Prefetches up to @budget nodes into cache. Returns number of nodes
// prefetched.
int Search::PrefetchIntoCache(Node* node, int budget,
                              CachingComputation* computation) {
  if (budget <= 0) return 0;

  // We are in a leaf, which is not yet being processed.
  if (node->n + node->n_in_flight == 0) {
    if (AddNodeToCompute(node, computation, false)) {
      return kAggresiveCaching ? 0 : 1;
    }
    return 1;
  }

  // If it's a node in progress of expansion or is terminal, not prefetching.
  if (!node->child) return 0;

  // Populate all subnodes and their scores.
  typedef std::pair<float, Node*> ScoredNode;
  std::vector<ScoredNode> scores;
  float factor = kCpuct * std::sqrt(std::max(node->n, 1u));
  for (Node* iter = node->child; iter; iter = iter->sibling) {
    if (iter->p == 0.0f) continue;
    // Flipping sign of a score to be able to easily sort.
    scores.emplace_back(-factor * iter->ComputeU() - iter->ComputeQ(), iter);
  }

  int first_unsorted_index = 0;
  int total_budget_spent = 0;
  int budget_to_spend = budget;  // Initializing for the case there's only
                                 // on child.
  for (int i = 0; i < scores.size(); ++i) {
    if (budget <= 0) break;

    // Sort next chunk of a vector. 3 of a time. Most of the times it's fine.
    if (first_unsorted_index != scores.size() &&
        i + 2 >= first_unsorted_index) {
      const int new_unsorted_index = std::min(
          static_cast<int>(scores.size()),
          budget < 2 ? first_unsorted_index + 2 : first_unsorted_index + 3);
      std::partial_sort(scores.begin() + first_unsorted_index,
                        scores.begin() + new_unsorted_index, scores.end());
      first_unsorted_index = new_unsorted_index;
    }

    Node* n = scores[i].second;
    // Last node gets the same budget as prev-to-last node.
    if (i != scores.size() - 1) {
      // Sign of the score was flipped for sorting, flipping back.
      const float next_score = -scores[i + 1].first;
      const float q = n->ComputeQ();
      if (next_score > q) {
        budget_to_spend = std::min(
            budget,
            int(n->p * factor / (next_score - q) - n->n - n->n_in_flight) + 1);
      } else {
        budget_to_spend = budget;
      }
    }
    const int budget_spent = PrefetchIntoCache(n, budget_to_spend, computation);
    budget -= budget_spent;
    total_budget_spent += budget_spent;
  }
  return total_budget_spent;
}

namespace {
// Returns a child with most visits.
Node* GetBestChild(Node* parent) {
  Node* best_node = nullptr;
  std::pair<int, float> best(-1, 0.0);
  for (Node* node = parent->child; node; node = node->sibling) {
    std::pair<int, float> val(node->n + node->n_in_flight, node->p);
    if (val > best) {
      best = val;
      best_node = node;
    }
  }
  return best_node;
}

Node* GetBestChildWithTemperature(Node* parent, float temperature) {
  std::vector<double> cumulative_sums;
  double sum = 0.0;

  for (Node* node = parent->child; node; node = node->sibling) {
    int n = node->n + node->n_in_flight;
    sum += std::pow(n, 1 / temperature);
    cumulative_sums.push_back(sum);
  }

  double toss = Random::Get().GetDouble(cumulative_sums.back());
  int idx =
      std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) -
      cumulative_sums.begin();

  for (Node* node = parent->child; node; node = node->sibling) {
    if (idx-- == 0) return node;
  }
  assert(false);
}
}  // namespace

void Search::SendUciInfo() REQUIRES(nodes_mutex_) {
  if (!best_move_node_) return;
  last_outputted_best_move_node_ = best_move_node_;
  uci_info_.depth = root_node_->full_depth;
  uci_info_.seldepth = root_node_->max_depth;
  uci_info_.time = GetTimeSinceStart();
  uci_info_.nodes = total_playouts_ + initial_visits_;
  uci_info_.hashfull =
      cache_->GetSize() * 1000LL / std::max(cache_->GetCapacity(), 1);
  uci_info_.nps =
      uci_info_.time ? (total_playouts_ * 1000 / uci_info_.time) : 0;
  uci_info_.score = -191 * log(2 / (best_move_node_->q * 0.99 + 1) - 1);
  uci_info_.pv.clear();

  for (Node* iter = best_move_node_; iter; iter = GetBestChild(iter)) {
    uci_info_.pv.push_back(iter->GetMoveAsWhite());
  }
  uci_info_.comment.clear();
  info_callback_(uci_info_);
}

// Decides whether anything important changed in stats and new info should be
// shown to a user.
void Search::MaybeOutputInfo() {
  SharedMutex::Lock lock(nodes_mutex_);
  if (best_move_node_ && (best_move_node_ != last_outputted_best_move_node_ ||
                          uci_info_.depth != root_node_->full_depth ||
                          uci_info_.seldepth != root_node_->max_depth)) {
    SendUciInfo();
  }
}

uint64_t Search::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

void Search::SendMovesStats() const {
  std::vector<const Node*> nodes;
  for (Node* iter = root_node_->child; iter; iter = iter->sibling) {
    nodes.emplace_back(iter);
  }
  std::sort(nodes.begin(), nodes.end(),
            [](const Node* a, const Node* b) { return a->n < b->n; });

  ThinkingInfo info;
  for (const Node* node : nodes) {
    std::ostringstream oss;
    oss << std::fixed;
    oss << std::left << std::setw(5) << node->GetMoveAsWhite().as_string();
    oss << " -> ";
    oss << std::right << std::setw(7) << node->n << " (+" << std::setw(2)
        << node->n_in_flight << ") ";
    oss << "(V: " << std::setw(6) << std::setprecision(2) << node->v * 100
        << "%) ";
    oss << "(P: " << std::setw(5) << std::setprecision(2) << node->p * 100
        << "%) ";
    oss << "(Q: " << std::setw(8) << std::setprecision(5) << node->ComputeQ()
        << ") ";
    oss << "(U: " << std::setw(6) << std::setprecision(5)
        << node->ComputeU() * kCpuct * std::sqrt(std::max(node->parent->n, 1u))
        << ") ";

    oss << "(Q+U: " << std::setw(8) << std::setprecision(5)
        << node->ComputeQ() + node->ComputeU() * kCpuct *
                                  std::sqrt(std::max(node->parent->n, 1u))
        << ") ";
    info.comment = oss.str();
    info_callback_(info);
  }
}

void Search::MaybeTriggerStop() {
  Mutex::Lock lock(counters_mutex_);
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  // Don't stop when the root node is not yet expanded.
  if (total_playouts_ == 0) return;
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

void Search::ExtendNode(Node* node) {
  // Not taking mutex because other threads will see that N=0 and N-in-flight=1
  // and will not touch this node.
  auto& board = node->board;
  auto valid_moves = board.GenerateValidMoves();

  // Check whether it's a draw/lose by rules.
  if (valid_moves.empty()) {
    // Checkmate or stalemate.
    node->is_terminal = true;
    if (board.IsUnderCheck()) {
      // Checkmate.
      node->v = 1.0f;
    } else {
      // Stalemate.
      node->v = 0.0f;
    }
    return;
  }

  if (!board.HasMatingMaterial()) {
    node->is_terminal = true;
    node->v = 0.0f;
    return;
  }

  if (node->no_capture_ply >= 100) {
    node->is_terminal = true;
    node->v = 0.0f;
    return;
  }

  node->repetitions = node->ComputeRepetitions();
  if (node->repetitions >= 2) {
    node->is_terminal = true;
    node->v = 0.0f;
    return;
  }

  // Add valid moves as children to this node.
  Node* prev_node = node;
  for (const auto& move : valid_moves) {
    Node* new_node = node_pool_->GetNode();

    new_node->parent = node;
    if (prev_node == node) {
      node->child = new_node;
    } else {
      prev_node->sibling = new_node;
    }

    new_node->move = move.move;
    new_node->board = move.board;
    new_node->board.Mirror();
    new_node->no_capture_ply =
        move.reset_50_moves ? 0 : (node->no_capture_ply + 1);
    new_node->ply_count = node->ply_count + 1;
    prev_node = new_node;
  }
}

Node* Search::PickNodeToExtend(Node* node) {
  while (true) {
    {
      SharedMutex::Lock lock(nodes_mutex_);
      // Check whether we are in the leave.
      if (node->n == 0 && node->n_in_flight > 0) {
        // The node is currently being processed by another thread.
        // Undo the increments of anschestor nodes, and return null.
        for (node = node->parent; node != root_node_->parent;
             node = node->parent) {
          --node->n_in_flight;
        }
        return nullptr;
      }
      ++node->n_in_flight;
      // Found leave, and we are the the first to visit it.
      if (!node->child) {
        return node;
      }
    }

    // Now we are not in leave, we need to go deeper.
    SharedMutex::SharedLock lock(nodes_mutex_);
    float factor = kCpuct * std::sqrt(std::max(node->n, 1u));
    float best = -100.0f;
    for (Node* iter = node->child; iter; iter = iter->sibling) {
      const float score = factor * iter->ComputeU() + iter->ComputeQ();
      if (score > best) {
        best = score;
        node = iter;
      }
    }
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
  if (!root_node_->child) return {};

  float temperature = kTemperature;
  if (temperature && kTempDecay)
    temperature *= std::pow(1 - kTempDecay, root_node_->ply_count / 2);
  if (temperature < 0.01) temperature = 0.0;

  Node* best_node = temperature
                        ? GetBestChildWithTemperature(root_node_, temperature)
                        : GetBestChild(root_node_);

  Move ponder_move;
  if (best_node->child) {
    ponder_move = GetBestChild(best_node)->GetMoveAsWhite();
  }
  return {best_node->GetMoveAsWhite(), ponder_move};
}

void Search::StartThreads(int how_many) {
  Mutex::Lock lock(threads_mutex_);
  while (threads_.size() < how_many) {
    threads_.emplace_back([&]() { Worker(); });
  }
}

void Search::RunSingleThreaded() { Worker(); }

void Search::RunBlocking(int threads) {
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
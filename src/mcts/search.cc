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
#include "mcts/node.h"

#include <cmath>
#include "neural/network_tf.h"
#include "utils/cache.h"

namespace lczero {

namespace {

const int kDefaultParallelPlayouts = 32;
const char* kParallelPlayoutsOption = "Number of parallel playouts";

const int kDefaultMiniBatchSize = 32;
const char* kMiniBatchSizeOption = "Minibatch size for NN inference";

const int kDefaultCpuct = 170;
const char* kCpuctOption = "Cpuct MCTS option (x100)";

const bool kDefaultPopulateMoves = false;
const char* kPopulateMovesOption = "(oldbug) Populate movecount plane";

const bool kDefaultFlipHistory = true;
const char* kFlipHistoryOption = "(oldbug) Flip opponents history";

const bool kDefaultFlipMove = true;
const char* kFlipMoveOption = "(oldbug) Flip black's moves";

const bool kDefaultDynamicParent = true;
const char* kDynamicParentOption = "Use dynamic parent score for leaves";

}  // namespace

void Search::PopulateUciParams(UciOptions* options) {
  options->Add(std::make_unique<SpinOption>(kMiniBatchSizeOption,
                                            kDefaultMiniBatchSize, 1, 128,
                                            std::function<void(int)>{}));

  options->Add(std::make_unique<SpinOption>(kParallelPlayoutsOption,
                                            kDefaultParallelPlayouts, 1, 128,
                                            std::function<void(int)>{}));

  options->Add(std::make_unique<SpinOption>(kCpuctOption, kDefaultCpuct, 0,
                                            9999, std::function<void(int)>{}));

  options->Add(std::make_unique<CheckOption>(kPopulateMovesOption,
                                             kDefaultPopulateMoves,
                                             std::function<void(bool)>{}));

  options->Add(std::make_unique<CheckOption>(
      kFlipHistoryOption, kDefaultFlipHistory, std::function<void(bool)>{}));

  options->Add(std::make_unique<CheckOption>(kFlipMoveOption, kDefaultFlipMove,
                                             std::function<void(bool)>{}));
  options->Add(std::make_unique<CheckOption>(kDynamicParentOption,
                                             kDefaultDynamicParent,
                                             std::function<void(bool)>{}));
}

Search::Search(Node* root_node, NodePool* node_pool, const Network* network,
               BestMoveInfo::Callback best_move_callback,
               UciInfo::Callback info_callback, const SearchLimits& limits,
               UciOptions* uci_options, NNCache* cache)
    : root_node_(root_node),
      node_pool_(node_pool),
      cache_(cache),
      network_(network),
      limits_(limits),
      start_time_(std::chrono::steady_clock::now()),
      best_move_callback_(best_move_callback),
      info_callback_(info_callback),
      kMiniBatchSize(uci_options
                         ? uci_options->GetIntValue(kMiniBatchSizeOption)
                         : kDefaultMiniBatchSize),
      kParallelPlayouts(uci_options
                            ? uci_options->GetIntValue(kParallelPlayoutsOption)
                            : kDefaultParallelPlayouts),
      kCpuct((uci_options ? uci_options->GetIntValue(kCpuctOption)
                          : kDefaultCpuct) /
             100.0f),
      kPopulateMoves(uci_options
                         ? uci_options->GetBoolValue(kPopulateMovesOption)
                         : kDefaultPopulateMoves),
      kFlipHistory(uci_options ? uci_options->GetBoolValue(kFlipHistoryOption)
                               : kDefaultFlipHistory),
      kFlipMove(uci_options ? uci_options->GetBoolValue(kFlipMoveOption)
                            : kDefaultFlipMove),
      kDynamicParent(uci_options ? uci_options->GetBoolValue(kFlipMoveOption)
                                 : kDefaultDynamicParent) {}

class NodesGatherer {
 public:
  NodesGatherer(Search* search, CachingComputation* computation,
                int extend_budget, int nn_fetch_limit)
      : search_(search),
        extend_budget_(extend_budget),
        nn_fetch_limit_(nn_fetch_limit),
        computation_(computation) {}

  void Gather(Node* node) {
    // TODO(mooskagh) Tune that magic 3 constant.
    GatherInternal(
        node, std::max(extend_budget_,
                       3 * (nn_fetch_limit_ - computation_->GetCacheMisses())));
  }

  const std::vector<Node*>& GetNodesToProcess() const {
    return nodes_to_process_;
  }

  int GetFreshNodes() const { return fresh_nodes_; }

 private:
  int GatherInternal(Node* node, int budget) {
    // No more nodes needed, leaving.
    if (budget <= 0 || ShouldStop()) return 0;

    if (!node->child) {
      // Found leaf.
      return MaybeAddLeaf(node);
    }

    // Node has children.
    // Gathering Nodes and their scores.
    typedef std::pair<float, Node*> ScoredNode;
    std::vector<ScoredNode> scores;

    // Populate all subnodes and their scores.
    float factor = search_->kCpuct * std::sqrt(node->n + 1);
    for (Node* iter = node->child; iter; iter = iter->sibling) {
      const float u = factor * iter->p / (1 + iter->n + iter->m);
      const float q = (iter->n ? iter->q
                               : (search_->kDynamicParent ? -iter->parent->q
                                                          : -iter->parent->v));
      const float v = u + q;
      scores.emplace_back(v, iter);
    }

    // Sort highest score first.
    // TODO(mooskagh) Sort partially as an optimization.
    std::sort(scores.rbegin(), scores.rend());

    int total_visits = 0;
    for (std::vector<ScoredNode>::iterator iter = scores.begin(),
                                           end = scores.end();
         iter != end; ++iter) {
      Node* n = iter->second;
      auto next = std::next(iter);
      const float next_v = next == end ? -100.f : next->first;
      const float q =
          (n->n ? n->q
                : (search_->kDynamicParent ? -n->parent->q : -n->parent->v));
      int budget = std::max(extend_budget_,
                            nn_fetch_limit_ - computation_->GetCacheMisses());
      if (q < next_v) {
        budget = std::min(budget,
                          int(n->p * factor / (next_v - q) - n->n - n->m) + 1);
      }  // else even visiting it infinitely won't change anything.
      int increment = GatherInternal(n, budget);
      node->m += increment;
      total_visits += increment;
      if (increment != budget) StopExtending();
      if (ShouldStop()) break;
    }
    return total_visits;
  }

  // Returns whether node was already in cache.
  bool AddNodeToCompute(Node* node) {
    auto hash = node->BoardHash();
    // If already in cache, no need to do anything.
    if (computation_->AddInputByHash(hash)) return true;
    auto planes = search_->EncodeNode(node);

    std::vector<uint16_t> pseudovalid_moves;
    bool flip = (search_->kFlipMove && node->board.flipped());

    if (node->child) {
      // Valid moves are known, using them.
      for (Node* iter = node->child; iter; iter = iter->sibling) {
        Move m = iter->move;
        if (flip) m.Mirror();
        pseudovalid_moves.emplace_back(m.as_nn_index());
      }
    } else {
      // Cache pseudovalid moves. A bit of a waste, but faster.
      for (Move m : node->board.GeneratePseudovalidMoves()) {
        if (flip) m.Mirror();
        pseudovalid_moves.emplace_back(m.as_nn_index());
      }
    }

    computation_->AddInput(hash, std::move(planes),
                           std::move(pseudovalid_moves));
    return false;
  }

  int MaybeAddLeaf(Node* node) {
    // Someone is already processing this leaf (in different thread), leaving.
    if (node->n == 0 && node->m > 0) {
      StopExtending();
      return 0;
    }

    if (extend_budget_) {
      // The node is for real evaluation.
      nodes_to_process_.emplace_back(node);
      --extend_budget_;
      ++node->m;
      // If node is already known as terminal (win/lose/draw according to rules
      // of the game), it means that we already visited this node before.
      if (node->is_terminal) return 1;
      ++fresh_nodes_;
      search_->ExtendNode(node);
      // Turned out to be terminal, no need to fetch data.
      if (node->is_terminal) return 1;
      AddNodeToCompute(node);
      return 1;
    }

    // Only need to fetch for caching.
    if (AddNodeToCompute(node)) {
      // Already in cache, no need to compute it.
      computation_->PopLastInputHit();
    }
    return 0;
  }

  void StopExtending() { extend_budget_ = 0; }
  bool ShouldStop() const {
    // Cases when processing should stop:
    // 1. When first node to extend doesn't require NN fetch
    //    (i.e. cached or is terminal)
    if (!nodes_to_process_.empty() && computation_->GetCacheMisses() == 0)
      return true;
    // 2. When we stopped gathering of nodes to extend, and computation is full.
    if (extend_budget_ == 0 &&
        computation_->GetCacheMisses() >= nn_fetch_limit_)
      return true;
    return false;
  }

  Search* search_;
  int extend_budget_;
  int nn_fetch_limit_;
  CachingComputation* computation_;
  std::vector<Node*> nodes_to_process_;
  // Number of leaves, visited for the first time.
  int fresh_nodes_ = 0;
};

void Search::Worker() {
  // do {} while  instead of  while{} because at least one iteration is
  // necessary to get candidates.
  do {
    auto computation = CachingComputation(network_->NewComputation(), cache_);
    NodesGatherer gatherer(this, &computation, kParallelPlayouts,
                           kMiniBatchSize);
    gatherer.Gather(root_node_);

    // Evaluate nodes through NN.
    if (computation.GetBatchSize() != 0) {
      computation.ComputeBlocking();

      int idx_in_computation = 0;
      for (Node* node : gatherer.GetNodesToProcess()) {
        if (node->is_terminal) continue;
        // Populate Q value.
        node->v = -computation.GetQVal(idx_in_computation);
        // Populate P values.
        float total = 0.0;
        for (Node* n = node->child; n; n = n->sibling) {
          Move m = n->move;
          if (kFlipMove && node->board.flipped()) m.Mirror();
          float p = computation.GetPVal(idx_in_computation, m.as_nn_index());
          total += p;
          n->p = p;
        }
        // Scale P values to add up to 1.0.
        if (total > 0.0f) {
          for (Node* n = node->child; n; n = n->sibling) {
            n->p /= total;
          }
        }
        ++idx_in_computation;
      }
    }

    {
      // Update nodes.
      std::unique_lock<std::shared_mutex> lock{nodes_mutex_};
      total_nodes_ += gatherer.GetFreshNodes();
      for (Node* node : gatherer.GetNodesToProcess()) {
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
          --n->m;
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
    }
    MaybeOutputInfo();
    MaybeTriggerStop();
  } while (!stop_);
}

namespace {
// Returns a child with most visits.
Node* GetBestChild(Node* parent) {
  Node* best_node = nullptr;
  int best = -1;
  for (Node* node = parent->child; node; node = node->sibling) {
    int n = node->n + node->m;
    if (n > best) {
      best = n;
      best_node = node;
    }
  }
  return best_node;
}
}  // namespace

// A nodes_mutex_ must be locked when this function is called.
void Search::SendUciInfo() {
  if (!best_move_node_) return;
  last_outputted_best_move_node_ = best_move_node_;
  uci_info_.depth = root_node_->full_depth;
  uci_info_.seldepth = root_node_->max_depth;
  uci_info_.time = GetTimeSinceStart();
  uci_info_.nodes = total_nodes_;
  uci_info_.nps =
      uci_info_.time ? (uci_info_.nodes * 1000 / uci_info_.time) : 0;
  uci_info_.score = -191 * log(2 / (best_move_node_->q * 0.99 + 1) - 1);
  uci_info_.pv.clear();

  for (Node* iter = best_move_node_; iter; iter = GetBestChild(iter)) {
    Move m = iter->move;
    if (!iter->board.flipped()) m.Mirror();
    uci_info_.pv.push_back(m);
  }
  uci_info_.comment.clear();
  info_callback_(uci_info_);
}

// Decides whether anything important changed in stats and new info should be
// shown to a user.
void Search::MaybeOutputInfo() {
  std::unique_lock<std::shared_mutex> lock{nodes_mutex_};
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

void Search::MaybeTriggerStop() {
  std::lock_guard<std::mutex> lock(counters_mutex_);
  if (limits_.nodes >= 0 && total_nodes_ >= limits_.nodes) {
    stop_ = true;
  }
  if (limits_.time_ms >= 0 && GetTimeSinceStart() >= limits_.time_ms) {
    stop_ = true;
  }
  if (stop_ && !responded_bestmove_) {
    responded_bestmove_ = true;
    SendUciInfo();
    best_move_callback_(GetBestMove());
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

  node->repetitions = ComputeRepetitions(node);
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
      std::unique_lock<std::shared_mutex> lock{nodes_mutex_};
      // Check whether we are in the leave.
      if (node->n == 0 && node->m > 0) {
        // The node is currently being processed by another thread.
        // Undo the increments of anschestor nodes, and return null.
        for (node = node->parent; node != root_node_->parent;
             node = node->parent) {
          --node->m;
        }
        return nullptr;
      }
      ++node->m;
      // Found leave, and we are the the first to visit it.
      if (!node->child) {
        return node;
      }
    }

    // Now we are not in leave, we need to go deeper.
    std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
    float factor = kCpuct * std::sqrt(node->n + 1);
    float best = -100.0f;
    for (Node* iter = node->child; iter; iter = iter->sibling) {
      const float u = factor * iter->p / (1 + iter->n + iter->m);
      const float v = u + (iter->n ? iter->q
                                   : (kDynamicParent ? -iter->parent->q
                                                     : -iter->parent->v));
      if (v > best) {
        best = v;
        node = iter;
      }
    }
  }
}

InputPlanes Search::EncodeNode(const Node* node) {
  const int kMoveHistory = 8;
  const int kAuxPlaneBase = 14 * kMoveHistory;

  InputPlanes result(kAuxPlaneBase + 8);

  const bool we_are_black = node->board.flipped();
  bool flip = false;

  for (int i = 0; i < kMoveHistory; ++i, flip = !flip) {
    if (!node) break;
    ChessBoard board = node->board;

    if ((flip) != (kFlipHistory && i % 2 == 1)) board.Mirror();

    const int base = i * 14;
    if (i == 0) {
      if (board.castlings().we_can_000()) result[kAuxPlaneBase + 0].SetAll();
      if (board.castlings().we_can_00()) result[kAuxPlaneBase + 1].SetAll();
      if (board.castlings().they_can_000()) result[kAuxPlaneBase + 2].SetAll();
      if (board.castlings().they_can_00()) result[kAuxPlaneBase + 3].SetAll();
      if (we_are_black) result[kAuxPlaneBase + 4].SetAll();
      result[kAuxPlaneBase + 5].Fill(node->no_capture_ply);
      if (kPopulateMoves) result[kAuxPlaneBase + 6].Fill(node->ply_count % 256);
    }

    result[base + 0].mask = (board.ours() * board.pawns()).as_int();
    result[base + 1].mask = (board.our_knights()).as_int();
    result[base + 2].mask = (board.ours() * board.bishops()).as_int();
    result[base + 3].mask = (board.ours() * board.rooks()).as_int();
    result[base + 4].mask = (board.ours() * board.queens()).as_int();
    result[base + 5].mask = (board.our_king()).as_int();

    result[base + 6].mask = (board.theirs() * board.pawns()).as_int();
    result[base + 7].mask = (board.their_knights()).as_int();
    result[base + 8].mask = (board.theirs() * board.bishops()).as_int();
    result[base + 9].mask = (board.theirs() * board.rooks()).as_int();
    result[base + 10].mask = (board.theirs() * board.queens()).as_int();
    result[base + 11].mask = (board.their_king()).as_int();

    const int repetitions = node->repetitions;
    if (repetitions >= 1) result[base + 12].SetAll();
    if (repetitions >= 2) result[base + 13].SetAll();

    node = node->parent;
  }

  return result;
}

Move Search::GetBestMove() const {
  std::shared_lock<std::shared_mutex> lock(nodes_mutex_);
  Node* best_node = GetBestChild(root_node_);
  Move move = best_node->move;
  if (!best_node->board.flipped()) move.Mirror();
  return move;
}

void Search::StartThreads(int how_many) {
  std::lock_guard<std::mutex> lock(counters_mutex_);
  while (threads_.size() < how_many) {
    threads_.emplace_back([&]() { this->Worker(); });
  }
}

void Search::Stop() {
  std::lock_guard<std::mutex> lock(counters_mutex_);
  stop_ = true;
}

void Search::Abort() {
  std::lock_guard<std::mutex> lock(counters_mutex_);
  responded_bestmove_ = true;
  stop_ = true;
}

void Search::AbortAndWait() {
  {
    std::lock_guard<std::mutex> lock(counters_mutex_);
    responded_bestmove_ = true;
    stop_ = true;
  }
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
}

Search::~Search() { AbortAndWait(); }

}  // namespace lczero
/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "chess/gamestate.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "neural/batchsplit.h"
#include "search/register.h"
#include "search/search.h"
#include "utils/logging.h"

namespace lczero {
namespace {

// A very small, single-threaded PUCT search implementation for demonstration
// and experimentation purposes. It uses the Backend API directly and keeps a
// minimal in-memory tree.

struct Edge;
struct Node {
  // Priors, edges and stats for children.
  std::vector<Edge> edges;
  // Total visits through this node (sum of child visits).
  int n_total = 0;
  // Was NN evaluated and edges initialized.
  bool expanded = false;
  // Cached Q estimate for the node itself (not strictly required).
  float value_estimate = 0.0f;  // From the perspective of side to move here.
};

struct Edge {
  Move move;
  float prior = 0.0f;   // P
  int n = 0;            // N
  float w = 0.0f;       // W (sum of values)
  std::unique_ptr<Node> child;  // Owned child node
};

class CodexSearch : public SearchBase {
 public:
  using SearchBase::SearchBase;

  void SetBackend(Backend* backend) override {
    // Wrap to respect backend maximum batch sizes automatically.
    batchsplit_backend_ = CreateBatchSplitingBackend(backend);
    backend_ = batchsplit_backend_.get();
  }

  void NewGame() override {
    StopWorker();
    root_.reset();
  }

  void SetPosition(const GameState& state) override {
    StopWorker();
    game_state_ = state;
    root_.reset();
  }

  void StartClock() override { move_start_time_ = std::chrono::steady_clock::now(); }

  void StartSearch(const GoParams& params) override {
    // Prepare and launch a worker thread. Non-blocking.
    if (!backend_) return;
    responded_bestmove_.store(false, std::memory_order_relaxed);
    aborted_.store(false, std::memory_order_relaxed);
    force_stop_.store(false, std::memory_order_relaxed);
    params_ = params;
    // Compute simple time budget.
    ComputeTimeBudget();
    worker_ = std::thread(&CodexSearch::SearchLoop, this);
  }

  void WaitSearch() override {
    if (worker_.joinable()) worker_.join();
  }

  void StopSearch() override {
    force_stop_.store(true, std::memory_order_relaxed);
    // Try to respond with current best move quickly.
    MaybeRespondBestMove();
  }

  void AbortSearch() override {
    aborted_.store(true, std::memory_order_relaxed);
  }

 private:
  // Configuration
  static constexpr float kCpuct = 1.5f;
  static constexpr int kMinVisitsBeforeStop = 16;

  void StopWorker() {
    aborted_.store(true, std::memory_order_relaxed);
    if (worker_.joinable()) worker_.join();
    aborted_.store(false, std::memory_order_relaxed);
    force_stop_.store(false, std::memory_order_relaxed);
    responded_bestmove_.store(false, std::memory_order_relaxed);
  }

  // Computes a very simple time budget based on GoParams and side to move.
  void ComputeTimeBudget() {
    time_budget_ms_ = std::nullopt;
    if (params_.infinite || params_.ponder) return;
    if (params_.movetime) {
      time_budget_ms_ = *params_.movetime;
      return;
    }
    const bool is_black = game_state_.CurrentPosition().IsBlackToMove();
    const auto total = is_black ? params_.btime : params_.wtime;
    const auto inc = is_black ? params_.binc : params_.winc;
    if (total) {
      // A conservative split: spend ~1/30 of remaining + half increment.
      int64_t budget = std::max<int64_t>(*total / 30, 1);
      if (inc) budget += *inc / 2;
      // Clamp to total if tiny
      if (inc && budget > *total) budget = std::min<int64_t>(budget, *total);
      time_budget_ms_ = budget;
    }
  }

  // Expand a node using NN. Returns the leaf value from the perspective of
  // side to move in the position given by `history`.
  float ExpandNode(Node* node, PositionHistory& history) {
    const ChessBoard& board = history.Last().GetBoard();
    const MoveList legal_moves = board.GenerateLegalMoves();

    // Terminal by no legal moves: checkmate or stalemate.
    if (legal_moves.empty()) {
      const bool in_check = board.IsUnderCheck();
      node->expanded = true;
      node->edges.clear();
      node->value_estimate = in_check ? -1.0f : 0.0f;  // from side-to-move
      return node->value_estimate;
    }

    // Terminal by immediate draw rules (50-move, repetition, TB draw).
    const GameResult gr = history.ComputeGameResult();
    if (gr == GameResult::DRAW) {
      node->expanded = true;
      node->edges.clear();
      node->value_estimate = 0.0f;
      return 0.0f;
    }

    // Query backend for priors and value.
    EvalResult res;
    res.p.resize(legal_moves.size());

    auto comp = backend_->CreateComputation();
    comp->AddInput(EvalPosition{history.GetPositions(), legal_moves},
                   EvalResultPtr{.q = &res.q, .d = &res.d, .m = &res.m,
                                 .p = std::span<float>(res.p.data(), res.p.size())});
    comp->ComputeBlocking();

    // Initialize edges with priors.
    node->edges.clear();
    node->edges.reserve(legal_moves.size());
    double sum_p = 0.0;
    for (float p : res.p) sum_p += std::max(0.0f, p);
    if (sum_p <= 0.0) sum_p = 1.0;  // Avoid division by zero
    for (size_t i = 0; i < legal_moves.size(); ++i) {
      Edge e;
      e.move = legal_moves[i];
      e.prior = static_cast<float>(std::max(0.0f, res.p[i]) / sum_p);
      node->edges.emplace_back(std::move(e));
    }
    node->expanded = true;
    node->value_estimate = res.q;  // NN returns value for side to move
    return res.q;
  }

  // Select a path from root using PUCT, expanding one new leaf. Returns the
  // value evaluated at the leaf (from leaf side-to-move viewpoint) and updates
  // statistics along the path (with alternating sign for parents).
  void RunOneSimulation() {
    if (!root_) root_ = std::make_unique<Node>();
    PositionHistory history(game_state_.GetPositions());
    std::vector<std::pair<Node*, int>> path;  // (node, edge_idx)
    Node* node = root_.get();

    // Expand root if needed.
    if (!node->expanded) {
      ExpandNode(node, history);
    }

    int depth = 0;
    // Selection down the tree.
    while (node->expanded && !node->edges.empty()) {
      const float sqrt_total = std::sqrt(static_cast<float>(std::max(1, node->n_total)));
      int best_idx = -1;
      float best_score = -std::numeric_limits<float>::infinity();
      for (size_t i = 0; i < node->edges.size(); ++i) {
        const Edge& e = node->edges[i];
        const float q = (e.n > 0) ? (e.w / e.n) : 0.0f;
        const float u = kCpuct * e.prior * sqrt_total / (1.0f + e.n);
        const float score = q + u;
        if (score > best_score) {
          best_score = score;
          best_idx = static_cast<int>(i);
        }
      }
      if (best_idx < 0) break;  // Shouldn't happen
      Edge& chosen = node->edges[best_idx];
      path.emplace_back(node, best_idx);
      history.Append(chosen.move);
      // If child not yet created, create an empty node and expand it as leaf.
      if (!chosen.child) {
        chosen.child = std::make_unique<Node>();
      }
      node = chosen.child.get();
      depth++;
    }

    max_depth_ = std::max(max_depth_, depth);
    // Now node is an unexpanded leaf or terminal. Expand/evaluate it.
    const float leaf_v = ExpandNode(node, history);

    // Backpropagate with sign alternation. leaf_v is from leaf side-to-move;
    // the last edge in path leads TO the leaf from its parent. From parent's
    // side-to-move perspective, value is -leaf_v. Alternate signs upwards.
    float v = -leaf_v;
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
      Node* parent = it->first;
      Edge& e = parent->edges[it->second];
      e.n += 1;
      e.w += v;
      parent->n_total += 1;
      v = -v;  // Flip perspective each ply
    }
    nodes_visited_ += 1;
  }

  // Prepare and send a thinking info update (best-so-far line and score).
  void SendThinkingInfo() {
    if (!root_ || root_->edges.empty()) return;
    const int64_t elapsed_ms = move_start_time_ ?
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - *move_start_time_).count() : 0;

    // Build PV by following max-N children.
    std::vector<Move> pv;
    const Node* node = root_.get();
    while (node && node->expanded && !node->edges.empty()) {
      const auto it = std::max_element(
          node->edges.begin(), node->edges.end(),
          [](const Edge& a, const Edge& b) { return a.n < b.n; });
      if (it == node->edges.end() || it->n == 0) break;
      pv.push_back(it->move);
      node = it->child.get();
      if (pv.size() >= 16) break;
    }

    // Estimate root score as average Q of best child, if any.
    float root_score_q = 0.0f;
    if (!root_->edges.empty()) {
      const auto it = std::max_element(
          root_->edges.begin(), root_->edges.end(),
          [](const Edge& a, const Edge& b) { return a.n < b.n; });
      if (it != root_->edges.end() && it->n > 0) root_score_q = it->w / it->n;
    }

    // Convert Q to a centipawn-like score (same transform as instamove).
    const int cp = static_cast<int>(std::round(90 * std::tan(1.5637541897 * root_score_q)));

    std::vector<ThinkingInfo> infos{{
        .depth = std::max(1, max_depth_),
        .seldepth = std::max(1, max_depth_),
        .time = elapsed_ms,
        .nodes = nodes_visited_,
        .nps = elapsed_ms > 0 ? static_cast<int>(1000.0 * nodes_visited_ / std::max<int64_t>(1, elapsed_ms)) : -1,
        .score = cp,
        .pv = pv,
        .multipv = 1,
    }};
    uci_responder_->OutputThinkingInfo(&infos);
  }

  // Respond with best move if not yet responded.
  void MaybeRespondBestMove() {
    if (responded_bestmove_.exchange(true)) return;
    Move best = Move{};
    Move ponder = Move{};
    if (root_ && !root_->edges.empty()) {
      const auto it = std::max_element(
          root_->edges.begin(), root_->edges.end(),
          [](const Edge& a, const Edge& b) { return a.n < b.n; });
      if (it != root_->edges.end()) {
        best = it->move;
        // Ponder is next best PV move if available.
        if (it->child && it->child->expanded && !it->child->edges.empty()) {
          const auto it2 = std::max_element(
              it->child->edges.begin(), it->child->edges.end(),
              [](const Edge& a, const Edge& b) { return a.n < b.n; });
          if (it2 != it->child->edges.end() && it2->n > 0) ponder = it2->move;
        }
      }
    }

    BestMoveInfo info{best, ponder};
    // Temporary compatibility: moves are encoded from current player
    // perspective, flip before sending when black to move.
    if (game_state_.CurrentPosition().IsBlackToMove()) {
      if (!info.bestmove.is_null()) info.bestmove.Flip();
      if (!info.ponder.is_null()) info.ponder.Flip();
    }
    uci_responder_->OutputBestMove(&info);
  }

  // Main worker loop.
  void SearchLoop() {
    // Ensure we have a fresh root.
    if (!root_) root_ = std::make_unique<Node>();

    const auto start = move_start_time_.value_or(std::chrono::steady_clock::now());
    const auto deadline = (time_budget_ms_ ? start + std::chrono::milliseconds(*time_budget_ms_)
                                           : std::chrono::steady_clock::time_point::max());

    auto last_info = std::chrono::steady_clock::now();

    // Ensure at least one expansion so we have a legal move to report.
    if (!aborted_.load(std::memory_order_relaxed)) {
      RunOneSimulation();
    }

    while (!aborted_.load(std::memory_order_relaxed)) {
      if (force_stop_.load(std::memory_order_relaxed)) break;
      if (!params_.infinite && std::chrono::steady_clock::now() >= deadline &&
          nodes_visited_ >= kMinVisitsBeforeStop) {
        break;
      }
      RunOneSimulation();

      // Periodic info every ~200ms.
      const auto now = std::chrono::steady_clock::now();
      if (now - last_info >= std::chrono::milliseconds(200)) {
        SendThinkingInfo();
        last_info = now;
      }
    }

    if (!aborted_.load(std::memory_order_relaxed)) {
      SendThinkingInfo();
      MaybeRespondBestMove();
    }
  }

  // State
  GameState game_state_;
  std::unique_ptr<Backend> batchsplit_backend_;
  std::unique_ptr<Node> root_;

  std::thread worker_;
  std::atomic<bool> aborted_{false};
  std::atomic<bool> force_stop_{false};
  std::atomic<bool> responded_bestmove_{false};

  GoParams params_;
  std::optional<int64_t> time_budget_ms_;
  std::optional<std::chrono::steady_clock::time_point> move_start_time_;

  // Counters/metrics
  int max_depth_ = 0;
  int64_t nodes_visited_ = 0;
};

class CodexFactory : public SearchFactory {
 public:
  std::string_view GetName() const override { return "codex"; }
  std::unique_ptr<SearchBase> CreateSearch(UciResponder* responder,
                                           const OptionsDict*) const override {
    return std::make_unique<CodexSearch>(responder);
  }

  void PopulateParams(OptionsParser*) const override {}
};

REGISTER_SEARCH(CodexFactory)

}  // namespace
}  // namespace lczero


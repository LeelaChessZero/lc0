/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "chess/gamestate.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "neural/batchsplit.h"
#include "search/register.h"
#include "search/search.h"
#include "utils/logging.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {

const OptionId kValueHeadDepthId{
    "valuehead-depth", "ValueHeadDepth",
    "Number of full-width minimax plies for value-head search."};
const OptionId kValueHeadMaxNodesId{
    "valuehead-max-nodes", "ValueHeadMaxNodes",
    "Maximum number of positions in a fixed-depth value-head search tree."};

constexpr int kDefaultValueHeadDepth = 1;
constexpr int kMaximumValueHeadDepth = 99;
constexpr int kDefaultValueHeadMaxNodes = 10000;
constexpr int kMaximumValueHeadMaxNodes = 100000000;

bool IsCancellationRequested(const std::atomic<bool>& cancellation_requested) {
  return cancellation_requested.load(std::memory_order_relaxed);
}

class InstamoveSearch : public SearchBase {
 public:
  using SearchBase::SearchBase;
  ~InstamoveSearch() override { Shutdown(); }

 private:
  virtual Move GetBestMove(const GameState& game_state,
                           const GoParams& go_params,
                           const std::atomic<bool>& cancellation_requested) = 0;

  void SetPosition(const GameState& game_state) final {
    game_state_ = game_state;
  }

  void StartSearch(const GoParams& go_params) final {
    if (worker_.joinable()) {
      {
        std::lock_guard<std::mutex> lock(response_mutex_);
        cancellation_requested_.store(true, std::memory_order_relaxed);
        responded_bestmove_.store(true, std::memory_order_relaxed);
      }
      worker_.join();
    }

    const MoveList legal_moves =
        game_state_.CurrentPosition().GetBoard().GenerateLegalMoves();
    {
      std::lock_guard<std::mutex> lock(bestmove_mutex_);
      bestmove_ = legal_moves.empty() ? Move{} : legal_moves.front();
    }
    {
      std::lock_guard<std::mutex> lock(response_mutex_);
      cancellation_requested_.store(false, std::memory_order_relaxed);
      responded_bestmove_.store(false, std::memory_order_relaxed);
    }
    try {
      worker_ = std::thread([this, go_params] { RunSearch(go_params); });
    } catch (...) {
      RespondBestMove();
      throw;
    }
  }
  void WaitSearch() final {
    if (worker_.joinable()) worker_.join();
  }
  void StopSearch() final {
    std::lock_guard<std::mutex> lock(response_mutex_);
    cancellation_requested_.store(true, std::memory_order_relaxed);
    RespondBestMoveLocked();
  }
  void AbortSearch() final {
    std::lock_guard<std::mutex> lock(response_mutex_);
    cancellation_requested_.store(true, std::memory_order_relaxed);
    responded_bestmove_.store(true, std::memory_order_relaxed);
  }
  void RespondBestMove() {
    std::lock_guard<std::mutex> lock(response_mutex_);
    RespondBestMoveLocked();
  }
  void RespondBestMoveLocked() {
    if (responded_bestmove_.exchange(true)) return;
    Move bestmove;
    {
      std::lock_guard<std::mutex> lock(bestmove_mutex_);
      bestmove = bestmove_;
    }
    BestMoveInfo info{bestmove};
    // TODO Remove this when move will be encoded from white perspective.
    if (game_state_.CurrentPosition().IsBlackToMove()) {
      info.bestmove.Flip();
    } else if (!info.ponder.is_null()) {
      info.ponder.Flip();
    }
    uci_responder_->OutputBestMove(&info);
  }

  void RunSearch(const GoParams& go_params) noexcept {
    try {
      Move bestmove =
          GetBestMove(game_state_, go_params, cancellation_requested_);
      PublishBestMoveIfActive(bestmove);
    } catch (const std::exception& exception) {
      LOGFILE << "Instamove search failed: " << exception.what();
    } catch (...) {
      LOGFILE << "Instamove search failed with an unknown exception.";
    }
    if (!go_params.infinite && !go_params.ponder) {
      RespondBestMove();
    }
  }

  void PublishBestMoveIfActive(Move bestmove) {
    std::lock_guard<std::mutex> response_lock(response_mutex_);
    if (cancellation_requested_.load(std::memory_order_relaxed) ||
        responded_bestmove_.load(std::memory_order_relaxed)) {
      return;
    }
    std::lock_guard<std::mutex> bestmove_lock(bestmove_mutex_);
    bestmove_ = bestmove;
  }

  void SetBackend(Backend* backend) override {
    batchsplit_backend_ = CreateBatchSplitingBackend(backend);
    backend_ = batchsplit_backend_.get();
  }
  void StartClock() final {}

 protected:
  void Shutdown() {
    {
      std::lock_guard<std::mutex> lock(response_mutex_);
      cancellation_requested_.store(true, std::memory_order_relaxed);
      responded_bestmove_.store(true, std::memory_order_relaxed);
    }
    if (worker_.joinable()) worker_.join();
  }

  void PublishBestMoveAndOutputThinkingInfoIfActive(
      Move bestmove, std::vector<ThinkingInfo>* infos) {
    std::lock_guard<std::mutex> lock(response_mutex_);
    if (cancellation_requested_.load(std::memory_order_relaxed) ||
        responded_bestmove_.load(std::memory_order_relaxed)) {
      return;
    }
    {
      std::lock_guard<std::mutex> bestmove_lock(bestmove_mutex_);
      bestmove_ = bestmove;
    }
    uci_responder_->OutputThinkingInfo(infos);
  }

 private:
  Move bestmove_;
  std::mutex bestmove_mutex_;
  std::mutex response_mutex_;
  std::thread worker_;
  std::atomic<bool> cancellation_requested_{false};
  std::atomic<bool> responded_bestmove_{true};
  std::unique_ptr<Backend> batchsplit_backend_;
  GameState game_state_;
};

class PolicyHeadSearch : public InstamoveSearch {
 public:
  using InstamoveSearch::InstamoveSearch;
  ~PolicyHeadSearch() override { Shutdown(); }

  Move GetBestMove(const GameState& game_state, const GoParams&,
                   const std::atomic<bool>& cancellation_requested) final {
    if (IsCancellationRequested(cancellation_requested)) return Move{};
    const std::vector<Position> positions = game_state.GetPositions();
    MoveList legal_moves = positions.back().GetBoard().GenerateLegalMoves();
    std::vector<EvalResult> res = backend_->EvaluateBatch(
        std::vector<EvalPosition>{EvalPosition{positions, legal_moves}});
    if (IsCancellationRequested(cancellation_requested)) return Move{};
    const size_t best_move_idx =
        std::max_element(res[0].p.begin(), res[0].p.end()) - res[0].p.begin();

    std::vector<ThinkingInfo> infos = {{
        .depth = 1,
        .seldepth = 1,
        .nodes = 1,
        .score = 90 * std::tan(1.5637541897 * res[0].q),
        .wdl =
            ThinkingInfo::WDL{
                static_cast<int>(std::round(500 * (1 + res[0].q - res[0].d))),
                static_cast<int>(std::round(1000 * res[0].d)),
                static_cast<int>(std::round(500 * (1 - res[0].q - res[0].d)))},
    }};
    Move best_move = legal_moves[best_move_idx];
    PublishBestMoveAndOutputThinkingInfoIfActive(best_move, &infos);
    return best_move;
  }
};

class ValueHeadSearch : public InstamoveSearch {
 public:
  ValueHeadSearch(UciResponder* responder, const OptionsDict* options)
      : InstamoveSearch(responder), options_(options) {}
  ~ValueHeadSearch() override { Shutdown(); }

  Move GetBestMove(const GameState& game_state, const GoParams& go_params,
                   const std::atomic<bool>& cancellation_requested) final {
    if (IsCancellationRequested(cancellation_requested)) return Move{};
    const int requested_depth = std::clamp(
        go_params.depth.value_or(options_->Get<int>(kValueHeadDepthId)), 1,
        kMaximumValueHeadDepth);
    const int64_t max_nodes = options_->Get<int>(kValueHeadMaxNodesId);

    std::unique_ptr<SearchTree> tree;
    // Build candidate trees without queuing NN inputs. A rejected tree must
    // not trigger evaluation through the batch-splitting backend.
    for (int depth = 1; depth <= requested_depth; ++depth) {
      if (IsCancellationRequested(cancellation_requested)) return Move{};
      auto candidate = std::make_unique<SearchTree>();
      candidate->depth = depth;
      PositionHistory history(game_state.GetPositions());
      const BuildResult result = BuildTree(
          &history, &candidate->root, depth, 1, max_nodes, &candidate->nodes,
          &candidate->seldepth, cancellation_requested);
      if (result == BuildResult::kCancelled) return Move{};
      if (result == BuildResult::kNodeLimit) break;
      tree = std::move(candidate);
    }

    if (!tree) {
      if (IsCancellationRequested(cancellation_requested)) return Move{};
      const MoveList legal_moves =
          game_state.CurrentPosition().GetBoard().GenerateLegalMoves();
      std::vector<ThinkingInfo> infos = {
          {.depth = 1, .seldepth = 0, .nodes = 0}};
      const Move bestmove = legal_moves.empty() ? Move{} : legal_moves.front();
      PublishBestMoveAndOutputThinkingInfoIfActive(bestmove, &infos);
      return bestmove;
    }

    if (tree->root.children.empty()) return Move{};

    std::unique_ptr<BackendComputation> computation =
        backend_->CreateComputation();
    PositionHistory history(game_state.GetPositions());
    size_t evaluations = 0;
    if (!QueueEvaluations(&history, &tree->root, computation.get(),
                          &evaluations, cancellation_requested)) {
      return Move{};
    }
    if (IsCancellationRequested(cancellation_requested)) return Move{};
    if (evaluations != 0) {
      computation->ComputeBlocking();
      if (IsCancellationRequested(cancellation_requested)) return Move{};
    }

    if (!ResolveScore(&tree->root, cancellation_requested)) return Move{};
    SearchNode* best_child = tree->root.children.front().get();
    Score best_score = FlipScore(best_child->score);
    for (size_t i = 1; i < tree->root.children.size(); ++i) {
      if (IsCancellationRequested(cancellation_requested)) return Move{};
      Score candidate = FlipScore(tree->root.children[i]->score);
      if (IsBetter(candidate, best_score)) {
        best_child = tree->root.children[i].get();
        best_score = candidate;
      }
    }

    auto to_int = [](double x) { return static_cast<int>(std::round(x)); };
    std::optional<int> mate;
    if (best_score.mate) {
      mate = best_score.mate->winning ? (best_score.mate->plies + 1) / 2
                                      : -(best_score.mate->plies / 2);
    }
    if (IsCancellationRequested(cancellation_requested)) return Move{};
    std::vector<ThinkingInfo> infos{
        {.depth = tree->depth,
         .seldepth = tree->seldepth,
         .nodes = tree->nodes,
         .mate = mate,
         .score = mate ? std::nullopt
                       : std::make_optional<int>(to_int(
                             90 * std::tan(1.5637541897 * best_score.q))),
         .wdl = mate ? std::nullopt
                     : std::make_optional<ThinkingInfo::WDL>(ThinkingInfo::WDL{
                           .w = to_int(500 * (1 + best_score.q - best_score.d)),
                           .d = to_int(1000 * best_score.d),
                           .l = to_int(500 * (1 - best_score.q - best_score.d)),
                       })}};
    PublishBestMoveAndOutputThinkingInfoIfActive(best_child->move, &infos);
    return best_child->move;
  }

 private:
  struct MateScore {
    bool winning;
    int plies;
  };

  struct Score {
    float q = 0.0f;
    float d = 0.0f;
    std::optional<MateScore> mate;
  };

  struct SearchNode {
    Score score;
    Move move;
    bool needs_evaluation = false;
    std::vector<std::unique_ptr<SearchNode>> children;
  };

  struct SearchTree {
    SearchNode root;
    int depth = 0;
    int seldepth = 0;
    int64_t nodes = 0;
  };

  enum class BuildResult { kComplete, kNodeLimit, kCancelled };

  static bool IsBetter(const Score& score, const Score& other) {
    if (score.mate && other.mate) {
      if (score.mate->winning != other.mate->winning) {
        return score.mate->winning;
      }
      return score.mate->winning ? score.mate->plies < other.mate->plies
                                 : score.mate->plies > other.mate->plies;
    }
    if (score.mate) return score.mate->winning;
    if (other.mate) return !other.mate->winning;
    return score.q > other.q;
  }

  static Score FlipScore(const Score& score) {
    Score flipped{.q = -score.q, .d = score.d};
    if (score.mate) {
      flipped.mate = MateScore{.winning = !score.mate->winning,
                               .plies = score.mate->plies + 1};
    }
    return flipped;
  }

  static bool ResolveScore(SearchNode* node,
                           const std::atomic<bool>& cancellation_requested) {
    if (IsCancellationRequested(cancellation_requested)) return false;
    if (node->children.empty()) return true;
    if (!ResolveScore(node->children.front().get(), cancellation_requested)) {
      return false;
    }
    node->score = FlipScore(node->children.front()->score);
    for (size_t i = 1; i < node->children.size(); ++i) {
      if (IsCancellationRequested(cancellation_requested)) return false;
      if (!ResolveScore(node->children[i].get(), cancellation_requested)) {
        return false;
      }
      Score candidate = FlipScore(node->children[i]->score);
      if (IsBetter(candidate, node->score)) node->score = candidate;
    }
    return true;
  }

  static BuildResult BuildTree(
      PositionHistory* history, SearchNode* node, int remaining_depth, int ply,
      int64_t max_nodes, int64_t* nodes, int* seldepth,
      const std::atomic<bool>& cancellation_requested) {
    if (IsCancellationRequested(cancellation_requested)) {
      return BuildResult::kCancelled;
    }
    const std::vector<Move> legal_moves =
        history->Last().GetBoard().GenerateLegalMoves();
    node->children.reserve(legal_moves.size());
    for (const Move& move : legal_moves) {
      if (IsCancellationRequested(cancellation_requested)) {
        return BuildResult::kCancelled;
      }
      if (*nodes >= max_nodes) return BuildResult::kNodeLimit;
      ++*nodes;
      auto child = std::make_unique<SearchNode>();
      child->move = move;
      history->Append(move);
      *seldepth = std::max(*seldepth, ply);
      switch (history->ComputeGameResult()) {
        case GameResult::UNDECIDED:
          if (remaining_depth == 1) {
            child->needs_evaluation = true;
          } else {
            const BuildResult result =
                BuildTree(history, child.get(), remaining_depth - 1, ply + 1,
                          max_nodes, nodes, seldepth, cancellation_requested);
            if (result != BuildResult::kComplete) return result;
          }
          break;
        case GameResult::DRAW:
          child->score = {.q = 0.0f, .d = 1.0f};
          break;
        default:
          child->score = {
              .q = -1.0f,
              .d = 0.0f,
              .mate = MateScore{.winning = false, .plies = 0},
          };
      }
      history->Pop();
      node->children.push_back(std::move(child));
    }
    return BuildResult::kComplete;
  }

  static bool QueueEvaluations(
      PositionHistory* history, SearchNode* node,
      BackendComputation* computation, size_t* evaluations,
      const std::atomic<bool>& cancellation_requested) {
    if (IsCancellationRequested(cancellation_requested)) return false;
    for (const auto& child : node->children) {
      if (IsCancellationRequested(cancellation_requested)) return false;
      history->Append(child->move);
      if (child->needs_evaluation) {
        computation->AddInput(
            EvalPosition{history->GetPositions(), {}},
            EvalResultPtr{.q = &child->score.q, .d = &child->score.d});
        ++*evaluations;
      } else if (!child->children.empty()) {
        if (!QueueEvaluations(history, child.get(), computation, evaluations,
                              cancellation_requested)) {
          return false;
        }
      }
      history->Pop();
    }
    return true;
  }

  const OptionsDict* options_;
};

class PolicyHeadFactory : public SearchFactory {
  std::string_view GetName() const override { return "policyhead"; }
  std::unique_ptr<SearchBase> CreateSearch(UciResponder* responder,
                                           const OptionsDict*) const override {
    return std::make_unique<PolicyHeadSearch>(responder);
  }
};

class ValueHeadFactory : public SearchFactory {
  std::string_view GetName() const override { return "valuehead"; }
  void PopulateParams(OptionsParser* options) const override {
    options->Add<IntOption>(kValueHeadDepthId, 1, kMaximumValueHeadDepth) =
        kDefaultValueHeadDepth;
    options->Add<IntOption>(kValueHeadMaxNodesId, 1,
                            kMaximumValueHeadMaxNodes) =
        kDefaultValueHeadMaxNodes;
  }
  std::unique_ptr<SearchBase> CreateSearch(
      UciResponder* responder, const OptionsDict* options) const override {
    return std::make_unique<ValueHeadSearch>(responder, options);
  }
};

REGISTER_SEARCH(PolicyHeadFactory)
REGISTER_SEARCH(ValueHeadFactory)

}  // namespace
}  // namespace lczero

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
#include <cmath>
#include <optional>
#include <vector>

#include "chess/gamestate.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "neural/batchsplit.h"
#include "search/register.h"
#include "search/search.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {

const OptionId kValueHeadDepthId{
    {.long_flag = "valuehead-depth",
     .uci_option = "ValueHeadDepth",
     .help_text = "Fixed minimax depth in plies for valuehead search. The "
                  "standard `go depth` command overrides this value.",
     .visibility = OptionId::kAlwaysVisible}};

MoveList StringsToMovelist(const std::vector<std::string>& moves,
                           const ChessBoard& board) {
  MoveList legal_moves = board.GenerateLegalMoves();
  if (moves.empty()) return legal_moves;

  MoveList result;
  result.reserve(moves.size());
  for (const auto& move : moves) {
    const Move parsed_move = board.ParseMove(move);
    if (std::find(legal_moves.begin(), legal_moves.end(), parsed_move) !=
        legal_moves.end()) {
      result.emplace_back(parsed_move);
    }
  }
  if (result.empty()) throw Exception("No legal searchmoves.");
  return result;
}

class InstamoveSearch : public SearchBase {
 public:
  using SearchBase::SearchBase;

 private:
  virtual Move GetBestMove(const GameState& game_state,
                           const GoParams& go_params) = 0;

  void SetPosition(const GameState& game_state) final {
    game_state_ = game_state;
  }

  void StartSearch(const GoParams& go_params) final {
    responded_bestmove_.store(false, std::memory_order_relaxed);
    bestmove_ = GetBestMove(game_state_, go_params);
    if (!go_params.infinite && !go_params.ponder) RespondBestMove();
  }
  void WaitSearch() final {
    while (!responded_bestmove_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  void StopSearch() final { RespondBestMove(); }
  void AbortSearch() final { responded_bestmove_.store(true); }
  void RespondBestMove() {
    if (responded_bestmove_.exchange(true)) return;
    BestMoveInfo info{bestmove_};
    // TODO Remove this when move will be encoded from white perspective.
    if (game_state_.CurrentPosition().IsBlackToMove()) {
      info.bestmove.Flip();
    } else if (!info.ponder.is_null()) {
      info.ponder.Flip();
    }
    uci_responder_->OutputBestMove(&info);
  }

  void SetBackend(Backend* backend) override {
    batchsplit_backend_ = CreateBatchSplitingBackend(backend);
    backend_ = batchsplit_backend_.get();
  }
  void StartClock() final {}

  Move bestmove_;
  std::atomic<bool> responded_bestmove_{false};
  std::unique_ptr<Backend> batchsplit_backend_;
  GameState game_state_;
};

class PolicyHeadSearch : public InstamoveSearch {
 public:
  using InstamoveSearch::InstamoveSearch;

  Move GetBestMove(const GameState& game_state,
                   const GoParams& go_params) final {
    const std::vector<Position> positions = game_state.GetPositions();
    MoveList legal_moves =
        StringsToMovelist(go_params.searchmoves, positions.back().GetBoard());
    if (legal_moves.empty()) return Move{};
    std::vector<EvalResult> res = backend_->EvaluateBatch(
        std::vector<EvalPosition>{EvalPosition{positions, legal_moves}});
    const size_t best_move_idx =
        std::max_element(res[0].p.begin(), res[0].p.end()) - res[0].p.begin();

    std::vector<ThinkingInfo> infos = {{
        .depth = 1,
        .seldepth = 1,
        .nodes = 1,
        .score = static_cast<int>(
            std::round(90 * std::tan(1.5637541897 * res[0].q))),
        .wdl =
            ThinkingInfo::WDL{
                static_cast<int>(std::round(
                    500 * (1 + res[0].q - res[0].d))),
                static_cast<int>(std::round(1000 * res[0].d)),
                static_cast<int>(std::round(
                    500 * (1 - res[0].q - res[0].d)))},
    }};
    uci_responder_->OutputThinkingInfo(&infos);

    Move best_move = legal_moves[best_move_idx];
    return best_move;
  }
};

class ValueHeadSearch : public InstamoveSearch {
 public:
  ValueHeadSearch(UciResponder* responder, const OptionsDict* options)
      : InstamoveSearch(responder), options_(options) {}

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

  struct SearchStats {
    int seldepth = 0;
    int64_t nodes = 0;
  };

  static Score FlipScore(const Score& score) {
    Score flipped{
        .q = -score.q,
        .d = score.d,
        .mate = score.mate
                    ? std::make_optional<MateScore>(
                          MateScore{.winning = !score.mate->winning,
                                    .plies = score.mate->plies + 1})
                    : std::nullopt,
    };
    return flipped;
  }

  static bool IsBetterScore(const Score& lhs, const Score& rhs) {
    if (lhs.mate || rhs.mate) {
      if (!lhs.mate) return rhs.mate->winning ? false : true;
      if (!rhs.mate) return lhs.mate->winning ? true : false;
      if (lhs.mate->winning != rhs.mate->winning) return lhs.mate->winning;
      if (lhs.mate->winning) return lhs.mate->plies < rhs.mate->plies;
      return lhs.mate->plies > rhs.mate->plies;
    }
    return lhs.q > rhs.q;
  }

  static int ToUciMate(const MateScore& mate) {
    return mate.winning ? (mate.plies + 1) / 2 : -(mate.plies / 2);
  }

  int GetSearchDepth(const GoParams& go_params) const {
    if (go_params.depth) return std::max(1, *go_params.depth);
    return std::max(1, options_->Get<int>(kValueHeadDepthId));
  }

  Score EvaluatePosition(PositionHistory* history, SearchStats* stats,
                         int ply_from_root) {
    stats->seldepth = std::max(stats->seldepth, ply_from_root);
    switch (history->ComputeGameResult()) {
      case GameResult::DRAW:
        return Score{.q = 0.0f, .d = 1.0f, .mate = std::nullopt};
      case GameResult::UNDECIDED:
        break;
      default:
        return Score{
            .q = -1.0f,
            .d = 0.0f,
            .mate = MateScore{.winning = false, .plies = 0},
        };
    }

    const EvalResult eval = backend_->EvaluateBatch(
        std::vector<EvalPosition>{EvalPosition{history->GetPositions(), {}}})[0];
    return Score{.q = eval.q, .d = eval.d, .mate = std::nullopt};
  }

  std::vector<Score> EvaluateMoves(PositionHistory* history,
                                   const MoveList& legal_moves, int depth,
                                   int ply_from_root, SearchStats* stats) {
    std::vector<Score> results(legal_moves.size());
    if (depth == 1) {
      std::unique_ptr<BackendComputation> computation =
          backend_->CreateComputation();
      std::vector<size_t> pending_indices;
      pending_indices.reserve(legal_moves.size());
      for (size_t i = 0; i < legal_moves.size(); ++i) {
        history->Append(legal_moves[i]);
        ++stats->nodes;
        stats->seldepth = std::max(stats->seldepth, ply_from_root + 1);
        switch (history->ComputeGameResult()) {
          case GameResult::UNDECIDED:
            computation->AddInput(
                EvalPosition{history->GetPositions(), {}},
                EvalResultPtr{.q = &results[i].q, .d = &results[i].d});
            pending_indices.emplace_back(i);
            break;
          case GameResult::DRAW:
            results[i] = Score{.q = 0.0f, .d = 1.0f, .mate = std::nullopt};
            break;
          default:
            results[i] = Score{
                .q = 1.0f,
                .d = 0.0f,
                .mate = MateScore{.winning = true, .plies = 1},
            };
            break;
        }
        history->Pop();
      }
      computation->ComputeBlocking();
      for (const size_t idx : pending_indices) {
        results[idx] = FlipScore(results[idx]);
      }
      return results;
    }

    for (size_t i = 0; i < legal_moves.size(); ++i) {
      history->Append(legal_moves[i]);
      ++stats->nodes;
      results[i] =
          FlipScore(EvaluateNode(history, depth - 1, ply_from_root + 1, stats));
      history->Pop();
    }
    return results;
  }

  Score EvaluateNode(PositionHistory* history, int depth, int ply_from_root,
                     SearchStats* stats) {
    if (depth <= 0) return EvaluatePosition(history, stats, ply_from_root);
    const MoveList legal_moves = history->Last().GetBoard().GenerateLegalMoves();
    if (legal_moves.empty()) {
      return EvaluatePosition(history, stats, ply_from_root);
    }

    const std::vector<Score> results =
        EvaluateMoves(history, legal_moves, depth, ply_from_root, stats);
    return *std::max_element(results.begin(), results.end(),
                             [](const Score& lhs, const Score& rhs) {
                               return IsBetterScore(rhs, lhs);
                             });
  }

  Move GetBestMove(const GameState& game_state,
                   const GoParams& go_params) final {
    PositionHistory history(game_state.GetPositions());
    const MoveList legal_moves =
        StringsToMovelist(go_params.searchmoves, history.Last().GetBoard());
    if (legal_moves.empty()) return Move{};

    SearchStats stats;
    const int search_depth = GetSearchDepth(go_params);
    const std::vector<Score> results =
        EvaluateMoves(&history, legal_moves, search_depth, 0, &stats);
    const size_t best_idx =
        std::max_element(results.begin(), results.end(),
                         [](const Score& lhs, const Score& rhs) {
                           return IsBetterScore(rhs, lhs);
                         }) -
        results.begin();

    const Score& r = results[best_idx];
    auto to_int = [](double x) { return static_cast<int>(std::round(x)); };
    std::vector<ThinkingInfo> infos{
        {.depth = search_depth,
         .seldepth = stats.seldepth,
         .nodes = stats.nodes,
         .mate = r.mate ? std::make_optional(ToUciMate(*r.mate))
                        : std::nullopt,
         .score = r.mate ? std::nullopt
                         : std::make_optional<int>(
                               static_cast<int>(
                                   std::round(90 * std::tan(1.5637541897 *
                                                            r.q)))),
         .wdl = r.mate
                    ? std::nullopt
                    : std::make_optional<ThinkingInfo::WDL>(ThinkingInfo::WDL{
                          .w = to_int(500 * (1 + r.q - r.d)),
                          .d = to_int(1000 * r.d),
                          .l = to_int(500 * (1 - r.q - r.d)),
                      })}};
    uci_responder_->OutputThinkingInfo(&infos);
    return legal_moves[best_idx];
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
  std::unique_ptr<SearchBase> CreateSearch(UciResponder* responder,
                                           const OptionsDict* options)
      const override {
    return std::make_unique<ValueHeadSearch>(responder, options);
  }
  void PopulateParams(OptionsParser* parser) const override {
    parser->Add<IntOption>(kValueHeadDepthId, 1, 99) = 1;
  }
};

REGISTER_SEARCH(PolicyHeadFactory)
REGISTER_SEARCH(ValueHeadFactory)

}  // namespace
}  // namespace lczero

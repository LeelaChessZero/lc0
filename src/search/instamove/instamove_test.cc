/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#include "gtest/gtest.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/gamestate.h"
#include "chess/position.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "search/register.h"
#include "search/search.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {

class CapturingResponder : public UciResponder {
 public:
  void OutputBestMove(BestMoveInfo* info) override {
    bestmove_ = info->bestmove;
  }
  void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) override {
    infos_ = *infos;
  }

  const Move& bestmove() const { return bestmove_; }
  const std::vector<ThinkingInfo>& infos() const { return infos_; }

 private:
  Move bestmove_;
  std::vector<ThinkingInfo> infos_;
};

class MapBackendComputation : public BackendComputation {
 public:
  MapBackendComputation(const std::unordered_map<std::string, EvalResult>* evals,
                        EvalResult fallback)
      : evals_(evals), fallback_(std::move(fallback)) {}

  size_t UsedBatchSize() const override { return entries_.size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    entries_.push_back({PositionToFen(pos.pos.back()), result});
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    for (const auto& entry : entries_) {
      const auto it = evals_->find(entry.fen);
      const EvalResult& eval = it == evals_->end() ? fallback_ : it->second;
      if (entry.result.q) *entry.result.q = eval.q;
      if (entry.result.d) *entry.result.d = eval.d;
      if (entry.result.m) *entry.result.m = eval.m;
    }
  }

 private:
  struct Entry {
    std::string fen;
    EvalResultPtr result;
  };

  const std::unordered_map<std::string, EvalResult>* evals_;
  EvalResult fallback_;
  std::vector<Entry> entries_;
};

class MapBackend : public Backend {
 public:
  explicit MapBackend(std::unordered_map<std::string, EvalResult> evals,
                      EvalResult fallback)
      : evals_(std::move(evals)), fallback_(std::move(fallback)) {}

  BackendAttributes GetAttributes() const override {
    return BackendAttributes{
        .has_mlh = false,
        .has_wdl = true,
        .runs_on_cpu = true,
        .suggested_num_search_threads = 1,
        .recommended_batch_size = 16,
        .maximum_batch_size = 16,
    };
  }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<MapBackendComputation>(&evals_, fallback_);
  }

 private:
  std::unordered_map<std::string, EvalResult> evals_;
  EvalResult fallback_;
};

GameState MakeGameState(const std::string& fen,
                        const std::vector<std::string>& moves) {
  GameState state;
  state.startpos = Position::FromFen(fen);
  ChessBoard board = state.startpos.GetBoard();
  state.moves.reserve(moves.size());
  for (const auto& move : moves) {
    const Move parsed_move = board.ParseMove(move);
    state.moves.push_back(parsed_move);
    board.ApplyMove(parsed_move);
    board.Mirror();
  }
  return state;
}

std::string FenAfterMoves(const std::vector<std::string>& moves) {
  return PositionToFen(
      MakeGameState(ChessBoard::kStartposFen, moves).CurrentPosition());
}

struct SearchResult {
  std::string bestmove;
  std::vector<ThinkingInfo> infos;
};

SearchResult RunValueHeadSearch(int configured_depth,
                                std::optional<int> go_depth) {
  auto* factory = SearchManager::Get()->GetFactoryByName("valuehead");
  EXPECT_NE(factory, nullptr);

  OptionsParser parser;
  factory->PopulateParams(&parser);
  parser.SetUciOption("ValueHeadDepth", std::to_string(configured_depth));
  const OptionsDict& options = parser.GetOptionsDict();

  const std::unordered_map<std::string, EvalResult> evals = {
      {FenAfterMoves({"e2e4"}), EvalResult{.q = -0.3f, .d = 0.2f, .m = 0.0f}},
      {FenAfterMoves({"d2d4"}), EvalResult{.q = -0.2f, .d = 0.2f, .m = 0.0f}},
      {FenAfterMoves({"e2e4", "e7e5"}),
       EvalResult{.q = -0.9f, .d = 0.1f, .m = 0.0f}},
      {FenAfterMoves({"d2d4", "d7d5"}),
       EvalResult{.q = -0.2f, .d = 0.2f, .m = 0.0f}},
  };
  MapBackend backend(evals, EvalResult{.q = -0.1f, .d = 0.2f, .m = 0.0f});
  CapturingResponder responder;

  std::unique_ptr<SearchBase> search =
      factory->CreateSearch(&responder, &options);
  search->SetBackend(&backend);
  search->SetPosition(MakeGameState(ChessBoard::kStartposFen, {}));

  GoParams params;
  params.searchmoves = {"e2e4", "d2d4"};
  params.depth = go_depth;
  search->StartSearch(params);
  search->WaitSearch();

  return SearchResult{
      .bestmove = responder.bestmove().ToString(false),
      .infos = responder.infos(),
  };
}

TEST(ValueHeadSearch, UsesConfiguredDepthWhenGoDepthIsAbsent) {
  const SearchResult depth_one = RunValueHeadSearch(1, std::nullopt);
  EXPECT_EQ(depth_one.bestmove, "e2e4");
  ASSERT_EQ(depth_one.infos.size(), 1u);
  EXPECT_EQ(depth_one.infos.front().depth, 1);

  const SearchResult depth_two = RunValueHeadSearch(2, std::nullopt);
  EXPECT_EQ(depth_two.bestmove, "d2d4");
  ASSERT_EQ(depth_two.infos.size(), 1u);
  EXPECT_EQ(depth_two.infos.front().depth, 2);
}

TEST(ValueHeadSearch, GoDepthOverridesConfiguredDepth) {
  const SearchResult result = RunValueHeadSearch(1, 2);
  EXPECT_EQ(result.bestmove, "d2d4");
  ASSERT_EQ(result.infos.size(), 1u);
  EXPECT_EQ(result.infos.front().depth, 2);
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}

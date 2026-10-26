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

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/gamestate.h"
#include "chess/position.h"
#include "chess/uciloop.h"
#include "gtest/gtest.h"
#include "neural/backend.h"
#include "search/register.h"
#include "search/search.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {

using EvaluationFunction = std::function<EvalResult(std::span<const Position>)>;

class CapturingResponder : public UciResponder {
 public:
  void OutputBestMove(BestMoveInfo* info) override {
    bestmove_ = info->bestmove;
    ++bestmove_count_;
  }
  void OutputThinkingInfo(std::vector<ThinkingInfo>* infos) override {
    infos_ = *infos;
  }

  const Move& bestmove() const { return bestmove_; }
  size_t bestmove_count() const { return bestmove_count_; }
  const std::vector<ThinkingInfo>& infos() const { return infos_; }

 private:
  Move bestmove_;
  size_t bestmove_count_ = 0;
  std::vector<ThinkingInfo> infos_;
};

class FunctionBackendComputation : public BackendComputation {
 public:
  FunctionBackendComputation(const EvaluationFunction* evaluator,
                             size_t* evaluations, size_t* computations)
      : evaluator_(evaluator),
        evaluations_(evaluations),
        computations_(computations) {}

  size_t UsedBatchSize() const override { return entries_.size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    entries_.push_back(
        {std::vector<Position>(pos.pos.begin(), pos.pos.end()), result});
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    ++*computations_;
    for (const auto& entry : entries_) {
      const EvalResult eval = (*evaluator_)(entry.positions);
      if (entry.result.q) *entry.result.q = eval.q;
      if (entry.result.d) *entry.result.d = eval.d;
      if (entry.result.m) *entry.result.m = eval.m;
      ++*evaluations_;
    }
  }

 private:
  struct Entry {
    std::vector<Position> positions;
    EvalResultPtr result;
  };

  const EvaluationFunction* evaluator_;
  size_t* evaluations_;
  size_t* computations_;
  std::vector<Entry> entries_;
};

class FunctionBackend : public Backend {
 public:
  explicit FunctionBackend(EvaluationFunction evaluator,
                           int maximum_batch_size = 20000)
      : evaluator_(std::move(evaluator)),
        maximum_batch_size_(maximum_batch_size) {}

  BackendAttributes GetAttributes() const override {
    return BackendAttributes{
        .has_mlh = false,
        .has_wdl = true,
        .runs_on_cpu = true,
        .suggested_num_search_threads = 1,
        .recommended_batch_size = 256,
        .maximum_batch_size = maximum_batch_size_,
    };
  }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<FunctionBackendComputation>(
        &evaluator_, &evaluations_, &computations_);
  }

  size_t evaluations() const { return evaluations_; }
  size_t computations() const { return computations_; }

 private:
  EvaluationFunction evaluator_;
  int maximum_batch_size_;
  size_t evaluations_ = 0;
  size_t computations_ = 0;
};

class BlockingEvaluation {
 public:
  EvalResult Evaluate() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!entered_) {
      entered_ = true;
      entered_cv_.notify_one();
      release_cv_.wait_for(lock, std::chrono::seconds(1),
                           [&] { return released_; });
    }
    return EvalResult{.q = 0.0f, .d = 1.0f, .m = 0.0f};
  }

  bool WaitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    return entered_cv_.wait_for(lock, std::chrono::seconds(2),
                                [&] { return entered_; });
  }

  void Release() {
    std::lock_guard<std::mutex> lock(mutex_);
    released_ = true;
    release_cv_.notify_one();
  }

 private:
  std::mutex mutex_;
  std::condition_variable entered_cv_;
  std::condition_variable release_cv_;
  bool entered_ = false;
  bool released_ = false;
};

class BlockingInfoResponder : public UciResponder {
 public:
  void OutputBestMove(BestMoveInfo* info) override {
    std::lock_guard<std::mutex> lock(mutex_);
    bestmove_ = info->bestmove;
    events_.push_back("bestmove");
    bestmove_cv_.notify_one();
  }

  void OutputThinkingInfo(std::vector<ThinkingInfo>*) override {
    std::unique_lock<std::mutex> lock(mutex_);
    info_entered_ = true;
    info_entered_cv_.notify_one();
    release_info_cv_.wait(lock, [&] { return release_info_; });
    events_.push_back("info");
  }

  bool WaitUntilInfoEntered() {
    std::unique_lock<std::mutex> lock(mutex_);
    return info_entered_cv_.wait_for(lock, std::chrono::seconds(2),
                                     [&] { return info_entered_; });
  }

  bool WaitForBestMove(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex_);
    return bestmove_cv_.wait_for(lock, timeout, [&] {
      return std::find(events_.begin(), events_.end(), "bestmove") !=
             events_.end();
    });
  }

  void ReleaseInfo() {
    std::lock_guard<std::mutex> lock(mutex_);
    release_info_ = true;
    release_info_cv_.notify_one();
  }

  std::vector<std::string> events() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return events_;
  }

  Move bestmove() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bestmove_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable info_entered_cv_;
  std::condition_variable release_info_cv_;
  std::condition_variable bestmove_cv_;
  bool info_entered_ = false;
  bool release_info_ = false;
  Move bestmove_;
  std::vector<std::string> events_;
};

GameState MakeGameState(const std::string& fen,
                        const std::vector<std::string>& moves) {
  GameState state;
  state.startpos = Position::FromFen(fen);
  Position position = state.startpos;
  state.moves.reserve(moves.size());
  for (const auto& move : moves) {
    const Move parsed_move = position.GetBoard().ParseMove(move);
    state.moves.push_back(parsed_move);
    position = Position(position, parsed_move);
  }
  return state;
}

std::string FenAfterMoves(const std::string& fen,
                          const std::vector<std::string>& moves) {
  return PositionToFen(MakeGameState(fen, moves).CurrentPosition());
}

struct SearchResult {
  std::string bestmove;
  std::vector<ThinkingInfo> infos;
  size_t evaluations;
  size_t computations;
};

SearchResult RunValueHeadSearch(
    const std::string& fen, EvaluationFunction evaluator,
    std::optional<int> configured_depth = std::nullopt,
    std::optional<int> go_depth = std::nullopt, int max_nodes = 10000) {
  SearchFactory* factory = SearchManager::Get()->GetFactoryByName("valuehead");
  EXPECT_NE(factory, nullptr);

  OptionsParser parser;
  factory->PopulateParams(&parser);
  if (configured_depth) {
    parser.SetUciOption("ValueHeadDepth", std::to_string(*configured_depth));
  }
  parser.SetUciOption("ValueHeadMaxNodes", std::to_string(max_nodes));
  const OptionsDict& options = parser.GetOptionsDict();

  FunctionBackend backend(std::move(evaluator));
  CapturingResponder responder;
  std::unique_ptr<SearchBase> search =
      factory->CreateSearch(&responder, &options);
  search->SetBackend(&backend);
  search->SetPosition(MakeGameState(fen, {}));

  GoParams params;
  params.depth = go_depth;
  search->StartSearch(params);
  search->WaitSearch();

  return SearchResult{
      .bestmove = responder.bestmove().ToString(false),
      .infos = responder.infos(),
      .evaluations = backend.evaluations(),
      .computations = backend.computations(),
  };
}

TEST(ValueHeadSearch, PreservesDepthOneAsDefault) {
  const std::string e4_fen = FenAfterMoves(ChessBoard::kStartposFen, {"e2e4"});
  const std::string d4_fen = FenAfterMoves(ChessBoard::kStartposFen, {"d2d4"});
  const SearchResult result = RunValueHeadSearch(
      ChessBoard::kStartposFen, [=](std::span<const Position> positions) {
        const std::string fen = PositionToFen(positions.back());
        const float q = fen == e4_fen ? -0.5f : fen == d4_fen ? -0.2f : 0.0f;
        return EvalResult{.q = q, .d = 0.2f, .m = 0.0f};
      });

  EXPECT_EQ(result.bestmove, "e2e4");
  ASSERT_EQ(result.infos.size(), 1u);
  EXPECT_EQ(result.infos.front().depth, 1);
  EXPECT_EQ(result.infos.front().seldepth, 1);
  EXPECT_EQ(result.infos.front().nodes, 20);
  EXPECT_EQ(result.evaluations, 20u);
  EXPECT_EQ(result.computations, 1u);
}

TEST(ValueHeadSearch, GoDepthRunsFixedDepthMinimax) {
  const std::string e4_fen = FenAfterMoves(ChessBoard::kStartposFen, {"e2e4"});
  const std::string d4_fen = FenAfterMoves(ChessBoard::kStartposFen, {"d2d4"});
  const std::string e4_e5_fen =
      FenAfterMoves(ChessBoard::kStartposFen, {"e2e4", "e7e5"});
  const SearchResult result = RunValueHeadSearch(
      ChessBoard::kStartposFen,
      [=](std::span<const Position> positions) {
        const std::string root_child_fen = PositionToFen(positions[1]);
        const std::string leaf_fen = PositionToFen(positions.back());
        const float q = root_child_fen == e4_fen
                            ? (leaf_fen == e4_e5_fen ? -0.9f : 0.8f)
                        : root_child_fen == d4_fen ? -0.2f
                                                   : -0.8f;
        return EvalResult{.q = q, .d = 0.1f, .m = 0.0f};
      },
      1, 2);

  EXPECT_EQ(result.bestmove, "d2d4");
  ASSERT_EQ(result.infos.size(), 1u);
  EXPECT_EQ(result.infos.front().depth, 2);
  EXPECT_EQ(result.infos.front().seldepth, 2);
  EXPECT_EQ(result.infos.front().nodes, 420);
  EXPECT_EQ(result.evaluations, 400u);
  EXPECT_EQ(result.computations, 1u);
}

TEST(ValueHeadSearch, UsesConfiguredDepthWithoutGoDepth) {
  const SearchResult result = RunValueHeadSearch(
      ChessBoard::kStartposFen,
      [](std::span<const Position>) {
        return EvalResult{.q = 0.0f, .d = 1.0f, .m = 0.0f};
      },
      2);

  ASSERT_EQ(result.infos.size(), 1u);
  EXPECT_EQ(result.infos.front().depth, 2);
  EXPECT_EQ(result.infos.front().nodes, 420);
  EXPECT_EQ(result.evaluations, 400u);
}

TEST(ValueHeadSearch, PrefersCheckmateWithoutEvaluatingIt) {
  const std::string fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1";
  const SearchResult result =
      RunValueHeadSearch(fen, [](std::span<const Position>) {
        return EvalResult{.q = 1.0f, .d = 0.0f, .m = 0.0f};
      });

  const GameState state = MakeGameState(fen, {result.bestmove});
  const std::vector<Position> positions = state.GetPositions();
  const PositionHistory history(positions);
  EXPECT_NE(history.ComputeGameResult(), GameResult::UNDECIDED);
  EXPECT_NE(history.ComputeGameResult(), GameResult::DRAW);
  ASSERT_EQ(result.infos.size(), 1u);
  ASSERT_TRUE(result.infos.front().mate.has_value());
  EXPECT_EQ(*result.infos.front().mate, 1);
}

TEST(ValueHeadSearch, ReportsRuleFiftyDrawWithoutNeuralEvaluation) {
  const std::string fen = "7k/8/8/8/8/8/R7/K7 w - - 99 1";
  const SearchResult result =
      RunValueHeadSearch(fen, [](std::span<const Position>) {
        return EvalResult{.q = 0.8f, .d = 0.0f, .m = 0.0f};
      });

  ASSERT_EQ(result.infos.size(), 1u);
  ASSERT_TRUE(result.infos.front().wdl.has_value());
  EXPECT_EQ(result.infos.front().wdl->w, 0);
  EXPECT_EQ(result.infos.front().wdl->d, 1000);
  EXPECT_EQ(result.infos.front().wdl->l, 0);
  EXPECT_EQ(result.evaluations, 0u);
}

TEST(ValueHeadSearch, ReturnsMoveFromBlackPerspective) {
  const std::string fen =
      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1";
  const std::string e5_fen = FenAfterMoves(fen, {"e7e5"});
  const SearchResult result =
      RunValueHeadSearch(fen, [=](std::span<const Position> positions) {
        const float q =
            PositionToFen(positions.back()) == e5_fen ? -0.8f : 0.0f;
        return EvalResult{.q = q, .d = 0.1f, .m = 0.0f};
      });

  EXPECT_EQ(result.bestmove, "e7e5");
}

TEST(ValueHeadSearch, ReducesDepthToFitNodeLimit) {
  const SearchResult result = RunValueHeadSearch(
      ChessBoard::kStartposFen,
      [](std::span<const Position>) {
        return EvalResult{.q = 0.0f, .d = 1.0f, .m = 0.0f};
      },
      4, std::nullopt, 10000);

  ASSERT_EQ(result.infos.size(), 1u);
  EXPECT_EQ(result.infos.front().depth, 3);
  EXPECT_EQ(result.infos.front().nodes, 9322);
  EXPECT_EQ(result.evaluations, 8902u);
  EXPECT_EQ(result.computations, 1u);
}

TEST(ValueHeadSearch, NormalizesOutOfRangeGoDepth) {
  const auto evaluator = [](std::span<const Position>) {
    return EvalResult{.q = 0.0f, .d = 1.0f, .m = 0.0f};
  };
  const SearchResult zero_depth =
      RunValueHeadSearch(ChessBoard::kStartposFen, evaluator, 1, 0, 100);
  const SearchResult excessive_depth =
      RunValueHeadSearch(ChessBoard::kStartposFen, evaluator, 1, 100, 100);

  ASSERT_EQ(zero_depth.infos.size(), 1u);
  EXPECT_EQ(zero_depth.infos.front().depth, 1);
  ASSERT_EQ(excessive_depth.infos.size(), 1u);
  EXPECT_EQ(excessive_depth.infos.front().depth, 1);
  EXPECT_EQ(excessive_depth.evaluations, 20u);
  EXPECT_EQ(excessive_depth.computations, 1u);
}

TEST(ValueHeadSearch, ReturnsLegalMoveWhenDepthOneExceedsLimit) {
  const SearchResult result = RunValueHeadSearch(
      ChessBoard::kStartposFen,
      [](std::span<const Position>) {
        return EvalResult{.q = 0.0f, .d = 1.0f, .m = 0.0f};
      },
      2, std::nullopt, 1);

  const ChessBoard& board =
      Position::FromFen(ChessBoard::kStartposFen).GetBoard();
  const Move move = board.ParseMove(result.bestmove);
  const MoveList legal_moves = board.GenerateLegalMoves();
  EXPECT_NE(std::find(legal_moves.begin(), legal_moves.end(), move),
            legal_moves.end());
  ASSERT_EQ(result.infos.size(), 1u);
  EXPECT_EQ(result.infos.front().depth, 1);
  EXPECT_EQ(result.infos.front().nodes, 0);
  EXPECT_EQ(result.evaluations, 0u);
  EXPECT_EQ(result.computations, 0u);
}

TEST(ValueHeadSearch, DoesNotOutputInfoAfterBestMoveWhenStopRaces) {
  SearchFactory* factory = SearchManager::Get()->GetFactoryByName("valuehead");
  ASSERT_NE(factory, nullptr);

  OptionsParser parser;
  factory->PopulateParams(&parser);
  const OptionsDict& options = parser.GetOptionsDict();
  const std::string e4_fen = FenAfterMoves(ChessBoard::kStartposFen, {"e2e4"});
  FunctionBackend backend([=](std::span<const Position> positions) {
    const float q = PositionToFen(positions.back()) == e4_fen ? -0.8f : 0.0f;
    return EvalResult{.q = q, .d = 0.1f, .m = 0.0f};
  });
  BlockingInfoResponder responder;
  std::unique_ptr<SearchBase> search =
      factory->CreateSearch(&responder, &options);
  search->SetBackend(&backend);
  search->SetPosition(MakeGameState(ChessBoard::kStartposFen, {}));

  search->StartSearch(GoParams{});
  ASSERT_TRUE(responder.WaitUntilInfoEntered());
  std::thread stopper([&] { search->StopSearch(); });
  EXPECT_FALSE(responder.WaitForBestMove(std::chrono::milliseconds(100)));

  responder.ReleaseInfo();
  stopper.join();
  search->WaitSearch();
  EXPECT_EQ(responder.events(), (std::vector<std::string>{"info", "bestmove"}));
  EXPECT_EQ(responder.bestmove().ToString(false), "e2e4");
}

TEST(ValueHeadSearch, StartIsNonBlockingAndStopCancelsEvaluation) {
  SearchFactory* factory = SearchManager::Get()->GetFactoryByName("valuehead");
  ASSERT_NE(factory, nullptr);

  OptionsParser parser;
  factory->PopulateParams(&parser);
  const OptionsDict& options = parser.GetOptionsDict();
  BlockingEvaluation blocking_evaluation;
  FunctionBackend backend(
      [&](std::span<const Position>) { return blocking_evaluation.Evaluate(); },
      1);
  CapturingResponder responder;
  std::unique_ptr<SearchBase> search =
      factory->CreateSearch(&responder, &options);
  search->SetBackend(&backend);
  search->SetPosition(MakeGameState(ChessBoard::kStartposFen, {}));

  const auto start = std::chrono::steady_clock::now();
  search->StartSearch(GoParams{});
  const auto start_duration = std::chrono::steady_clock::now() - start;
  ASSERT_TRUE(blocking_evaluation.WaitUntilEntered());
  EXPECT_LT(start_duration, std::chrono::milliseconds(500));

  search->StopSearch();
  EXPECT_EQ(responder.bestmove_count(), 1u);
  blocking_evaluation.Release();
  search->WaitSearch();

  EXPECT_EQ(backend.evaluations(), 1u);
  search->StopSearch();
  EXPECT_EQ(responder.bestmove_count(), 1u);
}

TEST(ValueHeadSearch, AbortCancelsWithoutResponding) {
  SearchFactory* factory = SearchManager::Get()->GetFactoryByName("valuehead");
  ASSERT_NE(factory, nullptr);

  OptionsParser parser;
  factory->PopulateParams(&parser);
  const OptionsDict& options = parser.GetOptionsDict();
  BlockingEvaluation blocking_evaluation;
  FunctionBackend backend(
      [&](std::span<const Position>) { return blocking_evaluation.Evaluate(); },
      1);
  CapturingResponder responder;
  std::unique_ptr<SearchBase> search =
      factory->CreateSearch(&responder, &options);
  search->SetBackend(&backend);
  search->SetPosition(MakeGameState(ChessBoard::kStartposFen, {}));

  search->StartSearch(GoParams{});
  ASSERT_TRUE(blocking_evaluation.WaitUntilEntered());
  search->AbortSearch();
  blocking_evaluation.Release();
  search->WaitSearch();

  EXPECT_EQ(backend.evaluations(), 1u);
  EXPECT_EQ(responder.bestmove_count(), 0u);
}

TEST(ValueHeadSearch, BackendFailureReturnsLegalMoveOnlyOnce) {
  SearchFactory* factory = SearchManager::Get()->GetFactoryByName("valuehead");
  ASSERT_NE(factory, nullptr);

  OptionsParser parser;
  factory->PopulateParams(&parser);
  const OptionsDict& options = parser.GetOptionsDict();
  FunctionBackend backend([](std::span<const Position>) -> EvalResult {
    throw Exception("Expected backend failure.");
  });
  CapturingResponder responder;
  std::unique_ptr<SearchBase> search =
      factory->CreateSearch(&responder, &options);
  search->SetBackend(&backend);
  search->SetPosition(MakeGameState(ChessBoard::kStartposFen, {}));

  EXPECT_NO_THROW(search->StartSearch(GoParams{}));
  search->WaitSearch();
  EXPECT_EQ(responder.bestmove_count(), 1u);
  const ChessBoard& board =
      Position::FromFen(ChessBoard::kStartposFen).GetBoard();
  const MoveList legal_moves = board.GenerateLegalMoves();
  EXPECT_NE(
      std::find(legal_moves.begin(), legal_moves.end(), responder.bestmove()),
      legal_moves.end());

  search->StopSearch();
  EXPECT_EQ(responder.bestmove_count(), 1u);
}

TEST(ValueHeadSearch, BackendFailureDuringInfiniteWaitsForStop) {
  SearchFactory* factory = SearchManager::Get()->GetFactoryByName("valuehead");
  ASSERT_NE(factory, nullptr);

  OptionsParser parser;
  factory->PopulateParams(&parser);
  const OptionsDict& options = parser.GetOptionsDict();
  FunctionBackend backend([](std::span<const Position>) -> EvalResult {
    throw Exception("Expected backend failure.");
  });
  CapturingResponder responder;
  std::unique_ptr<SearchBase> search =
      factory->CreateSearch(&responder, &options);
  search->SetBackend(&backend);
  search->SetPosition(MakeGameState(ChessBoard::kStartposFen, {}));

  GoParams params;
  params.infinite = true;
  search->StartSearch(params);
  search->WaitSearch();
  EXPECT_EQ(responder.bestmove_count(), 0u);

  search->StopSearch();
  EXPECT_EQ(responder.bestmove_count(), 1u);
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}

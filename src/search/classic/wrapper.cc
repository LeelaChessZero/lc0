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

#include "chess/gamestate.h"
#include "search/classic/search.h"
#include "search/classic/stoppers/factory.h"
#include "search/register.h"
#include "search/search.h"
#include "neural/shared_params.h"
#include "utils/trace.h"

namespace lczero {
namespace classic {
namespace {

const OptionId kThreadsOptionId{
    {.long_flag = "threads",
     .uci_option = "Threads",
     .help_text =
         "Number of (CPU) worker threads to use, 0 for the backend default.",
     .short_flag = 't',
     .visibility = OptionId::kAlwaysVisible}};
const OptionId kClearTree{
    {.long_flag = "",
     .uci_option = "ClearTree",
     .help_text = "Clear the tree before the next search.",
     .visibility = OptionId::kProOnly}};

class ClassicSearch : public SearchBase {
 public:
  ClassicSearch(UciResponder* responder, const OptionsDict* options)
      : SearchBase(responder), options_(options) {}

 private:
  void NewGame() override;
  void SetPosition(const GameState& pos) override;
  void StartSearch(const GoParams&) override;
  void StartClock() override {
    move_start_time_ = std::chrono::steady_clock::now();
  }
  void WaitSearch() override {
    if (search_) search_->Wait();
  }
  void StopSearch() override {
    if (search_) search_->Stop();
  }
  void AbortSearch() override {
    if (search_) search_->Abort();
  }

  const OptionsDict* options_;
  std::unique_ptr<TimeManager> time_manager_;
  std::unique_ptr<Search> search_;
  std::unique_ptr<NodeTree> tree_;
  std::optional<std::chrono::steady_clock::time_point> move_start_time_;
};

MoveList StringsToMovelist(const std::vector<std::string>& moves,
                           const ChessBoard& board) {
  MoveList result;
  if (moves.size()) {
    result.reserve(moves.size());
    const auto legal_moves = board.GenerateLegalMoves();
    for (const auto& move : moves) {
      const Move m = board.ParseMove(move);
      if (std::find(legal_moves.begin(), legal_moves.end(), m) !=
          legal_moves.end()) {
        result.emplace_back(m);
      }
    }
    if (result.empty()) throw Exception("No legal searchmoves.");
  }
  return result;
}

void ClassicSearch::NewGame() {
  LCTRACE_FUNCTION_SCOPE;
  search_.reset();
  tree_.reset();
  time_manager_ = MakeTimeManager(*options_);
}

void ClassicSearch::SetPosition(const GameState& pos) {
  LCTRACE_FUNCTION_SCOPE;
  if (!tree_) tree_ = std::make_unique<NodeTree>();
  const bool is_same_game = tree_->ResetToPosition(pos);
  if (!is_same_game) time_manager_ = MakeTimeManager(*options_);
}

void ClassicSearch::StartSearch(const GoParams& params) {
  LCTRACE_FUNCTION_SCOPE;
  auto forwarder =
      std::make_unique<NonOwningUciRespondForwarder>(uci_responder_);
  if (options_->Get<Button>(kClearTree).TestAndReset()) tree_->TrimTreeAtHead();

  const auto cache_size =
      options_->Get<int>(SharedBackendParams::kNNCacheSizeId);
  const size_t kAvgNodeSize =
      sizeof(Node) + MemoryWatchingStopper::kAvgMovesPerPosition * sizeof(Edge);
  const size_t kAvgCacheItemSize =
      3 * sizeof(float) + sizeof(std::unique_ptr<float[]>) +
      sizeof(float[MemoryWatchingStopper::kAvgMovesPerPosition]);
  size_t total_memory = tree_.get()->GetCurrentHead()->GetN() * kAvgNodeSize +
                        cache_size * kAvgCacheItemSize;
  auto stopper = time_manager_->GetStopper(
      params, tree_.get()->HeadPosition(), total_memory, kAvgNodeSize,
      tree_.get()->GetCurrentHead()->GetN());
  search_ = std::make_unique<Search>(
      *tree_, backend_, std::move(forwarder),
      StringsToMovelist(params.searchmoves, tree_->HeadPosition().GetBoard()),
      *move_start_time_, std::move(stopper), params.infinite, params.ponder,
      *options_, syzygy_tb_);

  LOGFILE << "Timer started at "
          << FormatTime(SteadyClockToSystemClock(*move_start_time_));
  search_->StartThreads(options_->Get<int>(kThreadsOptionId));
}

class ClassicSearchFactory : public SearchFactory {
  std::string_view GetName() const override { return "classic"; }
  std::unique_ptr<SearchBase> CreateSearch(
      UciResponder* responder, const OptionsDict* options) const override {
    LCTRACE_FUNCTION_SCOPE;
    return std::make_unique<ClassicSearch>(responder, options);
  }

  void PopulateParams(OptionsParser* parser) const override {
    parser->Add<IntOption>(kThreadsOptionId, 0, 128) = 0;
    SearchParams::Populate(parser);
    PopulateTimeManagementOptions(RunType::kUci, parser);

    parser->Add<ButtonOption>(kClearTree);
  }
};

REGISTER_SEARCH(ClassicSearchFactory);

}  // namespace
}  // namespace classic
}  // namespace lczero

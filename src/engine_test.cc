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

#include "engine.h"

#include <future>

#include "gtest/gtest.h"
#include "neural/mock_backend.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "search/mock_search.h"
#include "search/search.h"

namespace lczero {
namespace {

using testing::_;
using testing::Invoke;
using testing::Return;

class EngineTest : public ::testing::Test {
 protected:
  EngineTest() {
    Engine::PopulateOptions(&options_parser_);
    SharedBackendParams::Populate(&options_parser_);
    options_ = options_parser_.GetMutableOptions();
    auto backend_factory = std::make_unique<MockBackendFactory>();
    backend_factory_ = backend_factory.get();
    ON_CALL(*backend_factory_, GetName()).WillByDefault(Return("mock"));
    ON_CALL(*backend_factory_, Create(_))
        .WillByDefault([this](const OptionsDict&) {
          auto backend = std::make_unique<MockBackend>();
          backend_ = backend.get();
          return backend;
        });

    BackendManager::Get()->AddBackend(std::move(backend_factory));

    options_->Set<std::string>(SharedBackendParams::kBackendId, "mock");
    options_->Set<int>(SharedBackendParams::kNNCacheSizeId, 10);

    EXPECT_CALL(search_factory_, CreateSearch(_, _))
        .WillOnce([&](UciResponder* responder, const OptionsDict*) {
          auto search = std::make_unique<MockSearch>(responder);
          search_ = search.get();
          return search;
        });
  }

  ~EngineTest() { BackendManager::Get()->RemoveBackend(backend_factory_); }

  OptionsParser options_parser_;
  OptionsDict* options_ = nullptr;  // absl_notnull
  MockBackend* backend_ = nullptr;
  MockBackendFactory* backend_factory_ = nullptr;
  MockSearchFactory search_factory_;
  MockSearch* search_ = nullptr;
  std::unique_ptr<Engine> engine_;
};

class WaitingUciResponder : public UciResponder {
 public:
  WaitingUciResponder() {
    bestmove_promise_ = std::make_unique<std::promise<void>>();
  }

  virtual void OutputBestMove(BestMoveInfo*) override {
    bestmove_promise_->set_value();
  }
  virtual void OutputThinkingInfo(std::vector<ThinkingInfo>*) override {}

  void Wait() {
    bestmove_promise_->get_future().wait();
    bestmove_promise_ = std::make_unique<std::promise<void>>();
  }

 private:
  std::unique_ptr<std::promise<void>> bestmove_promise_;
};

TEST_F(EngineTest, BackendReloadByUpdateBackendConfig) {
  WaitingUciResponder uci_responder;
  Engine engine(search_factory_, *options_);
  engine.RegisterUciResponder(&uci_responder);
  EXPECT_EQ(backend_, nullptr);  // Backend not created before the search.
  EXPECT_CALL(*search_, StartSearch(_)).WillRepeatedly([&](const GoParams&) {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    static BestMoveInfo bestmove_info(Move::White(kSquareE1, kSquareA1));
    search_->GetUciResponder()->OutputBestMove(&bestmove_info);
  });
  engine.Go(GoParams{.nodes = 10});
  uci_responder.Wait();
  EXPECT_NE(backend_, nullptr);  // Backend created after the search.
  Backend* prev_backend = backend_;
  EXPECT_CALL(*backend_, UpdateConfiguration(_))
      .WillOnce(Return(Backend::UPDATE_OK));
  engine.NewGame();
  engine.Go(GoParams{.nodes = 10});
  uci_responder.Wait();
  EXPECT_EQ(backend_, prev_backend);  // Backend not recreated.
  EXPECT_CALL(*backend_, UpdateConfiguration(_))
      .WillOnce(Return(Backend::NEED_RESTART));
  engine.Go(GoParams{.nodes = 10});  // Go alone should not restart the backend.
  uci_responder.Wait();
  EXPECT_EQ(backend_, prev_backend);
  engine.NewGame();
  EXPECT_NE(backend_, prev_backend);  // Backend recreated.
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

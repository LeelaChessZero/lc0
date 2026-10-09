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
*/

#include <atomic>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "neural/backend.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/exception.h"

namespace lczero {
namespace {

struct ChildConfiguration {
  int threads = 0;
  int max_batch = 0;
  int batch_step = 0;
  std::string weights;
  float temperature = 0;
  std::string history;
  int updates = 0;
  std::atomic<int> computations = 0;
};

class TestComputation : public BackendComputation {
 public:
  explicit TestComputation(size_t capacity) : capacity_(capacity) {}

  size_t UsedBatchSize() const override { return entries_.size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    EXPECT_LT(entries_.size(), capacity_);
    entries_.push_back(
        {result, static_cast<float>(pos.pos.back().GetGamePly()),
         static_cast<float>(pos.legal_moves.front().raw_data())});
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    for (const auto& entry : entries_) {
      if (entry.result.q) *entry.result.q = entry.ply;
      if (entry.result.d) *entry.result.d = entry.move;
    }
  }

 private:
  struct Entry {
    EvalResultPtr result;
    float ply;
    float move;
  };
  size_t capacity_;
  std::vector<Entry> entries_;
};

class TestBackend : public Backend {
 public:
  TestBackend(BackendAttributes attributes,
              std::shared_ptr<ChildConfiguration> config,
              const OptionsDict& options)
      : attributes_(attributes), config_(std::move(config)) {
    UpdateConfiguration(options);
  }

  BackendAttributes GetAttributes() const override { return attributes_; }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    ++config_->computations;
    return std::make_unique<TestComputation>(attributes_.maximum_batch_size);
  }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    Backend::UpdateConfiguration(options);
    config_->weights =
        options.Get<std::string>(SharedBackendParams::kWeightsId);
    config_->temperature =
        options.Get<float>(SharedBackendParams::kPolicySoftmaxTemp);
    config_->history =
        options.Get<std::string>(SharedBackendParams::kHistoryFill);
    ++config_->updates;
    return UPDATE_OK;
  }

 private:
  BackendAttributes attributes_;
  std::shared_ptr<ChildConfiguration> config_;
};

class TestBackendFactory : public BackendFactory {
 public:
  // Make the no-children default selection independent of compiled backends.
  int GetPriority() const override { return 1000000; }
  std::string_view GetName() const override { return "demux-test"; }

  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    OptionsDict child_options;
    child_options.AddSubdictFromString(
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId));
    const int capacity = child_options.GetOrDefault<int>("capacity", 8);
    auto config = std::make_shared<ChildConfiguration>();
    config->max_batch = child_options.GetOrDefault<int>("max_batch", 0);
    config->batch_step = child_options.GetOrDefault<int>("batch_step", 0);
    config->threads = child_options.GetOrDefault<int>("threads", 0);
    BackendAttributes attributes{
        .has_mlh = child_options.GetOrDefault<bool>("mlh", false),
        .has_wdl = child_options.GetOrDefault<bool>("wdl", true),
        .runs_on_cpu = child_options.GetOrDefault<bool>("cpu", true),
        .suggested_num_search_threads =
            child_options.GetOrDefault<int>("search_threads", 1),
        .recommended_batch_size =
            child_options.GetOrDefault<int>("recommended", capacity),
        .maximum_batch_size = config->max_batch ? config->max_batch : capacity};
    child_options.CheckAllOptionsRead("demux-test");
    children.push_back(config);
    return std::make_unique<TestBackend>(attributes, config, options);
  }

  std::vector<std::shared_ptr<ChildConfiguration>> children;
};

class DemuxTest : public testing::Test {
 protected:
  void SetUp() override {
    auto factory = std::make_unique<TestBackendFactory>();
    factory_ = factory.get();
    BackendManager::Get()->AddBackend(std::move(factory));
  }

  void TearDown() override { BackendManager::Get()->RemoveBackend(factory_); }

  static OptionsDict Options(const std::string& backend_options) {
    OptionsDict options;
    options.Set<std::string>(SharedBackendParams::kBackendId, "demux");
    options.Set<std::string>(SharedBackendParams::kBackendOptionsId,
                             backend_options);
    options.Set<std::string>(SharedBackendParams::kWeightsId, "parent-weights");
    options.Set<float>(SharedBackendParams::kPolicySoftmaxTemp, 1.359f);
    options.Set<std::string>(SharedBackendParams::kHistoryFill, "fen_only");
    return options;
  }

  std::unique_ptr<Backend> Create(const std::string& backend_options) {
    return BackendManager::Get()->CreateFromParams(Options(backend_options));
  }

  static void AddInput(BackendComputation* computation, int ply,
                       EvalResultPtr result) {
    std::vector<Position> positions{Position(ChessBoard{}, 0, ply)};
    MoveList moves{Move::White(Square::FromIdx(12), Square::FromIdx(28))};
    EXPECT_EQ(computation->AddInput({positions, moves}, result),
              BackendComputation::ENQUEUED_FOR_EVAL);
    // Destroy and change the caller's input before ComputeBlocking().
    positions[0] = Position(ChessBoard{}, 0, 999);
    moves[0] = Move{};
  }

  TestBackendFactory* factory_ = nullptr;
};

TEST_F(DemuxTest, RejectsNonpositiveBatchStepBeforeCreatingChildren) {
  EXPECT_THROW(Create("backend=demux-test,batch_step=0"), Exception);
  EXPECT_THROW(Create("backend=demux-test,batch_step=-1,first(),second()"),
               Exception);
  EXPECT_TRUE(factory_->children.empty());
}

TEST_F(DemuxTest, OwnsDeferredInputs) {
  auto backend = Create("backend=demux-test,threads=1");
  auto computation = backend->CreateComputation();
  EvalResult result{};
  AddInput(computation.get(), 7, result.AsPtr());
  EXPECT_EQ(computation->UsedBatchSize(), 1);
  computation->ComputeBlocking();
  EXPECT_EQ(result.q, 7);
  EXPECT_EQ(result.d,
            Move::White(Square::FromIdx(12), Square::FromIdx(28)).raw_data());
}

TEST_F(DemuxTest, CollectsConcurrentlyWithinAdvertisedCapacity) {
  auto backend = Create("backend=demux-test,(capacity=64),(capacity=64)");
  const size_t capacity = backend->GetAttributes().maximum_batch_size;
  ASSERT_EQ(capacity, 128);
  for (int repetition = 0; repetition < 20; ++repetition) {
    auto computation = backend->CreateComputation();
    std::vector<EvalResult> results(capacity);
    std::vector<std::thread> threads;
    for (size_t t = 0; t < 8; ++t) {
      threads.emplace_back([&, t] {
        for (size_t i = t; i < capacity; i += 8) {
          AddInput(computation.get(), i, results[i].AsPtr());
        }
      });
    }
    for (auto& thread : threads) thread.join();
    EXPECT_EQ(computation->UsedBatchSize(), capacity);
    computation->ComputeBlocking();
    for (size_t i = 0; i < capacity; ++i) EXPECT_EQ(results[i].q, i);
  }
}

TEST_F(DemuxTest, ChunksRoundedAssignmentsWithinSummedCapacity) {
  auto backend =
      Create("backend=demux-test,batch_step=3,(capacity=2),(capacity=5)");
  ASSERT_EQ(backend->GetAttributes().maximum_batch_size, 7);
  auto computation = backend->CreateComputation();
  std::vector<EvalResult> results(7);
  for (size_t i = 0; i < results.size(); ++i) {
    AddInput(computation.get(), i, results[i].AsPtr());
  }
  computation->ComputeBlocking();
  for (size_t i = 0; i < results.size(); ++i) EXPECT_EQ(results[i].q, i);
  EXPECT_EQ(factory_->children[0]->computations, 3);
  EXPECT_EQ(factory_->children[1]->computations, 1);
}

TEST_F(DemuxTest, PreservesAttributeFormulas) {
  auto backend = Create(
      "backend=demux-test,(capacity=8,recommended=3,search_threads=2,mlh=true),"
      "(capacity=10,recommended=5,search_threads=4,cpu=false,wdl=false)");
  const auto attributes = backend->GetAttributes();
  EXPECT_EQ(attributes.maximum_batch_size, 18);
  EXPECT_EQ(attributes.recommended_batch_size, 6);
  EXPECT_EQ(attributes.suggested_num_search_threads, 1);
  EXPECT_FALSE(attributes.has_mlh);
  EXPECT_FALSE(attributes.has_wdl);
  EXPECT_FALSE(attributes.runs_on_cpu);
}

TEST_F(DemuxTest, PreservesChildSelection) {
  for (const auto& options : {"", "demux-test()", "backend=demux-test,named()",
                              "named(backend=demux-test)"}) {
    auto backend = Create(options);
    EXPECT_EQ(backend->GetAttributes().maximum_batch_size, 8);
  }
  EXPECT_EQ(factory_->children.size(), 4);
  // Fail before starting workers; constructor-unwind hardening is out of scope.
  EXPECT_THROW(Create("unknown-demux-child()"), Exception);
  EXPECT_THROW(Create("backend=unknown-demux-child,named()"), Exception);
}

TEST_F(DemuxTest, ForwardsThreadsAndChildOptions) {
  auto backend = Create(
      "backend=demux-test,threads=2,batch_step=3,max_batch=12,"
      "first(),second(threads=1,batch_step=5,max_batch=9)");
  ASSERT_EQ(factory_->children.size(), 2);
  EXPECT_EQ(factory_->children[0]->threads, 2);
  EXPECT_EQ(factory_->children[1]->threads, 1);
  EXPECT_EQ(factory_->children[0]->batch_step, 0);
  EXPECT_EQ(factory_->children[1]->batch_step, 5);
  EXPECT_EQ(factory_->children[0]->max_batch, 12);
  EXPECT_EQ(factory_->children[1]->max_batch, 9);
  EXPECT_EQ(backend->GetAttributes().maximum_batch_size, 21);
}

TEST_F(DemuxTest, FiltersOnlyRootBatchStepWithoutChildren) {
  auto backend =
      Create("backend=demux-test,threads=1,batch_step=3,max_batch=12");
  ASSERT_EQ(factory_->children.size(), 1);
  EXPECT_EQ(factory_->children[0]->threads, 1);
  EXPECT_EQ(factory_->children[0]->batch_step, 0);
  EXPECT_EQ(factory_->children[0]->max_batch, 12);
  EXPECT_EQ(backend->GetAttributes().maximum_batch_size, 12);
}

TEST_F(DemuxTest, InitializesAndUpdatesSharedConfiguration) {
  auto options = Options("backend=demux-test,first(),second()");
  auto backend = BackendManager::Get()->CreateFromParams(options);
  EXPECT_TRUE(backend->IsSameConfiguration(options));
  ASSERT_EQ(factory_->children.size(), 2);
  for (const auto& child : factory_->children) {
    EXPECT_EQ(child->weights, "parent-weights");
    EXPECT_EQ(child->temperature, 1.359f);
    EXPECT_EQ(child->history, "fen_only");
  }
  options.Set<float>(SharedBackendParams::kPolicySoftmaxTemp, 2.0f);
  options.Set<std::string>(SharedBackendParams::kHistoryFill, "always");
  EXPECT_FALSE(backend->IsSameConfiguration(options));
  EXPECT_EQ(backend->UpdateConfiguration(options), Backend::UPDATE_OK);
  EXPECT_TRUE(backend->IsSameConfiguration(options));
  for (const auto& child : factory_->children) {
    EXPECT_EQ(child->temperature, 2.0f);
    EXPECT_EQ(child->history, "always");
  }
  options.Set<std::string>(SharedBackendParams::kWeightsId, "changed-weights");
  EXPECT_EQ(backend->UpdateConfiguration(options), Backend::NEED_RESTART);
  options.Set<std::string>(SharedBackendParams::kWeightsId, "parent-weights");
  options.Set<std::string>(SharedBackendParams::kBackendOptionsId,
                           "backend=demux-test,threads=2");
  EXPECT_EQ(backend->UpdateConfiguration(options), Backend::NEED_RESTART);
}

TEST_F(DemuxTest, EmptyComputationCompletes) {
  auto backend = Create("");
  auto computation = backend->CreateComputation();
  EXPECT_EQ(computation->UsedBatchSize(), 0);
  computation->ComputeBlocking();
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

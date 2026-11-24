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
#include "utils/atomic_vector.h"

namespace lczero {
namespace {

class TestComputation : public BackendComputation {
 public:
  TestComputation(int id, size_t capacity, std::atomic<size_t>& inputs)
      : id_(id), entries_(capacity), inputs_(inputs) {}
  size_t UsedBatchSize() const override { return entries_.size(); }
  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    entries_.emplace_back(Entry{result, pos.pos.back().GetGamePly()});
    ++inputs_;
    return ENQUEUED_FOR_EVAL;
  }
  void ComputeBlocking() override {
    for (const auto& entry : entries_) {
      *entry.result.q = entry.ply;
      *entry.result.d = id_;
    }
  }

 private:
  struct Entry {
    EvalResultPtr result;
    int ply;
  };
  int id_;
  AtomicVector<Entry> entries_;
  std::atomic<size_t>& inputs_;
};

class TestBackend : public Backend {
 public:
  TestBackend(int id, int capacity) : id_(id), capacity_(capacity) {}
  BackendAttributes GetAttributes() const override {
    return {.has_mlh = false,
            .has_wdl = true,
            .runs_on_cpu = true,
            .suggested_num_search_threads = 1,
            .recommended_batch_size = capacity_,
            .maximum_batch_size = capacity_};
  }
  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<TestComputation>(id_, capacity_, inputs);
  }
  std::atomic<size_t> inputs = 0;

 private:
  int id_;
  int capacity_;
};

class TestBackendFactory : public BackendFactory {
 public:
  int GetPriority() const override { return 1000000; }
  std::string_view GetName() const override { return "demux-test"; }
  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    OptionsDict child;
    child.AddSubdictFromString(
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId));
    auto backend = std::make_unique<TestBackend>(child.Get<int>("id"),
                                                 child.Get<int>("capacity"));
    children.push_back(backend.get());
    return backend;
  }
  std::vector<TestBackend*> children;
};

class DemuxTest : public testing::Test {
 protected:
  void SetUp() override {
    auto factory = std::make_unique<TestBackendFactory>();
    factory_ = factory.get();
    BackendManager::Get()->AddBackend(std::move(factory));
  }
  void TearDown() override { BackendManager::Get()->RemoveBackend(factory_); }
  std::unique_ptr<Backend> Create(const std::string& backend_options) {
    OptionsDict options;
    options.Set<std::string>(SharedBackendParams::kBackendId, "demux");
    options.Set<std::string>(SharedBackendParams::kBackendOptionsId,
                             backend_options);
    options.Set<std::string>(SharedBackendParams::kWeightsId, "parent-weights");
    options.Set<float>(SharedBackendParams::kPolicySoftmaxTemp, 1.359f);
    options.Set<std::string>(SharedBackendParams::kHistoryFill, "fen_only");
    return BackendManager::Get()->CreateFromParams(options);
  }
  static void AddInput(BackendComputation* computation, int ply,
                       EvalResultPtr result) {
    std::vector<Position> positions{Position(ChessBoard{}, 0, ply)};
    MoveList moves{Move::White(Square::FromIdx(12), Square::FromIdx(28))};
    EXPECT_EQ(computation->AddInput({positions, moves}, result),
              BackendComputation::ENQUEUED_FOR_EVAL);
    // Child capture must survive mutation and destruction before evaluation.
    positions[0] = Position(ChessBoard{}, 0, 999);
  }
  TestBackendFactory* factory_ = nullptr;
};

TEST_F(DemuxTest, GroupedRoutingRespectsConservativeCapacity) {
  for (int group : {3, 1}) {
    factory_->children.clear();
    auto backend =
        Create("batch_step=" + std::to_string(group) +
               ",a(backend=demux-test,id=0,capacity=5,demux_threads=1),"
               "b(backend=demux-test,id=1,capacity=9,demux_threads=1)");
    const int capacity = group == 3 ? 8 : 10;
    EXPECT_EQ(backend->GetAttributes().maximum_batch_size, capacity);
    for (int start = 0; start < 2; ++start) {
      for (auto* child : factory_->children) child->inputs = 0;
      auto computation = backend->CreateComputation();
      std::vector<EvalResult> results(capacity);
      size_t expected[2] = {};
      for (int i = 0; i < capacity; ++i) {
        AddInput(computation.get(), i, results[i].AsPtr());
        ++expected[(start + i / group) % 2];
      }
      // Routing happens in AddInput, not when ComputeBlocking is called.
      EXPECT_EQ(factory_->children[0]->inputs.load(), expected[0]);
      EXPECT_EQ(factory_->children[1]->inputs.load(), expected[1]);
      EXPECT_EQ(computation->UsedBatchSize(), capacity);
      computation->ComputeBlocking();
      for (int i = 0; i < capacity; ++i) {
        EXPECT_EQ(results[i].q, i);
        EXPECT_EQ(results[i].d, (start + i / group) % 2);
      }
    }
  }
}

TEST_F(DemuxTest, CollectsConcurrentlyWithinAdvertisedCapacity) {
  auto backend = Create(
      "batch_step=3,a(backend=demux-test,id=0,capacity=16,demux_threads=1),"
      "b(backend=demux-test,id=1,capacity=16,demux_threads=1)");
  const int capacity = backend->GetAttributes().maximum_batch_size;
  ASSERT_EQ(capacity, 31);
  auto computation = backend->CreateComputation();
  std::vector<EvalResult> results(capacity);
  std::vector<std::thread> producers;
  for (int thread = 0; thread < 3; ++thread) {
    producers.emplace_back([&, thread] {
      for (int i = thread; i < capacity; i += 3) {
        AddInput(computation.get(), i, results[i].AsPtr());
      }
    });
  }
  for (auto& producer : producers) producer.join();
  EXPECT_EQ(computation->UsedBatchSize(), capacity);
  EXPECT_EQ(factory_->children[0]->inputs.load(), 16);
  EXPECT_EQ(factory_->children[1]->inputs.load(), 15);
  computation->ComputeBlocking();
  for (int i = 0; i < capacity; ++i) {
    EXPECT_EQ(results[i].q, i);
    EXPECT_TRUE(results[i].d == 0 || results[i].d == 1);
  }
}

}  // namespace
}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

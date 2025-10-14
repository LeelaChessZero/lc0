/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors

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
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <queue>
#include <thread>

#include "neural/factory.h"

namespace lczero {
namespace {

class DemuxingComputation;

struct DemuxingWork {
  DemuxingComputation* source_ = nullptr;
  std::unique_ptr<NetworkComputation> computation_;
  int start_ = 0;
  int end_ = 0;

  DemuxingWork(int sample) : end_(sample) {}
  DemuxingWork(DemuxingComputation* source, int start, int end)
      : source_(source), start_(start), end_(end) {
    assert(start_ != end_);
  }

  auto operator<=>(const DemuxingWork& b) const { return end_ <=> b.end_; }
};

class DemuxingNetwork;
class DemuxingBackend;
class DemuxingComputation final : public NetworkComputation {
  std::tuple<const std::unique_ptr<NetworkComputation>&, int> GetParent(
      int sample) const {
    auto iter = std::lower_bound(parents_.begin(), parents_.end(), sample + 1);
    assert(iter != parents_.end());
    assert(sample >= iter->start_);
    assert(sample < iter->end_);
    return {iter->computation_, sample - iter->start_};
  }

 public:
  DemuxingComputation(DemuxingNetwork* network) : network_(network) {}
  ~DemuxingComputation() {
    // Wait for other threads to stop using this thread. It must be spinloop for
    // correct synchronization between notify_one and destructor.
    while (dataready_.load(std::memory_order_acquire) != -1) {
      SpinloopPause();
    }
  }

  void AddInput(InputPlanes&& input) override {
    planes_.emplace_back(std::move(input));
  }

  void ComputeBlocking() override;

  int GetBatchSize() const override { return planes_.size(); }

  float GetQVal(int sample) const override {
    auto [parent, offset] = GetParent(sample);
    if (!parent) return 0;
    return parent->GetQVal(offset);
  }

  float GetDVal(int sample) const override {
    auto [parent, offset] = GetParent(sample);
    if (!parent) return 0;
    return parent->GetDVal(offset);
  }

  float GetMVal(int sample) const override {
    auto [parent, offset] = GetParent(sample);
    if (!parent) return 0;
    return parent->GetMVal(offset);
  }

  float GetPVal(int sample, int move_id) const override {
    auto [parent, offset] = GetParent(sample);
    if (!parent) return 0;
    return parent->GetPVal(offset, move_id);
  }

  void NotifyComplete() {
    if (1 == dataready_.fetch_sub(1, std::memory_order_release)) {
      {
        std::lock_guard lock(mutex_);
      }
      dataready_cv_.notify_one();
      dataready_.store(-1, std::memory_order_release);
    }
  }

 private:
  std::vector<InputPlanes> planes_;
  DemuxingNetwork* network_;
  std::vector<DemuxingWork> parents_;

  std::mutex mutex_;
  std::condition_variable dataready_cv_;
  std::atomic<int> dataready_ = -1;

  friend class DemuxingBackend;
};

class DemuxingBackend {
 public:
  ~DemuxingBackend() {
    while (!threads_.empty()) {
      threads_.back().join();
      threads_.pop_back();
    }
    while (!queue_.empty()) {
      queue_.front()->source_->NotifyComplete();
      queue_.pop();
    }
  }

  void Assign(std::unique_ptr<Network>&& network, const OptionsDict& opts,
              std::atomic<bool>& abort) {
    network_ = std::move(network);
    int nn_threads = opts.GetOrDefault<int>("threads", 0);
    if (nn_threads == 0) {
      nn_threads = network_->GetThreads();
    }
    for (int i = 0; i < nn_threads; i++) {
      threads_.emplace_back([&] { Worker(abort); });
    }
  }

  void Enqueue(DemuxingWork* work) {
    {
      std::unique_lock lock(mutex_);
      queue_.push(work);
    }
    dataready_cv_.notify_one();
  }

  void Abort() {
    {
      std::unique_lock lock(mutex_);
    }
    dataready_cv_.notify_all();
  }

  void Worker(std::atomic<bool>& abort) {
    while (!abort.load(std::memory_order_relaxed)) {
      DemuxingWork* work = nullptr;
      {
        std::unique_lock lock(mutex_);
        dataready_cv_.wait(lock, [&] {
          return abort.load(std::memory_order_relaxed) || !queue_.empty();
        });
        if (abort.load(std::memory_order_relaxed)) return;
        if (!queue_.empty()) {
          work = queue_.front();
          queue_.pop();
        }
      }
      if (work) {
        work->computation_ = network_->NewComputation();
        auto& planes = work->source_->planes_;
        for (int i = work->start_; i < work->end_; i++) {
          work->computation_->AddInput(std::move(planes[i]));
        }
        work->computation_->ComputeBlocking();
        work->source_->NotifyComplete();
      }
    }
  }

 private:
  std::mutex mutex_;
  std::condition_variable dataready_cv_;
  std::vector<std::thread> threads_;
  std::unique_ptr<Network> network_;
  std::queue<DemuxingWork*> queue_;
};

class DemuxingNetwork final : public Network {
 public:
  DemuxingNetwork(const std::optional<WeightsFile>& weights,
                  const OptionsDict& options)
      : backends_(std::max(size_t(1), options.ListSubdicts().size())) {
    const auto parents = options.ListSubdicts();
    if (parents.empty()) {
      // If options are empty, or multiplexer configured in root object,
      // initialize on root object and default backend.
      auto backends = NetworkFactory::Get()->GetBackendsList();
      AddBackend(0, backends[0], weights, options);
    }

    int i = 0;
    for (const auto& name : parents) {
      AddBackend(i++, name, weights, options.GetSubdict(name));
    }
  }

  void AddBackend(int index, const std::string& name,
                  const std::optional<WeightsFile>& weights,
                  const OptionsDict& opts) {
    const std::string backend = opts.GetOrDefault<std::string>("backend", name);

    auto network = NetworkFactory::Get()->Create(backend, weights, opts);

    min_batch_size_ = std::min(min_batch_size_, network->GetMiniBatchSize());
    batch_step_ = std::max(batch_step_, network->GetPreferredBatchStep());
    is_cpu_ &= network->IsCpu();
    if (index == 0) {
      capabilities_ = network->GetCapabilities();
    } else {
      capabilities_.Merge(network->GetCapabilities());
    }
    backends_[index].Assign(std::move(network), opts, abort_);
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<DemuxingComputation>(this);
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  int GetMiniBatchSize() const override {
    return min_batch_size_ * backends_.size();
  }

  int GetPreferredBatchStep() const override { return batch_step_; }

  bool IsCpu() const override { return is_cpu_; }

  ~DemuxingNetwork() { Abort(); }

  void Abort() {
    abort_.store(true, std::memory_order_relaxed);
    for (auto& b : backends_) {
      b.Abort();
    }
  }

  std::vector<DemuxingBackend> backends_;
  NetworkCapabilities capabilities_;
  int min_batch_size_ = std::numeric_limits<int>::max();
  int batch_step_ = 1;
  bool is_cpu_ = true;
  std::atomic<int64_t> start_index_;
  std::atomic<bool> abort_ = false;
};

void DemuxingComputation::ComputeBlocking() {
  if (GetBatchSize() == 0) return;
  // Calculate batch_step_ size split count.
  int splits = 1 + (GetBatchSize() - 1) / network_->batch_step_;
  // Calculate the minimum number of splits per backend.
  int split_size_per_backend = splits / network_->backends_.size();
  // Calculate how many backends get extra work.
  int extra_split_backends =
      splits - split_size_per_backend * network_->backends_.size();

  // Find the first backend which got less work from the previous batch.
  int start_index =
      network_->start_index_.fetch_add(std::max(1, extra_split_backends),
                                       std::memory_order_relaxed) %
      network_->backends_.size();

  int end_index =
      (start_index + extra_split_backends) % network_->backends_.size();
  int work_start = 0;
  int work_items = split_size_per_backend > 0 ? network_->backends_.size()
                                             : extra_split_backends;
  // First store the work item count and reserve memory from them.
  dataready_.store(work_items, std::memory_order_relaxed);
  parents_.reserve(work_items);
  int i = start_index;
  // First send work to backends which get extra work.
  int split_size = split_size_per_backend + 1;
  for (; i != end_index; i = (i + 1) % network_->backends_.size()) {
    assert(work_start != GetBatchSize());
    int work_end = work_start + split_size * network_->batch_step_;
    work_end = std::min(work_end, GetBatchSize());
    parents_.emplace_back(this, work_start, work_end);
    network_->backends_[i].Enqueue(&parents_.back());
    work_start = work_end;
  }
  // Queue remaining work items which don't get extra work.
  split_size--;
  if (split_size > 0) {
    do {
      assert(work_start != GetBatchSize());
      int work_end = work_start + split_size * network_->batch_step_;
      work_end = std::min(work_end, GetBatchSize());
      parents_.emplace_back(this, work_start, work_end);
      network_->backends_[i].Enqueue(&parents_.back());
      work_start = work_end;
      i = (i + 1) % network_->backends_.size();
    } while (i != start_index);
  }
  assert(work_start == GetBatchSize());
  assert(work_items == (int)parents_.size());
  // Wait until all backends complete their work.
  std::unique_lock<std::mutex> lock(mutex_);
  dataready_cv_.wait(lock, [this]() {
    return dataready_.load(std::memory_order_acquire) <= 0;
  });
}

std::unique_ptr<Network> MakeDemuxingNetwork(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  return std::make_unique<DemuxingNetwork>(weights, options);
}

REGISTER_NETWORK("demux", MakeDemuxingNetwork, -1001)

}  // namespace
}  // namespace lczero

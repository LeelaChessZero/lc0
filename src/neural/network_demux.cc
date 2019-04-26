/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#include "neural/factory.h"

#include <condition_variable>
#include <queue>
#include <thread>
#include "utils/exception.h"

namespace lczero {
namespace {

class DemuxingNetwork;
class DemuxingComputation : public NetworkComputation {
 public:
  DemuxingComputation(DemuxingNetwork* network) : network_(network) {}

  void AddInput(InputPlanes&& input) override { planes_.emplace_back(input); }

  void ComputeBlocking() override;

  int GetBatchSize() const override { return planes_.size(); }

  float GetQVal(int sample) const override {
    const int idx = sample / partial_size_;
    const int offset = sample % partial_size_;
    return parents_[idx]->GetQVal(offset);
  }

  float GetDVal(int sample) const override {
    int idx = sample / partial_size_;
    int offset = sample % partial_size_;
    return parents_[idx]->GetDVal(offset);
  }

  float GetPVal(int sample, int move_id) const override {
    const int idx = sample / partial_size_;
    const int offset = sample % partial_size_;
    return parents_[idx]->GetPVal(offset, move_id);
  }

  void NotifyComplete() {
    std::unique_lock<std::mutex> lock(mutex_);
    dataready_--;
    if (dataready_ == 0) {
      dataready_cv_.notify_one();
    }
  }

  NetworkComputation* AddParentFromNetwork(Network* network) {
    std::unique_lock<std::mutex> lock(mutex_);
    parents_.emplace_back(network->NewComputation());
    const int cur_idx = (parents_.size() - 1) * partial_size_;
    for (int i = cur_idx; i < std::min(GetBatchSize(), cur_idx + partial_size_);
         i++) {
      parents_.back()->AddInput(std::move(planes_[i]));
    }
    return parents_.back().get();
  }

 private:
  std::vector<InputPlanes> planes_;
  DemuxingNetwork* network_;
  std::vector<std::unique_ptr<NetworkComputation>> parents_;

  std::mutex mutex_;
  std::condition_variable dataready_cv_;
  int dataready_ = 0;
  int partial_size_ = 0;
};

class DemuxingNetwork : public Network {
 public:
  DemuxingNetwork(const WeightsFile& weights, const OptionsDict& options) {
    minimum_split_size_ = options.GetOrDefault<int>("minimum-split-size", 0);
    const auto parents = options.ListSubdicts();
    if (parents.empty()) {
      // If options are empty, or multiplexer configured in root object,
      // initialize on root object and default backend.
      auto backends = NetworkFactory::Get()->GetBackendsList();
      AddBackend(backends[0], weights, options);
    }

    for (const auto& name : parents) {
      AddBackend(name, weights, options.GetSubdict(name));
    }
  }

  void AddBackend(const std::string& name, const WeightsFile& weights,
                  const OptionsDict& opts) {
    const int nn_threads = opts.GetOrDefault<int>("threads", 1);
    const std::string backend = opts.GetOrDefault<std::string>("backend", name);

    networks_.emplace_back(
        NetworkFactory::Get()->Create(backend, weights, opts));

    for (int i = 0; i < nn_threads; ++i) {
      threads_.emplace_back([this]() { Worker(); });
    }
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<DemuxingComputation>(this);
  }

  void Enqueue(DemuxingComputation* computation) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(computation);
    cv_.notify_one();
  }

  ~DemuxingNetwork() {
    Abort();
    Wait();
    // Unstuck waiting computations.
    while (!queue_.empty()) {
      queue_.front()->NotifyComplete();
      queue_.pop();
    }
  }

  void Worker() {
    // While Abort() is not called (and it can only be called from destructor).
    while (!abort_) {
      {
        {
          std::unique_lock<std::mutex> lock(mutex_);
          // Wait until there's come work to compute.
          cv_.wait(lock, [&] { return abort_ || !queue_.empty(); });
          if (abort_) break;
        }

        // While there is a work in queue, process it.
        while (true) {
          DemuxingComputation* to_notify;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            if (queue_.empty()) break;
            to_notify = queue_.front();
            queue_.pop();
          }
          long long net_idx = ++(counter_) % networks_.size();
          NetworkComputation* to_compute =
              to_notify->AddParentFromNetwork(networks_[net_idx].get());
          to_compute->ComputeBlocking();
          to_notify->NotifyComplete();
        }
      }
    }
  }

  void Abort() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      abort_ = true;
    }
    cv_.notify_all();
  }

  void Wait() {
    while (!threads_.empty()) {
      threads_.back().join();
      threads_.pop_back();
    }
  }

  std::vector<std::unique_ptr<Network>> networks_;
  std::queue<DemuxingComputation*> queue_;
  int minimum_split_size_ = 0;
  std::atomic<long long> counter_;
  bool abort_ = false;

  std::mutex mutex_;
  std::condition_variable cv_;

  std::vector<std::thread> threads_;
};

void DemuxingComputation::ComputeBlocking() {
  if (GetBatchSize() == 0) return;
  partial_size_ = (GetBatchSize() + network_->networks_.size() - 1) /
                  network_->networks_.size();
  if (partial_size_ < network_->minimum_split_size_) {
    partial_size_ = std::min(GetBatchSize(), network_->minimum_split_size_);
  }
  const int splits = (GetBatchSize() + partial_size_ - 1) / partial_size_;

  std::unique_lock<std::mutex> lock(mutex_);
  dataready_ = splits;
  for (int j = 0; j < splits; j++) {
    network_->Enqueue(this);
  }
  dataready_cv_.wait(lock, [this]() { return dataready_ == 0; });
}

std::unique_ptr<Network> MakeDemuxingNetwork(const WeightsFile& weights,
                                             const OptionsDict& options) {
  return std::make_unique<DemuxingNetwork>(weights, options);
}

REGISTER_NETWORK("demux", MakeDemuxingNetwork, -1001)

}  // namespace
}  // namespace lczero

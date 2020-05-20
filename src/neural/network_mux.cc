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

#include <condition_variable>
#include <queue>
#include <thread>

#include "neural/factory.h"
#include "utils/exception.h"

namespace lczero {
namespace {

class MuxingNetwork;
class MuxingComputation : public NetworkComputation {
 public:
  MuxingComputation(MuxingNetwork* network) : network_(network) {}

  void AddInput(InputPlanes&& input) override { planes_.emplace_back(input); }

  void ComputeBlocking() override;

  int GetBatchSize() const override { return planes_.size(); }

  float GetQVal(int sample) const override {
    return parent_->GetQVal(sample + idx_in_parent_);
  }

  float GetDVal(int sample) const override {
    return parent_->GetDVal(sample + idx_in_parent_);
  }

  float GetMVal(int sample) const override {
    return parent_->GetMVal(sample + idx_in_parent_);
  }

  float GetPVal(int sample, int move_id) const override {
    return parent_->GetPVal(sample + idx_in_parent_, move_id);
  }

  void PopulateToParent(std::shared_ptr<NetworkComputation> parent) {
    // Populate our batch into batch of batches.
    parent_ = parent;
    idx_in_parent_ = parent->GetBatchSize();
    for (auto& x : planes_) parent_->AddInput(std::move(x));
  }

  void NotifyReady() {
    std::unique_lock<std::mutex> lock(mutex_);
    dataready_ = true;
    dataready_cv_.notify_one();
  }

 private:
  std::vector<InputPlanes> planes_;
  MuxingNetwork* network_;
  std::shared_ptr<NetworkComputation> parent_;
  int idx_in_parent_ = 0;

  std::mutex mutex_;
  std::condition_variable dataready_cv_;
  bool dataready_ = false;
};

class MuxingNetwork : public Network {
 public:
  MuxingNetwork(const std::optional<WeightsFile>& weights,
                const OptionsDict& options) {
    // int threads, int max_batch)
    //: network_(std::move(network)), max_batch_(max_batch) {

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

  void AddBackend(const std::string& name,
                  const std::optional<WeightsFile>& weights,
                  const OptionsDict& opts) {
    const int nn_threads = opts.GetOrDefault<int>("threads", 1);
    const int max_batch = opts.GetOrDefault<int>("max_batch", 256);
    const std::string backend = opts.GetOrDefault<std::string>("backend", name);

    networks_.emplace_back(
        NetworkFactory::Get()->Create(backend, weights, opts));
    Network* net = networks_.back().get();

    if (networks_.size() == 1) {
      capabilities_ = net->GetCapabilities();
    } else {
      capabilities_.Merge(net->GetCapabilities());
    }

    for (int i = 0; i < nn_threads; ++i) {
      threads_.emplace_back(
          [this, net, max_batch]() { Worker(net, max_batch); });
    }
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<MuxingComputation>(this);
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  void Enqueue(MuxingComputation* computation) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(computation);
    cv_.notify_one();
  }

  ~MuxingNetwork() {
    Abort();
    Wait();
    // Unstuck waiting computations.
    while (!queue_.empty()) {
      queue_.front()->NotifyReady();
      queue_.pop();
    }
  }

  void Worker(Network* network, const int max_batch) {
    // While Abort() is not called (and it can only be called from destructor).
    while (!abort_) {
      std::vector<MuxingComputation*> children;
      // Create new computation in "upstream" network, to gather batch into
      // there.
      std::shared_ptr<NetworkComputation> parent(network->NewComputation());
      {
        std::unique_lock<std::mutex> lock(mutex_);
        // Wait until there's come work to compute.
        cv_.wait(lock, [&] { return abort_ || !queue_.empty(); });
        if (abort_) break;

        // While there is a work in queue, add it.
        while (!queue_.empty()) {
          // If we are reaching batch size limit, stop adding.
          // However, if a single input batch is larger than output batch limit,
          // we still have to add it.
          if (parent->GetBatchSize() != 0 &&
              parent->GetBatchSize() + queue_.front()->GetBatchSize() >
                  max_batch) {
            break;
          }
          // Remember which of "input" computations we serve.
          children.push_back(queue_.front());
          queue_.pop();
          // Make "input" computation populate data into output batch.
          children.back()->PopulateToParent(parent);
        }
      }

      // Compute.
      parent->ComputeBlocking();
      // Notify children that data is ready!
      for (auto child : children) child->NotifyReady();
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

 private:
  std::vector<std::unique_ptr<Network>> networks_;
  std::queue<MuxingComputation*> queue_;
  bool abort_ = false;
  NetworkCapabilities capabilities_;

  std::mutex mutex_;
  std::condition_variable cv_;

  std::vector<std::thread> threads_;
};

void MuxingComputation::ComputeBlocking() {
  network_->Enqueue(this);
  std::unique_lock<std::mutex> lock(mutex_);
  dataready_cv_.wait(lock, [this]() { return dataready_; });
}

std::unique_ptr<Network> MakeMuxingNetwork(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  return std::make_unique<MuxingNetwork>(weights, options);
}

REGISTER_NETWORK("multiplexing", MakeMuxingNetwork, -1000)

}  // namespace
}  // namespace lczero

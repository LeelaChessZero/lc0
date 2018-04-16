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
*/

#include "neural/network_mux.h"
#include <condition_variable>
#include <queue>
#include <thread>

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

  float GetPVal(int sample, int move_id) const override {
    return parent_->GetPVal(sample + idx_in_parent_, move_id);
  }

  void PopulateToParent(std::shared_ptr<NetworkComputation> parent) {
    parent_ = parent;
    idx_in_parent_ = parent->GetBatchSize();
    for (auto& x : planes_) parent_->AddInput(std::move(x));
  }

  void NotifyReady() { dataready_cv_.notify_one(); }

 private:
  std::vector<InputPlanes> planes_;
  MuxingNetwork* network_;
  std::shared_ptr<NetworkComputation> parent_;
  int idx_in_parent_ = 0;

  std::condition_variable dataready_cv_;
};

class MuxingNetwork : public Network {
 public:
  MuxingNetwork(std::unique_ptr<Network> network, int threads, int max_batch)
      : network_(std::move(network)), max_batch_(max_batch) {
    while (threads_.size() < threads) {
      threads_.emplace_back([&]() { Worker(); });
    }
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<MuxingComputation>(this);
  }

  void Enqueue(MuxingComputation* computation) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(computation);
    cv_.notify_one();
  }

  ~MuxingNetwork() {
    Abort();
    Wait();
    // Unstuck waining computations.
    while (!queue_.empty()) {
      queue_.front()->NotifyReady();
      queue_.pop();
    }
  }

  void Worker() {
    while (!abort_) {
      std::vector<MuxingComputation*> children;
      std::shared_ptr<NetworkComputation> parent(network_->NewComputation());
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return abort_ || !queue_.empty(); });
        if (abort_) break;

        while (!queue_.empty()) {
          if (parent->GetBatchSize() != 0 &&
              parent->GetBatchSize() + queue_.front()->GetBatchSize() >
                  max_batch_) {
            break;
          }
          children.push_back(queue_.front());
          queue_.pop();
          children.back()->PopulateToParent(parent);
        }
      }

      parent->ComputeBlocking();
      for (auto child : children) child->NotifyReady();
    }
  }

  void Abort() {
    std::lock_guard<std::mutex> lock(mutex_);
    abort_ = true;
    cv_.notify_all();
  }

  void Wait() {
    while (!threads_.empty()) {
      threads_.back().join();
      threads_.pop_back();
    }
  }

 private:
  std::unique_ptr<Network> network_;
  std::queue<MuxingComputation*> queue_;
  bool abort_ = false;
  int max_batch_;

  std::mutex mutex_;
  std::condition_variable cv_;

  std::vector<std::thread> threads_;
};

void MuxingComputation::ComputeBlocking() {
  std::mutex mx;
  std::unique_lock<std::mutex> lock(mx);
  network_->Enqueue(this);
  dataready_cv_.wait(lock);
}

}  // namespace

std::unique_ptr<Network> MakeMuxingNetwork(std::unique_ptr<Network> parent,
                                           int threads, int max_batch) {
  return std::make_unique<MuxingNetwork>(std::move(parent), threads, max_batch);
}

}  // namespace lczero
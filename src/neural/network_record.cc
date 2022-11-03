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

#include <iostream>

#include "neural/factory.h"
#include "utils/hashcat.h"

namespace lczero {
namespace {

class RecordComputation : public NetworkComputation {
 public:
  RecordComputation(std::unique_ptr<NetworkComputation>&& inner,
                    const std::string& record_file)
      : inner_(std::move(inner)), record_file_(record_file) {}
  static uint64_t make_hash(const InputPlanes& input) {
    std::uint64_t hash = 0x2134435D4534LL;
    for (const auto& plane : input) {
      hash = HashCat({hash, plane.mask});
      std::uint32_t tmp;
      std::memcpy(&tmp, &plane.value, sizeof(float));
      const std::uint64_t value_hash = tmp;
      hash = HashCat({hash, value_hash});
    }
    return hash;
  }
  // Adds a sample to the batch.
  void AddInput(InputPlanes&& input) override {
    hashes_.push_back(make_hash(input));
    requests_.emplace_back();
    q_count_.push_back(0);
    inner_->AddInput(std::move(input));
  }
  // Do the computation.
  void ComputeBlocking() override { inner_->ComputeBlocking(); }
  // Returns how many times AddInput() was called.
  int GetBatchSize() const override { return inner_->GetBatchSize(); }
  float Capture(float value, int index) const {
    // Only capture until we see Q again - the rest can be infered from that
    // set.
    if (q_count_[index] > 1) return value;
    requests_[index].push_back(value);
    return value;
  }
  // Returns Q value of @sample.
  float GetQVal(int sample) const override {
    q_count_[sample]++;
    return Capture(inner_->GetQVal(sample), sample);
  }
  float GetDVal(int sample) const override {
    return Capture(inner_->GetDVal(sample), sample);
  }
  // Returns P value @move_id of @sample.
  float GetPVal(int sample, int move_id) const override {
    return Capture(inner_->GetPVal(sample, move_id), sample);
  }
  float GetMVal(int sample) const override {
    return Capture(inner_->GetMVal(sample), sample);
  }
  virtual ~RecordComputation() {
    Mutex::Lock lock(mutex_);
    std::fstream output(record_file_, std::ios::app | std::ios_base::binary);
    for (size_t i = 0; i < hashes_.size(); i++) {
      uint64_t value = hashes_[i];
      output.write(reinterpret_cast<const char*>(&value), sizeof(value));
      int32_t length = static_cast<int32_t>(requests_[i].size());
      output.write(reinterpret_cast<const char*>(&length), sizeof(length));
      for (int j = 0; j < length; j++) {
        float recorded = requests_[i][j];
        output.write(reinterpret_cast<const char*>(&recorded),
                     sizeof(recorded));
      }
    }
  }
  std::unique_ptr<NetworkComputation> inner_;
  std::string record_file_;
  std::vector<uint64_t> hashes_;
  mutable std::vector<int> q_count_;
  mutable std::vector<std::vector<float>> requests_;
  static Mutex mutex_;
};

Mutex RecordComputation::mutex_;

class ReplayComputation : public NetworkComputation {
 public:
  ReplayComputation(std::unordered_map<uint64_t, std::vector<float>>* lookup)
      : lookup_(lookup) {}
  // Adds a sample to the batch.
  void AddInput(InputPlanes&& input) override {
    hashes_.push_back(RecordComputation::make_hash(input));
    replay_counter_.push_back(0);
  }
  // Do the computation.
  void ComputeBlocking() override {}
  // Returns how many times AddInput() was called.
  int GetBatchSize() const override { return static_cast<int>(hashes_.size()); }
  float Replay(int index) const {
    const auto& entry_ptr = lookup_->find(hashes_[index]);
    if (entry_ptr == lookup_->end()) {
      return 0.0f;
    }
    const auto& entry = entry_ptr->second;
    size_t counter = replay_counter_[index];
    if (counter >= entry.size()) {
      // Second pass reads the same things in the same order as first.
      counter = counter - entry.size();
      if (counter >= entry.size()) {
        // Third pass skips the first 3, then reads the rest in the same order.
        counter = counter - entry.size() + 3;
        if (counter >= entry.size()) {
          return 0.0f;
        }
      }
    }
    replay_counter_[index]++;
    return entry[counter];
  }
  // Returns Q value of @sample.
  float GetQVal(int sample) const override { return Replay(sample); }
  float GetDVal(int sample) const override { return Replay(sample); }
  // Returns P value @move_id of @sample.
  float GetPVal(int sample, int) const override { return Replay(sample); }
  float GetMVal(int sample) const override { return Replay(sample); }
  virtual ~ReplayComputation() {}

  std::unique_ptr<NetworkComputation> inner_;
  std::vector<uint64_t> hashes_;
  mutable std::vector<size_t> replay_counter_;
  std::unordered_map<uint64_t, std::vector<float>>* lookup_;
};

class RecordReplayNetwork : public Network {
 public:
  RecordReplayNetwork(const std::optional<WeightsFile>& weights,
                      const OptionsDict& options) {
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
    replay_file_ = options.GetOrDefault<std::string>("replay_file", "");
    record_file_ = options.GetOrDefault<std::string>("record_file", "");
    if (replay_file_.size() > 0) {
      lookup_ =
          std::make_unique<std::unordered_map<uint64_t, std::vector<float>>>();
      std::ifstream input(replay_file_, std::ios_base::binary);
      input.seekg(0, input.end);
      auto file_length = input.tellg();
      input.seekg(0, input.beg);
      while (input.tellg() < file_length) {
        uint64_t value = 0;
        input.read(reinterpret_cast<char*>(&value), sizeof(value));
        int32_t length = 0;
        input.read(reinterpret_cast<char*>(&length), sizeof(length));
        auto& entry = (*lookup_)[value];
        // Only use the first recorded value for any hash collisions.
        bool fill = entry.size() == 0;
        for (int j = 0; j < length; j++) {
          float recorded = 0.0f;
          input.read(reinterpret_cast<char*>(&recorded), sizeof(recorded));
          if (fill) {
            entry.push_back(recorded);
          }
        }
      }
    }
  }

  void AddBackend(const std::string& name,
                  const std::optional<WeightsFile>& weights,
                  const OptionsDict& opts) {
    const std::string backend = opts.GetOrDefault<std::string>("backend", name);

    networks_.emplace_back(
        NetworkFactory::Get()->Create(backend, weights, opts));

    if (networks_.size() == 1) {
      capabilities_ = networks_.back()->GetCapabilities();
    } else {
      capabilities_.Merge(networks_.back()->GetCapabilities());
    }
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    if (!lookup_) {
      const long long val = ++counter_;
      return std::make_unique<RecordComputation>(
          networks_[val % networks_.size()]->NewComputation(), record_file_);
    }
    return std::make_unique<ReplayComputation>(lookup_.get());
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  ~RecordReplayNetwork() {}

 private:
  std::vector<std::unique_ptr<Network>> networks_;
  std::atomic<long long> counter_;
  NetworkCapabilities capabilities_;
  std::string replay_file_;
  std::string record_file_;
  std::unique_ptr<std::unordered_map<uint64_t, std::vector<float>>> lookup_;
};

std::unique_ptr<Network> MakeRecordReplayNetwork(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  return std::make_unique<RecordReplayNetwork>(weights, options);
}

REGISTER_NETWORK("recordreplay", MakeRecordReplayNetwork, -999)

}  // namespace
}  // namespace lczero

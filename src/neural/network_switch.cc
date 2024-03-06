/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2024 The LCZero Authors

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

#include <thread>

#include "chess/bitboard.h"
#include "neural/factory.h"
#include "neural/network.h"
#include "utils/logging.h"

namespace lczero {

namespace {

class SwitchNetwork;

class SwitchComputation : public NetworkComputation {
 public:
  SwitchComputation(std::unique_ptr<NetworkComputation> main_comp,
                    std::unique_ptr<NetworkComputation> endgame_comp,
                    int threads)
      : main_comp_(std::move(main_comp)),
        endgame_comp_(std::move(endgame_comp)),
        threads_(threads) {}

  void AddInput(InputPlanes&& input) override {
    auto pieces = input[1].mask | input[2].mask | input[3].mask |
                  input[4].mask | input[7].mask | input[8].mask |
                  input[9].mask | input[10].mask;
    if (BitBoard(pieces).count() > 6) {
      is_endgame_.push_back(false);
      rev_idx_.push_back(main_idx_.size());
      main_idx_.push_back(main_idx_.size());
      main_comp_->AddInput(std::move(input));
    } else {
      is_endgame_.push_back(true);
      rev_idx_.push_back(endgame_idx_.size());
      endgame_idx_.push_back(endgame_idx_.size());
      endgame_comp_->AddInput(std::move(input));
    }
  }

  void ComputeBlocking() override {
    if (threads_ > 1 && main_idx_.size() > 0 && endgame_idx_.size() > 0) {
      std::thread main(
          [](NetworkComputation* comp) { comp->ComputeBlocking(); },
          main_comp_.get());
      endgame_comp_->ComputeBlocking();
      main.join();
    } else {
      if (main_idx_.size() > 0) main_comp_->ComputeBlocking();
      if (endgame_idx_.size() > 0) endgame_comp_->ComputeBlocking();
    }
  }

  int GetBatchSize() const override {
    return static_cast<int>(main_comp_->GetBatchSize() +
                            endgame_comp_->GetBatchSize());
  }

  float GetQVal(int sample) const override {
    if (is_endgame_[sample]) {
      return endgame_comp_->GetQVal(rev_idx_[sample]);
    }
    return main_comp_->GetQVal(rev_idx_[sample]);
  }

  float GetDVal(int sample) const override {
    if (is_endgame_[sample]) {
      return endgame_comp_->GetDVal(rev_idx_[sample]);
    }
    return main_comp_->GetDVal(rev_idx_[sample]);
  }

  float GetMVal(int sample) const override {
    if (is_endgame_[sample]) {
      return endgame_comp_->GetMVal(rev_idx_[sample]);
    }
    return main_comp_->GetMVal(rev_idx_[sample]);
  }

  float GetPVal(int sample, int move_id) const override {
    if (is_endgame_[sample]) {
      return endgame_comp_->GetPVal(rev_idx_[sample], move_id);
    }
    return main_comp_->GetPVal(rev_idx_[sample], move_id);
  }

 private:
  std::unique_ptr<NetworkComputation> main_comp_;
  std::unique_ptr<NetworkComputation> endgame_comp_;
  std::vector<size_t> main_idx_;
  std::vector<size_t> endgame_idx_;
  std::vector<size_t> rev_idx_;
  std::vector<bool> is_endgame_;
  int threads_;
};

class SwitchNetwork : public Network {
 public:
  SwitchNetwork(const std::optional<WeightsFile>& weights,
                const OptionsDict& options) {
    auto backends = NetworkFactory::Get()->GetBackendsList();

    auto endgame_net = options.Get<std::string>("endgame_weights");
    threads_ = options.GetOrDefault<int>("threads", 1);

    auto& main_options =
        options.HasSubdict("main") ? options.GetSubdict("main") : options;

    main_net_ = NetworkFactory::Get()->Create(
        main_options.GetOrDefault<std::string>("backend", backends[0]), weights,
        main_options);

    CERR << "Loading endgame weights file from: " << endgame_net;
    std::optional<WeightsFile> endgame_weights =
        LoadWeightsFromFile(endgame_net);

    auto& endgame_options =
        options.HasSubdict("endgame") ? options.GetSubdict("endgame") : options;

    endgame_net_ = NetworkFactory::Get()->Create(
        endgame_options.GetOrDefault<std::string>("backend", backends[0]),
        endgame_weights, endgame_options);

    capabilities_ = main_net_->GetCapabilities();
    capabilities_.Merge(endgame_net_->GetCapabilities());
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    std::unique_ptr<NetworkComputation> main_comp = main_net_->NewComputation();
    std::unique_ptr<NetworkComputation> endgame_comp =
        endgame_net_->NewComputation();
    return std::make_unique<SwitchComputation>(
        std::move(main_comp), std::move(endgame_comp), threads_);
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  int GetMiniBatchSize() const override {
    return std::min(main_net_->GetMiniBatchSize(),
                    endgame_net_->GetMiniBatchSize());
  }

  bool IsCpu() const override {
    return main_net_->GetMiniBatchSize() | endgame_net_->GetMiniBatchSize();
  }

  void InitThread(int id) override {
    main_net_->InitThread(id);
    endgame_net_->InitThread(id);
  }

 private:
  std::unique_ptr<Network> main_net_;
  std::unique_ptr<Network> endgame_net_;
  int threads_;
  NetworkCapabilities capabilities_;
};

std::unique_ptr<Network> MakeSwitchNetwork(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  return std::make_unique<SwitchNetwork>(weights, options);
}

REGISTER_NETWORK("switch", MakeSwitchNetwork, -800)

}  // namespace
}  // namespace lczero

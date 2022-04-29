/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

#pragma once

#include <string>

#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "utils/fastmath.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace python {

class Weights {
 public:
  using InputFormat = pblczero::NetworkFormat::InputFormat;
  using PolicyFormat = pblczero::NetworkFormat::PolicyFormat;
  using ValueFormat = pblczero::NetworkFormat::ValueFormat;
  using MovesLeftFormat = pblczero::NetworkFormat::MovesLeftFormat;

  // Exported methods.
  Weights(const std::optional<std::string>& filename)
      : filename_(filename ? *filename : DiscoverWeightsFile()),
        weights_(LoadWeightsFromFile(filename_)) {}

  std::string_view filename() const { return filename_; }
  std::string_view license() const { return weights_.license(); }
  std::string min_version() const {
    const auto& ver = weights_.min_version();
    return std::to_string(ver.major()) + '.' + std::to_string(ver.minor()) +
           '.' + std::to_string(ver.patch());
  }
  int input_format() const {
    return weights_.format().network_format().input();
  }
  int policy_format() const {
    return weights_.format().network_format().policy();
  }
  int value_format() const {
    return weights_.format().network_format().value();
  }
  int moves_left_format() const {
    return weights_.format().network_format().moves_left();
  }
  int blocks() const { return weights_.weights().residual_size(); }
  int filters() const {
    return weights_.weights().residual(0).conv1().weights().params().size() /
           2304;
  }

  // Not exported methods.

  const WeightsFile& weights() const { return weights_; }

 private:
  const std::string filename_;
  const WeightsFile weights_;
};

inline std::vector<std::string> GetAvailableBackends() {
  return NetworkFactory::Get()->GetBackendsList();
}

class Input {
 public:
  // Exported functions.
  Input() = default;
  void set_mask(int plane, uint64_t mask) {
    CheckPlaneExists(plane);
    data_[plane].mask = mask;
  }
  void set_val(int plane, float val) {
    CheckPlaneExists(plane);
    data_[plane].value = val;
  }
  uint64_t mask(int plane) const {
    CheckPlaneExists(plane);
    return data_[plane].mask;
  }
  float val(int plane) const {
    CheckPlaneExists(plane);
    return data_[plane].value;
  }
  std::unique_ptr<Input> clone() const {
    return std::make_unique<Input>(data_);
  }

  // Not exported.
  const InputPlanes GetPlanes() const { return data_; }
  Input(const InputPlanes& data) : data_(data) {}

 private:
  void CheckPlaneExists(int plane) const {
    if (plane < 0 || plane >= static_cast<int>(data_.size())) {
      throw Exception("Plane index must be between 0 and " +
                      std::to_string(data_.size()));
    }
  }

  InputPlanes data_{kInputPlanes};
};

class Output {
 public:
  // Not exposed.
  Output(const NetworkComputation& computation, int idx) {
    for (int i = 0; i < 1858; ++i) p_[i] = computation.GetPVal(idx, i);
    q_ = computation.GetQVal(idx);
    d_ = computation.GetDVal(idx);
    m_ = computation.GetMVal(idx);
  }
  float q() const { return q_; }
  float d() const { return d_; }
  float m() const { return m_; }
  std::vector<float> p_raw(const std::vector<int>& indicies) {
    std::vector<float> result(indicies.size());
    for (size_t i = 0; i < indicies.size(); ++i) {
      int idx = indicies[i];
      if (idx < 0 || idx > 1857) {
        throw Exception("Policy index must be between 0 and 1857.");
      }
      result[i] = p_[idx];
    }
    return result;
  }

  std::vector<float> p_softmax(const std::vector<int>& indicies) {
    auto p_vals = p_raw(indicies);
    float max_p = -std::numeric_limits<float>::infinity();
    for (auto x : p_vals) max_p = std::max(max_p, x);

    float total = 0.0;
    for (auto& x : p_vals) {
      x = FastExp(x - max_p);
      total += x;
    }
    // Normalize P values to add up to 1.0.
    if (total > 0.0f) {
      const float scale = 1.0f / total;
      for (auto& x : p_vals) x *= scale;
    }
    return p_vals;
  }

 private:
  float p_[1858];
  float q_;
  float d_;
  float m_;
};

class BackendCapabilities {
 public:
  // Exported.
  int input_format() const { return caps_.input_format; }
  int moves_left_format() const { return caps_.moves_left; }

  // Not exposed.
  BackendCapabilities(NetworkCapabilities caps) : caps_(caps) {}

 private:
  const NetworkCapabilities caps_;
};

class Backend {
 public:
  // Exported methods.
  static inline std::vector<std::string> available_backends() {
    return NetworkFactory::Get()->GetBackendsList();
  }

  Backend(const Weights* weights, const std::optional<std::string>& backend,
          const std::optional<std::string>& options) {
    std::optional<WeightsFile> w;
    if (weights) w = weights->weights();
    const auto& backends = GetAvailableBackends();
    const std::string be =
        backend.value_or(backends.empty() ? "<none>" : backends[0]);
    OptionsDict network_options;
    if (options) network_options.AddSubdictFromString(*options);
    network_ = NetworkFactory::Get()->Create(be, w, network_options);
  }

  BackendCapabilities capabilities() const {
    return BackendCapabilities(network_->GetCapabilities());
  }

  std::vector<std::unique_ptr<Output>> evaluate(
      const std::vector<Input*>& inputs) const {
    if (inputs.empty()) return {};
    auto computation = network_->NewComputation();
    for (const auto* input : inputs) {
      InputPlanes input_copy = input->GetPlanes();
      computation->AddInput(std::move(input_copy));
    }
    computation->ComputeBlocking();
    std::vector<std::unique_ptr<Output>> result;
    for (int i = 0; i < computation->GetBatchSize(); ++i) {
      result.push_back(std::make_unique<Output>(*computation, i));
    }
    return result;
  }

 private:
  std::unique_ptr<::lczero::Network> network_;
};

class GameState {
 public:
  GameState(const std::optional<std::string> startpos,
            const std::vector<std::string>& moves) {
    ChessBoard starting_board;
    int no_capture_ply;
    int full_moves;
    starting_board.SetFromFen(startpos.value_or(ChessBoard::kStartposFen),
                              &no_capture_ply, &full_moves);

    history_.Reset(starting_board, no_capture_ply,
                   full_moves * 2 - (starting_board.flipped() ? 1 : 2));

    for (const auto& m : moves) {
      Move move(m, history_.IsBlackToMove());
      move = history_.Last().GetBoard().GetModernMove(move);
      history_.Append(move);
    }
  }

  std::unique_ptr<Input> as_input(const Backend& backend) const {
    int tmp;
    return std::make_unique<Input>(
        EncodePositionForNN(static_cast<pblczero::NetworkFormat::InputFormat>(
                                backend.capabilities().input_format()),
                            history_, 8, FillEmptyHistory::FEN_ONLY, &tmp));
  }

  std::vector<std::string> moves() const {
    auto ms = history_.Last().GetBoard().GenerateLegalMoves();
    bool is_black = history_.IsBlackToMove();
    std::vector<std::string> result;
    for (auto m : ms) {
      if (is_black) m.Mirror();
      result.push_back(m.as_string());
    }
    return result;
  }

  std::vector<int> policy_indices() const {
    auto ms = history_.Last().GetBoard().GenerateLegalMoves();
    std::vector<int> result;
    for (auto m : ms) {
      result.push_back(m.as_nn_index(/* transform= */ 0));
    }
    return result;
  }

  std::string as_string() const {
    bool is_black = history_.IsBlackToMove();
    return (is_black ? history_.Last().GetThemBoard()
                     : history_.Last().GetBoard())
        .DebugString();
  }

 private:
  PositionHistory history_;
};

}  // namespace python
}  // namespace lczero

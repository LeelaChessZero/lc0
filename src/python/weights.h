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

#include "neural/factory.h"
#include "neural/loader.h"
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

 private:
  void CheckPlaneExists(int plane) const {
    if (plane < 0 || plane >= static_cast<int>(data_.size())) {
      throw Exception("Plane index must be between 0 and " +
                      std::to_string(data_.size()));
    }
  }

  InputPlanes data_{kInputPlanes};
};

class Backend {
 public:
  // Exported methods.

  static inline std::vector<std::string> available_backends() {
    return NetworkFactory::Get()->GetBackendsList();
  }

  Backend(const std::optional<std::string>& backend, const Weights* weights,
          const std::optional<std::string>& options) {
    std::optional<WeightsFile> w;
    if (weights) w = weights->weights();
    const auto& backends = GetAvailableBackends();
    const std::string be =
        backend.value_or(backends.empty() ? "<none>" : backends[0]);
    OptionsDict network_options;
    if (options) network_options.AddSubdictFromString(*options);
    NetworkFactory::Get()->Create(be, w, network_options);
  }

 private:
  std::unique_ptr<::lczero::Network> network_;
};

}  // namespace python
}  // namespace lczero
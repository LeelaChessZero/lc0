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
#include <cmath>
#include <iomanip>

#include "neural/decoder.h"
#include "neural/factory.h"
#include "neural/network.h"
#include "utils/histogram.h"
#include "utils/logging.h"
#include "utils/random.h"

namespace lczero {

namespace {

class CheckNetwork;

enum CheckMode {
  kCheckOnly,
  kErrorDisplay,
  kHistogram,
};

struct CheckParams {
  CheckMode mode;
  double absolute_tolerance;
  double relative_tolerance;
  pblczero::NetworkFormat::InputFormat input_format;
};

class CheckComputation : public NetworkComputation {
 public:
  CheckComputation(const CheckParams& params,
                   std::unique_ptr<NetworkComputation> work_comp,
                   std::unique_ptr<NetworkComputation> check_comp)
      : params_(params),
        work_comp_(std::move(work_comp)),
        check_comp_(std::move(check_comp)) {}

  void AddInput(InputPlanes&& input) override {
    InputPlanes x = input;
    InputPlanes y = input;
    work_comp_->AddInput(std::move(x));
    check_comp_->AddInput(std::move(y));

    ChessBoard board;
    int rule50;
    int gameply;
    PopulateBoard(params_.input_format, input, &board, &rule50, &gameply);
    moves_.emplace_back(board.GenerateLegalMoves());
  }

  void ComputeBlocking() override {
    work_comp_->ComputeBlocking();
    check_comp_->ComputeBlocking();
    switch (params_.mode) {
      case kCheckOnly:
        CheckOnly();
        break;
      case kErrorDisplay:
        DisplayError();
        break;
      case kHistogram:
        DisplayHistogram();
        break;
    }
  }

  int GetBatchSize() const override {
    return static_cast<int>(work_comp_->GetBatchSize());
  }

  float GetQVal(int sample) const override {
    return work_comp_->GetQVal(sample);
  }

  float GetDVal(int sample) const override {
    return work_comp_->GetDVal(sample);
  }

  float GetMVal(int sample) const override {
    return work_comp_->GetMVal(sample);
  }

  float GetPVal(int sample, int move_id) const override {
    return work_comp_->GetPVal(sample, move_id);
  }

 private:
  const CheckParams& params_;
  std::vector<MoveList> moves_;

  std::vector<float> PolicySoftMax(const NetworkComputation* comp, int sample,
                                   const std::vector<Move>& moves) const {
    float max_p = -std::numeric_limits<float>::infinity();
    std::vector<float> policy;
    policy.reserve(moves.size());
    for (const auto move : moves) {
      policy.emplace_back(comp->GetPVal(sample, move.as_nn_index(0)));
      max_p = std::max(max_p, policy.back());
    }
    float total = 0;
    for (auto& p : policy) {
      p = std::exp(p - max_p);
      total += p;
    }
    if (total > 0) {
      for (auto& p : policy) {
        p /= total;
      }
    }
    return policy;
  }

  void CheckOnly() const {
    bool valueAlmostEqual = true;
    const int size = GetBatchSize();
    for (int i = 0; i < size && valueAlmostEqual; i++) {
      const float v1 = work_comp_->GetQVal(i);
      const float v2 = check_comp_->GetQVal(i);
      valueAlmostEqual &= IsAlmostEqual(v1, v2);
    }

    bool policyAlmostEqual = true;
    for (int i = 0; i < size && policyAlmostEqual; i++) {
      const auto work = PolicySoftMax(work_comp_.get(), i, moves_[i]);
      const auto check = PolicySoftMax(check_comp_.get(), i, moves_[i]);
      for (size_t j = 0; j < work.size(); j++) {
        policyAlmostEqual &= IsAlmostEqual(work[j], check[j]);
      }
    }

    if (valueAlmostEqual && policyAlmostEqual) {
      CERR << "Check passed for a batch of " << size << ".";
      return;
    }

    if (!valueAlmostEqual && !policyAlmostEqual) {
      CERR << "*** ERROR check failed for a batch of " << size
           << " both value and policy incorrect.";
      return;
    }

    if (!valueAlmostEqual) {
      CERR << "*** ERROR check failed for a batch of " << size
           << " value incorrect (but policy ok).";
      return;
    }

    CERR << "*** ERROR check failed for a batch of " << size
         << " policy incorrect (but value ok).";
  }

  bool IsAlmostEqual(double a, double b) const {
    return std::abs(a - b) <= std::max(params_.relative_tolerance *
                                           std::max(std::abs(a), std::abs(b)),
                                       params_.absolute_tolerance);
  }

  void DisplayHistogram() {
    Histogram histogram(-15, 1, 5);

    const int size = GetBatchSize();
    for (int i = 0; i < size; i++) {
      const float qv1 = work_comp_->GetQVal(i);
      const float qv2 = check_comp_->GetQVal(i);
      histogram.Add(qv2 - qv1);
      const auto work = PolicySoftMax(work_comp_.get(), i, moves_[i]);
      const auto check = PolicySoftMax(check_comp_.get(), i, moves_[i]);
      for (size_t j = 0; j < work.size(); j++) {
        histogram.Add(check[j] - work[j]);
      }
    }
    CERR << "Absolute error histogram for a batch of " << size;
    histogram.Dump();
  }

  // Compute maximum absolute/relative errors.
  struct MaximumError {
    double max_absolute_error = 0;
    double max_relative_error = 0;

    void Add(double a, double b) {
      const double absolute_error = GetAbsoluteError(a, b);
      if (absolute_error > max_absolute_error) {
        max_absolute_error = absolute_error;
      }
      const double relative_error = GetRelativeError(a, b);
      if (relative_error > max_relative_error) {
        max_relative_error = relative_error;
      }
    }

    void Dump(const char* name) {
      CERR << std::scientific << std::setprecision(1) << name
           << ": absolute: " << max_absolute_error
           << ", relative: " << max_relative_error << ".";
    }

    static double GetRelativeError(double a, double b) {
      const double max = std::max(std::abs(a), std::abs(b));
      return max == 0 ? 0 : std::abs(a - b) / max;
    }

    static double GetAbsoluteError(double a, double b) {
      return std::abs(a - b);
    }
  };

  void DisplayError() {
    MaximumError value_error;
    const int size = GetBatchSize();
    for (int i = 0; i < size; i++) {
      const float v1 = work_comp_->GetQVal(i);
      const float v2 = check_comp_->GetQVal(i);
      value_error.Add(v1, v2);
    }

    MaximumError policy_error;
    for (int i = 0; i < size; i++) {
      const auto work = PolicySoftMax(work_comp_.get(), i, moves_[i]);
      const auto check = PolicySoftMax(check_comp_.get(), i, moves_[i]);
      for (size_t j = 0; j < work.size(); j++) {
        policy_error.Add(work[j], check[j]);
      }
    }

    CERR << "maximum error for a batch of " << size << ":";

    value_error.Dump("  value");
    policy_error.Dump("  policy");
  }

  std::unique_ptr<NetworkComputation> work_comp_;
  std::unique_ptr<NetworkComputation> check_comp_;
};

class CheckNetwork : public Network {
 public:
  static constexpr CheckMode kDefaultMode = kCheckOnly;
  static constexpr double kDefaultCheckFrequency = 0.2;
  static constexpr double kDefaultAbsoluteTolerance = 1e-5;
  static constexpr double kDefaultRelativeTolerance = 1e-4;

  CheckNetwork(const std::optional<WeightsFile>& weights,
               const OptionsDict& options) {
    params_.mode = kDefaultMode;
    params_.absolute_tolerance = kDefaultAbsoluteTolerance;
    params_.relative_tolerance = kDefaultRelativeTolerance;
    check_frequency_ = kDefaultCheckFrequency;

    OptionsDict dict1;
    std::string backendName1 = "opencl";
    OptionsDict& backend1_dict = dict1;

    OptionsDict dict2;
    std::string backendName2 = "eigen";
    OptionsDict& backend2_dict = dict2;

    const std::string mode = options.GetOrDefault<std::string>("mode", "check");
    if (mode == "check") {
      params_.mode = kCheckOnly;
    } else if (mode == "histo") {
      params_.mode = kHistogram;
    } else if (mode == "display") {
      params_.mode = kErrorDisplay;
    }

    params_.absolute_tolerance =
        options.GetOrDefault<float>("atol", kDefaultAbsoluteTolerance);
    params_.relative_tolerance =
        options.GetOrDefault<float>("rtol", kDefaultRelativeTolerance);

    const auto parents = options.ListSubdicts();
    if (parents.size() > 0) {
      backendName1 = parents[0];
      backend1_dict = options.GetSubdict(backendName1);
      backendName1 =
          backend1_dict.GetOrDefault<std::string>("backend", backendName1);
    }
    if (parents.size() > 1) {
      backendName2 = parents[1];
      backend2_dict = options.GetSubdict(backendName2);
      backendName2 =
          backend2_dict.GetOrDefault<std::string>("backend", backendName2);
    }
    if (parents.size() > 2) {
      CERR << "Warning, cannot check more than two backends";
    }

    CERR << "Working backend set to " << backendName1 << ".";
    CERR << "Reference backend set to " << backendName2 << ".";

    work_net_ =
        NetworkFactory::Get()->Create(backendName1, weights, backend1_dict);
    check_net_ =
        NetworkFactory::Get()->Create(backendName2, weights, backend2_dict);

    capabilities_ = work_net_->GetCapabilities();
    capabilities_.Merge(check_net_->GetCapabilities());

    params_.input_format = capabilities_.input_format;

    check_frequency_ =
        options.GetOrDefault<float>("freq", kDefaultCheckFrequency);
    switch (params_.mode) {
      case kCheckOnly:
        CERR << std::scientific << std::setprecision(1)
             << "Check mode: check only with relative tolerance "
             << params_.relative_tolerance << ", absolute tolerance "
             << params_.absolute_tolerance << ".";
        break;
      case kErrorDisplay:
        CERR << "Check mode: error display.";
        break;
      case kHistogram:
        CERR << "Check mode: histogram.";
        break;
    }
    CERR << "Check rate: " << std::fixed << std::setprecision(0)
         << 100 * check_frequency_ << "%.";
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    const double draw = Random::Get().GetDouble(1.0);
    const bool check = draw < check_frequency_;
    if (check) {
      std::unique_ptr<NetworkComputation> work_comp =
          work_net_->NewComputation();
      std::unique_ptr<NetworkComputation> check_comp =
          check_net_->NewComputation();
      return std::make_unique<CheckComputation>(params_, std::move(work_comp),
                                                std::move(check_comp));
    }
    return work_net_->NewComputation();
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

 private:
  CheckParams params_;

  // How frequently an iteration is checked (0: never, 1: always).
  double check_frequency_;
  std::unique_ptr<Network> work_net_;
  std::unique_ptr<Network> check_net_;
  NetworkCapabilities capabilities_;
};

std::unique_ptr<Network> MakeCheckNetwork(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  return std::make_unique<CheckNetwork>(weights, options);
}

REGISTER_NETWORK("check", MakeCheckNetwork, -800)

}  // namespace
}  // namespace lczero

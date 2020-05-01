
#pragma once

#include <string>

#include "neural/loader.h"

namespace lczero {
namespace python {

class Weights {
 public:
  using InputFormat = pblczero::NetworkFormat::InputFormat;
  using PolicyFormat = pblczero::NetworkFormat::PolicyFormat;
  using ValueFormat = pblczero::NetworkFormat::ValueFormat;
  using MovesLeftFormat = pblczero::NetworkFormat::MovesLeftFormat;

  // Exported methods.
  Weights(const std::string& filename)
      : filename_(filename.empty() ? DiscoverWeightsFile() : filename),
        weights_(LoadWeightsFromFile(filename_)) {}

  std::string_view filename() const { return filename_; }
  std::string_view license() const { return weights_.license(); }
  std::string min_version() const {
    const auto& ver = weights_.min_version();
    return std::to_string(ver.major()) + '.' + std::to_string(ver.minor()) +
           '.' + std::to_string(ver.patch());
  }
  InputFormat input_format() const {
    return weights_.format().network_format().input();
  }
  PolicyFormat policy_format() const {
    return weights_.format().network_format().policy();
  }
  ValueFormat value_format() const {
    return weights_.format().network_format().value();
  }
  MovesLeftFormat moves_left_format() const {
    return weights_.format().network_format().moves_left();
  }
  int blocks() const { return weights_.weights().residual_size(); }
  int filters() const {
    return weights_.weights().residual(0).conv1().weights().params().size() /
           2304;
  }

 private:
  const std::string filename_;
  const WeightsFile weights_;
};

}  // namespace python
}  // namespace lczero
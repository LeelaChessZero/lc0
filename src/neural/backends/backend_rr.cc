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

#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/exception.h"

namespace lczero {
namespace {

class RoundRobinBackend : public Backend {
 public:
  RoundRobinBackend(const OptionsDict& options,
                    const OptionsDict& backend_options) {
    const auto parents = backend_options.ListSubdicts();
    if (parents.empty()) {
      // If options are empty, or multiplexer configured in root object,
      // initialize on root object and default backend.
      auto backends = BackendManager::Get()->GetBackendNames();
      AddBackend(backends[0], options, backend_options);
    }

    for (const auto& name : parents) {
      AddBackend(name, options, backend_options.GetSubdict(name));
    }
  }

  void AddBackend(const std::string& name, const OptionsDict& opts,
                  const OptionsDict& backend_opts) {
    const std::string backend =
        backend_opts.GetOrDefault<std::string>("backend", name);
    backends_.emplace_back(
        BackendManager::Get()->CreateFromName(backend, opts, backend_opts));

    auto attributes = backends_.back()->GetAttributes();
    if (backends_.size() == 1) {
      attributes_ = attributes;
    } else {
      attributes_.Merge(attributes);
    }
  }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    const long long val = ++counter_;
    return backends_[val % backends_.size()]->CreateComputation();
  }

  BackendAttributes GetAttributes() const override { return attributes_; }

  ~RoundRobinBackend() {}

 private:
  std::vector<std::unique_ptr<Backend>> backends_;
  std::atomic<long long> counter_;
  BackendAttributes attributes_;
};

class RoundRobinFactory : public BackendFactory {
  int GetPriority() const override { return -999; }
  std::string_view GetName() const override { return "roundrobin"; }
  std::unique_ptr<Backend> Create(const OptionsDict& options,
                                  const OptionsDict& backend_options) override {
    return std::make_unique<RoundRobinBackend>(options, backend_options);
  }
};

REGISTER_BACKEND(RoundRobinFactory)

}  // namespace
}  // namespace lczero

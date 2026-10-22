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

BackendAttributes MergeAttributes(const BackendAttributes& in,
                                  const BackendAttributes& other) {
  BackendAttributes out = in;
  out.has_mlh &= other.has_mlh;
  out.has_wdl &= other.has_wdl;
  out.runs_on_cpu |= other.runs_on_cpu;
  out.suggested_num_search_threads += other.suggested_num_search_threads;
  out.recommended_batch_size =
      std::min(out.recommended_batch_size, other.recommended_batch_size);
  out.maximum_batch_size =
      std::min(out.maximum_batch_size, other.maximum_batch_size);
  return out;
}

struct ChildBackend {
  std::unique_ptr<Backend> backend_;
  std::string backend_name_;
  std::string backend_opts_str_;
};

class RoundRobinBackend : public Backend {
 public:
  RoundRobinBackend(const OptionsDict& options)
      : backend_opts_(
            options.Get<std::string>(SharedBackendParams::kBackendOptionsId)),
        weights_path_(
            options.Get<std::string>(SharedBackendParams::kWeightsId)) {
    OptionsDict backend_options;
    backend_options.AddSubdictFromString(backend_opts_);

    const auto subdicts = backend_options.ListSubdicts();

    if (subdicts.empty()) {
      // If options are empty, or multiplexer configured in root object,
      // initialize on root object and default backend.
      auto backends = BackendManager::Get()->GetBackendNames();
      OptionsDict child_opts(&options);
      child_opts.Set<std::string>(SharedBackendParams::kBackendId, backends[0]);
      AddBackend(backends[0], child_opts);
    }

    for (const auto& subdict_name : subdicts) {
      const auto& child_dict = backend_options.GetSubdict(subdict_name);
      const std::string child_backend =
          child_dict.GetOrDefault<std::string>("backend", subdict_name);

      // Children validate options independently. Backend-specific options for
      // heterogeneous children must go in their respective subdicts.
      auto child_options = backend_options.CloneScalars();
      child_options->MergeFrom(child_dict);
      child_options->Remove("backend");
      auto child_opts_str = child_options->Serialize();

      OptionsDict child_opts(&options);
      child_opts.Set<std::string>(SharedBackendParams::kBackendId,
                                  child_backend);
      child_opts.Set<std::string>(SharedBackendParams::kBackendOptionsId,
                                  child_opts_str);

      AddBackend(child_backend, child_opts);
    }

    UpdateConfiguration(options);
  }

  void AddBackend(const std::string& name, const OptionsDict& opts) {
    const std::string backend = opts.GetOrDefault<std::string>("backend", name);
    backends_.emplace_back(std::make_unique<ChildBackend>(ChildBackend{
        BackendManager::Get()->CreateFromName(backend, opts),
        opts.Get<std::string>(SharedBackendParams::kBackendId),
        opts.Get<std::string>(SharedBackendParams::kBackendOptionsId)}));

    auto attributes = backends_.back()->backend_->GetAttributes();
    if (backends_.size() == 1) {
      attributes_ = attributes;
    } else {
      attributes_ = MergeAttributes(attributes_, attributes);
    }
  }

  Backend::UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    auto rv = Backend::UpdateConfiguration(options);
    if (rv != UPDATE_OK) return rv;
    if (backend_opts_ !=
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId)) {
      return NEED_RESTART;
    }
    if (weights_path_ !=
        options.Get<std::string>(SharedBackendParams::kWeightsId)) {
      return NEED_RESTART;
    }
    for (auto& child : backends_) {
      OptionsDict child_opts(&options);
      child_opts.Set<std::string>(SharedBackendParams::kBackendId,
                                  child->backend_name_);
      child_opts.Set<std::string>(SharedBackendParams::kBackendOptionsId,
                                  child->backend_opts_str_);
      if (child->backend_->UpdateConfiguration(child_opts) == NEED_RESTART) {
        return NEED_RESTART;
      }
    }
    return UPDATE_OK;
  }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    const long long val = ++counter_;
    return backends_[val % backends_.size()]->backend_->CreateComputation();
  }

  BackendAttributes GetAttributes() const override { return attributes_; }

  ~RoundRobinBackend() {}

 private:
  std::vector<std::unique_ptr<ChildBackend>> backends_;
  std::atomic<long long> counter_;
  BackendAttributes attributes_;

  const std::string backend_opts_;
  const std::string weights_path_;
};

class RoundRobinFactory : public BackendFactory {
  int GetPriority() const override { return -999; }
  std::string_view GetName() const override { return "roundrobin"; }
  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    return std::make_unique<RoundRobinBackend>(options);
  }
};

REGISTER_BACKEND(RoundRobinFactory)

}  // namespace
}  // namespace lczero

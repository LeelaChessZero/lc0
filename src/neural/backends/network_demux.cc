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
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "neural/backend.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/exception.h"
#include "utils/mutex.h"
#include "utils/optionsdict.h"

namespace lczero {
namespace {

class DemuxingBackend;
class DemuxingComputation;

struct DemuxingWork {
  DemuxingComputation* source = nullptr;
  BackendComputation* computation = nullptr;
};

class DemuxingComputation final : public BackendComputation {
 public:
  explicit DemuxingComputation(DemuxingBackend* backend);

  ~DemuxingComputation() override {
    // Wait for notify_one to finish before destroying the condition variable.
    while (dataready_.load(std::memory_order_acquire) != -1) {
      SpinloopPause();
    }
  }

  size_t UsedBatchSize() const override {
    return input_count_.load(std::memory_order_relaxed);
  }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override;

  void ComputeBlocking() override;

  void NotifyComplete() {
    if (dataready_.fetch_sub(1, std::memory_order_release) == 1) {
      {
        std::lock_guard lock(mutex_);
      }
      dataready_cv_.notify_one();
      dataready_.store(-1, std::memory_order_release);
    }
  }

 private:
  DemuxingBackend* backend_;
  const size_t start_index_;
  std::atomic<size_t> input_count_ = 0;
  std::vector<std::unique_ptr<BackendComputation>> computations_;
  std::vector<DemuxingWork> work_items_;

  std::mutex mutex_;
  std::condition_variable dataready_cv_;
  std::atomic<int> dataready_ = -1;
};

class DemuxingChildBackend {
 public:
  DemuxingChildBackend(std::unique_ptr<Backend> backend,
                       std::string backend_name, std::string backend_opts_str,
                       int num_threads)
      : backend_(std::move(backend)),
        backend_name_(std::move(backend_name)),
        backend_opts_str_(std::move(backend_opts_str)) {
    try {
      for (int i = 0; i < num_threads; ++i) {
        threads_.emplace_back([this] { Worker(); });
      }
    } catch (...) {
      StopAndJoin();
      throw;
    }
  }

  ~DemuxingChildBackend() {
    StopAndJoin();
    while (!queue_.empty()) {
      queue_.front()->source->NotifyComplete();
      queue_.pop();
    }
  }

  void Enqueue(DemuxingWork* work) {
    {
      std::lock_guard lock(mutex_);
      queue_.push(work);
    }
    dataready_cv_.notify_one();
  }

  void Abort() {
    {
      std::lock_guard lock(mutex_);
      abort_.store(true, std::memory_order_relaxed);
    }
    dataready_cv_.notify_all();
  }

  void Worker() {
    while (!abort_.load(std::memory_order_relaxed)) {
      DemuxingWork* work = nullptr;
      {
        std::unique_lock lock(mutex_);
        dataready_cv_.wait(lock, [&] {
          return abort_.load(std::memory_order_relaxed) || !queue_.empty();
        });
        if (abort_.load(std::memory_order_relaxed)) return;
        if (!queue_.empty()) {
          work = queue_.front();
          queue_.pop();
        }
      }
      if (work) {
        work->computation->ComputeBlocking();
        work->source->NotifyComplete();
      }
    }
  }

  std::unique_ptr<Backend> backend_;
  std::string backend_name_;
  std::string backend_opts_str_;

 private:
  void StopAndJoin() {
    Abort();
    for (auto& t : threads_) {
      if (t.joinable()) t.join();
    }
  }

  std::atomic<bool> abort_{false};
  std::mutex mutex_;
  std::condition_variable dataready_cv_;
  std::vector<std::thread> threads_;
  std::queue<DemuxingWork*> queue_;
};

BackendAttributes AggregateDemuxAttributes(
    const std::vector<std::unique_ptr<DemuxingChildBackend>>& children,
    size_t batch_step) {
  if (children.empty()) return {};
  BackendAttributes result{
      .has_mlh = true,
      .has_wdl = true,
      .runs_on_cpu = true,
      .suggested_num_search_threads = 1,
      .recommended_batch_size = std::numeric_limits<int>::max(),
      .maximum_batch_size = std::numeric_limits<int>::max(),
  };
  for (const auto& child : children) {
    const auto& attr = child->backend_->GetAttributes();
    result.has_mlh &= attr.has_mlh;
    result.has_wdl &= attr.has_wdl;
    result.runs_on_cpu &= attr.runs_on_cpu;
    result.recommended_batch_size =
        std::min(result.recommended_batch_size, attr.recommended_batch_size);
    result.maximum_batch_size =
        std::min(result.maximum_batch_size, attr.maximum_batch_size);
  }
  // Grouped round robin cannot use a child's partial group until every child
  // has received the same number of full groups.
  const size_t minimum_capacity = result.maximum_batch_size;
  result.maximum_batch_size =
      children.size() * (minimum_capacity / batch_step) * batch_step +
      minimum_capacity % batch_step;
  result.recommended_batch_size =
      std::min<size_t>(result.recommended_batch_size * children.size(),
                       result.maximum_batch_size);
  return result;
}

class DemuxingBackend final : public Backend {
 public:
  explicit DemuxingBackend(const OptionsDict& options);
  ~DemuxingBackend() override { Abort(); }

  BackendAttributes GetAttributes() const override { return attrs_; }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<DemuxingComputation>(this);
  }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override;

 private:
  void Abort() {
    for (auto& b : backends_) {
      b->Abort();
    }
  }

  std::vector<std::unique_ptr<DemuxingChildBackend>> backends_;
  BackendAttributes attrs_;
  int batch_step_ = 1;
  std::atomic<size_t> start_index_ = 0;

  std::string backend_opts_;
  std::string weights_path_;

  friend class DemuxingComputation;
};

DemuxingBackend::DemuxingBackend(const OptionsDict& options)
    : backend_opts_(
          options.Get<std::string>(SharedBackendParams::kBackendOptionsId)),
      weights_path_(options.Get<std::string>(SharedBackendParams::kWeightsId)) {
  OptionsDict backend_options;
  backend_options.AddSubdictFromString(backend_opts_);

  batch_step_ = backend_options.GetOrDefault<int>("batch_step", 1);
  if (batch_step_ < 1) throw Exception("demux batch_step must be at least 1.");

  const auto subdicts = backend_options.ListSubdicts();
  if (subdicts.empty()) throw Exception("demux needs child backends.");

  auto* backend_manager = BackendManager::Get();
  for (const auto& subdict_name : subdicts) {
    const auto& child_dict = backend_options.GetSubdict(subdict_name);
    const std::string child_backend =
        child_dict.GetOrDefault<std::string>("backend", subdict_name);
    int child_threads = child_dict.GetOrDefault<int>("demux_threads", 0);

    // Children validate options independently. Backend-specific options for
    // heterogeneous children must go in their respective subdicts.
    auto child_options = backend_options.CloneScalars();
    child_options->Remove("batch_step");
    child_options->MergeFrom(child_dict);
    child_options->Remove("backend");
    child_options->Remove("demux_threads");
    auto child_opts_str = child_options->Serialize();

    OptionsDict child_opts(&options);
    child_opts.Set<std::string>(SharedBackendParams::kBackendId, child_backend);
    child_opts.Set<std::string>(SharedBackendParams::kBackendOptionsId,
                                child_opts_str);

    auto child_be = backend_manager->CreateFromParams(child_opts);
    if (child_threads == 0) {
      child_threads = child_be->GetAttributes().suggested_num_search_threads;
    }

    backends_.push_back(std::make_unique<DemuxingChildBackend>(
        std::move(child_be), child_backend, child_opts_str, child_threads));
  }

  attrs_ = AggregateDemuxAttributes(backends_, batch_step_);
  UpdateConfiguration(options);
}

Backend::UpdateConfigurationResult DemuxingBackend::UpdateConfiguration(
    const OptionsDict& options) {
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

DemuxingComputation::DemuxingComputation(DemuxingBackend* backend)
    : backend_(backend),
      start_index_(
          backend->start_index_.fetch_add(1, std::memory_order_relaxed) %
          backend->backends_.size()) {
  computations_.reserve(backend_->backends_.size());
  work_items_.reserve(backend_->backends_.size());
  for (const auto& child : backend_->backends_) {
    computations_.push_back(child->backend_->CreateComputation());
    work_items_.push_back({this, computations_.back().get()});
  }
}

BackendComputation::AddInputResult DemuxingComputation::AddInput(
    const EvalPosition& pos, EvalResultPtr result) {
  const size_t ticket = input_count_.fetch_add(1, std::memory_order_relaxed);
  assert(ticket < static_cast<size_t>(backend_->attrs_.maximum_batch_size));
  const size_t child =
      (start_index_ + ticket / backend_->batch_step_) % computations_.size();
  // We don't reclaim capacity for FETCHED_IMMEDIATELY as it's not trivial at
  // all without a lock.
  return computations_[child]->AddInput(pos, result);
}

void DemuxingComputation::ComputeBlocking() {
  // All AddInput calls must finish before computing, as with child backends.
  const size_t total_size = UsedBatchSize();
  const size_t groups = total_size / backend_->batch_step_ +
                        (total_size % backend_->batch_step_ != 0);
  const size_t jobs = std::min(groups, computations_.size());
  if (jobs == 0) return;

  // Publish the full count before any worker can complete a job.
  dataready_.store(jobs, std::memory_order_relaxed);
  for (size_t i = 0; i < jobs; ++i) {
    const size_t child = (start_index_ + i) % computations_.size();
    backend_->backends_[child]->Enqueue(&work_items_[child]);
  }

  // Wait until all backends complete their work.
  std::unique_lock lock(mutex_);
  dataready_cv_.wait(lock, [this]() {
    return dataready_.load(std::memory_order_acquire) <= 0;
  });
}

class DemuxingBackendFactory : public BackendFactory {
 public:
  int GetPriority() const override { return -1001; }
  std::string_view GetName() const override { return "demux"; }
  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    return std::make_unique<DemuxingBackend>(options);
  }
};

REGISTER_BACKEND(DemuxingBackendFactory)

}  // namespace
}  // namespace lczero

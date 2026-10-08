/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2025 The LCZero Authors

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
#include <condition_variable>
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
  size_t start = 0;
  size_t end = 0;
};

class DemuxingChildBackend {
 public:
  DemuxingChildBackend(std::unique_ptr<Backend> backend,
                       std::string backend_name,
                       std::string backend_opts_str,
                       std::string weights_path,
                       int num_threads,
                       std::atomic<bool>& abort)
      : backend_(std::move(backend)),
        backend_name_(std::move(backend_name)),
        backend_opts_str_(std::move(backend_opts_str)),
        weights_path_(std::move(weights_path)) {
    for (int i = 0; i < num_threads; ++i) {
      threads_.emplace_back([this, &abort] { Worker(abort); });
    }
  }

  ~DemuxingChildBackend();

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
    }
    dataready_cv_.notify_all();
  }

  void Worker(std::atomic<bool>& abort);

  std::unique_ptr<Backend> backend_;
  std::string backend_name_;
  std::string backend_opts_str_;
  std::string weights_path_;

 private:
  std::mutex mutex_;
  std::condition_variable dataready_cv_;
  std::vector<std::thread> threads_;
  std::queue<DemuxingWork*> queue_;
};

class DemuxingComputation final : public BackendComputation {
 public:
  explicit DemuxingComputation(DemuxingBackend* backend)
      : backend_(backend) {}

  ~DemuxingComputation() override {
    while (dataready_.load(std::memory_order_acquire) > 0) {
      SpinloopPause();
    }
  }

  size_t UsedBatchSize() const override { return entries_.size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    entries_.push_back({pos, result});
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override;

  void NotifyComplete() {
    if (dataready_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      std::lock_guard lock(mutex_);
      dataready_cv_.notify_one();
    }
  }

 private:
  struct Entry {
    EvalPosition pos;
    EvalResultPtr result;
  };

  DemuxingBackend* backend_;
  std::vector<Entry> entries_;
  std::vector<DemuxingWork> work_items_;

  std::mutex mutex_;
  std::condition_variable dataready_cv_;
  std::atomic<int> dataready_ = 0;

  friend class DemuxingChildBackend;
};

DemuxingChildBackend::~DemuxingChildBackend() {
  Abort();
  for (auto& t : threads_) {
    if (t.joinable()) t.join();
  }
  while (!queue_.empty()) {
    queue_.front()->source->NotifyComplete();
    queue_.pop();
  }
}

void DemuxingChildBackend::Worker(std::atomic<bool>& abort) {
  while (!abort.load(std::memory_order_relaxed)) {
    DemuxingWork* work = nullptr;
    {
      std::unique_lock lock(mutex_);
      dataready_cv_.wait(lock, [&] {
        return abort.load(std::memory_order_relaxed) || !queue_.empty();
      });
      if (abort.load(std::memory_order_relaxed)) return;
      if (!queue_.empty()) {
        work = queue_.front();
        queue_.pop();
      }
    }
    if (work) {
      auto computation = backend_->CreateComputation();
      auto& entries = work->source->entries_;
      for (size_t i = work->start; i < work->end; ++i) {
        computation->AddInput(entries[i].pos, entries[i].result);
      }
      computation->ComputeBlocking();
      work->source->NotifyComplete();
    }
  }
}

BackendAttributes AggregateDemuxAttributes(
    const std::vector<std::unique_ptr<DemuxingChildBackend>>& children) {
  BackendAttributes result{};
  if (children.empty()) return result;
  const auto& first = children[0]->backend_->GetAttributes();
  result.has_mlh = first.has_mlh;
  result.has_wdl = first.has_wdl;
  result.runs_on_cpu = first.runs_on_cpu;
  result.suggested_num_search_threads = first.suggested_num_search_threads;
  result.recommended_batch_size = first.recommended_batch_size;
  result.maximum_batch_size = first.maximum_batch_size;

  for (size_t i = 1; i < children.size(); ++i) {
    const auto& attr = children[i]->backend_->GetAttributes();
    result.has_mlh &= attr.has_mlh;
    result.has_wdl &= attr.has_wdl;
    result.runs_on_cpu &= attr.runs_on_cpu;
    result.suggested_num_search_threads += attr.suggested_num_search_threads;
    result.recommended_batch_size += attr.recommended_batch_size;
    result.maximum_batch_size += attr.maximum_batch_size;
  }
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
    abort_.store(true, std::memory_order_relaxed);
    for (auto& b : backends_) {
      b->Abort();
    }
  }

  std::vector<std::unique_ptr<DemuxingChildBackend>> backends_;
  BackendAttributes attrs_;
  int batch_step_ = 1;
  std::atomic<int64_t> start_index_ = 0;
  std::atomic<bool> abort_ = false;

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
  if (batch_step_ < 1) batch_step_ = 1;

  auto subdicts = backend_options.ListSubdicts();
  if (subdicts.empty()) {
    subdicts.push_back("");
  }

  auto* backend_manager = BackendManager::Get();
  auto all_backends = backend_manager->GetBackendNames();
  std::vector<std::string> valid_backends;
  for (const auto& b : all_backends) {
    if (b != "demux") valid_backends.push_back(b);
  }
  if (valid_backends.empty()) {
    throw Exception("No available backend to demux across.");
  }

  for (const auto& subdict_name : subdicts) {
    std::string child_backend;
    std::string child_weights;
    int child_threads = 0;

    if (subdict_name.empty()) {
      child_backend = backend_options.GetOrDefault<std::string>(
          "backend", valid_backends[0]);
      child_threads = backend_options.GetOrDefault<int>("threads", 0);
    } else {
      const auto& child_dict = backend_options.GetSubdict(subdict_name);
      child_threads = child_dict.GetOrDefault<int>("threads", 0);
      if (child_dict.Exists<std::string>("backend")) {
        child_backend = child_dict.Get<std::string>("backend");
      } else if (backend_options.Exists<std::string>("backend")) {
        child_backend = backend_options.Get<std::string>("backend");
      } else if (backend_manager->GetFactoryByName(subdict_name) != nullptr) {
        child_backend = subdict_name;
      } else {
        child_backend = valid_backends[0];
      }
      if (child_dict.Exists<std::string>("weights")) {
        child_weights = child_dict.Get<std::string>("weights");
      }
    }

    std::string child_opts_str = backend_options.FlattenSubdictToString(
        subdict_name, {"backend", "threads"});

    OptionsDict child_opts(&options);
    child_opts.Set<std::string>(SharedBackendParams::kBackendId, child_backend);
    child_opts.Set<std::string>(SharedBackendParams::kBackendOptionsId,
                                child_opts_str);
    if (!child_weights.empty()) {
      child_opts.Set<std::string>(SharedBackendParams::kWeightsId,
                                  child_weights);
    }

    auto child_be = backend_manager->CreateFromParams(child_opts);
    if (child_threads == 0) {
      child_threads = std::max(
          1, child_be->GetAttributes().suggested_num_search_threads);
    }

    backends_.push_back(std::make_unique<DemuxingChildBackend>(
        std::move(child_be), child_backend, child_opts_str, child_weights,
        child_threads, abort_));
  }

  attrs_ = AggregateDemuxAttributes(backends_);
  if (backend_options.Exists<int>("max_batch")) {
    attrs_.maximum_batch_size = backend_options.Get<int>("max_batch");
    attrs_.recommended_batch_size =
        std::min(attrs_.recommended_batch_size, attrs_.maximum_batch_size);
  }
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
    if (!child->weights_path_.empty()) {
      child_opts.Set<std::string>(SharedBackendParams::kWeightsId,
                                  child->weights_path_);
    }
    if (child->backend_->UpdateConfiguration(child_opts) == NEED_RESTART) {
      return NEED_RESTART;
    }
  }
  return UPDATE_OK;
}

void DemuxingComputation::ComputeBlocking() {
  if (entries_.empty()) return;

  const size_t num_backends = backend_->backends_.size();
  assert(num_backends > 0);

  const int batch_step = backend_->batch_step_;
  const size_t total_size = entries_.size();

  const int splits = 1 + (total_size - 1) / batch_step;
  const int split_size_per_backend = splits / num_backends;
  const int extra_split_backends =
      splits - split_size_per_backend * num_backends;

  const size_t start_index =
      backend_->start_index_.fetch_add(std::max(1, extra_split_backends),
                                       std::memory_order_relaxed) %
      num_backends;
  const size_t end_index =
      (start_index + extra_split_backends) % num_backends;

  const int work_items = split_size_per_backend > 0 ? num_backends
                                                    : extra_split_backends;
  dataready_.store(work_items, std::memory_order_relaxed);
  work_items_.clear();
  work_items_.reserve(work_items);

  size_t work_start = 0;
  size_t i = start_index;

  // First send work to backends which get extra work.
  int split_size = split_size_per_backend + 1;
  for (; i != end_index; i = (i + 1) % num_backends) {
    assert(work_start != total_size);
    size_t work_end =
        std::min(work_start + split_size * batch_step, total_size);
    work_items_.push_back(DemuxingWork{
        .source = this,
        .start = work_start,
        .end = work_end,
    });
    backend_->backends_[i]->Enqueue(&work_items_.back());
    work_start = work_end;
  }

  // Queue remaining work items which don't get extra work.
  split_size--;
  if (split_size > 0) {
    do {
      assert(work_start != total_size);
      size_t work_end =
          std::min(work_start + split_size * batch_step, total_size);
      work_items_.push_back(DemuxingWork{
          .source = this,
          .start = work_start,
          .end = work_end,
      });
      backend_->backends_[i]->Enqueue(&work_items_.back());
      work_start = work_end;
      i = (i + 1) % num_backends;
    } while (i != start_index);
  }

  assert(work_start == total_size);
  assert(work_items == static_cast<int>(work_items_.size()));

  // Wait until all backends complete their work.
  std::unique_lock lock(mutex_);
  dataready_cv_.wait(lock, [this]() {
    return dataready_.load(std::memory_order_acquire) == 0;
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

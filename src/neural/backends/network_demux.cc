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
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <queue>
#include <thread>

#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/shared_params.h"
#include "utils/atomic_vector.h"
#include "utils/fastmath.h"

namespace lczero {
namespace {

class DemuxingComputation;

struct DemuxingWork {
  DemuxingComputation* source_ = nullptr;
  std::unique_ptr<NetworkComputation> computation_;
  int start_ = 0;
  int end_ = 0;

  DemuxingWork(int sample) : end_(sample) {}
  DemuxingWork(DemuxingComputation* source, int start, int end)
      : source_(source), start_(start), end_(end) {
    assert(start_ != end_);
  }

  void ProcessResults();

  auto operator<=>(const DemuxingWork& b) const { return end_ <=> b.end_; }
};

class DemuxingComputation;
class DemuxingChildBackend;

class DemuxingChildBackend {
 public:
  ~DemuxingChildBackend();

  void Assign(std::unique_ptr<Network>&& network, const OptionsDict& opts,
              std::atomic<bool>& abort) {
    network_ = std::move(network);
    int nn_threads = opts.GetOrDefault<int>("threads", 0);
    if (nn_threads == 0) {
      nn_threads = network_->GetThreads();
    }
    for (int i = 0; i < nn_threads; i++) {
      threads_.emplace_back([&] { Worker(abort); });
    }
  }

  void Enqueue(DemuxingWork* work) {
    {
      std::unique_lock lock(mutex_);
      queue_.push(work);
    }
    dataready_cv_.notify_one();
  }

  void Abort() {
    {
      std::unique_lock lock(mutex_);
    }
    dataready_cv_.notify_all();
  }

  void Worker(std::atomic<bool>& abort);

 private:
  std::mutex mutex_;
  std::condition_variable dataready_cv_;
  std::vector<std::thread> threads_;
  std::unique_ptr<Network> network_;
  std::queue<DemuxingWork*> queue_;
};

class DemuxingBackend final : public Backend {
 public:
  DemuxingBackend(const std::optional<WeightsFile>& weights,
                  const OptionsDict& options, const OptionsDict& backend_options)
      : backends_(std::max(size_t(1), backend_options.ListSubdicts().size())),
        backend_opts_(
            options.Get<std::string>(SharedBackendParams::kBackendOptionsId)),
        weights_path_(
            options.Get<std::string>(SharedBackendParams::kWeightsId)) {
    UpdateConfiguration(options);
    const auto parents = backend_options.ListSubdicts();
    if (parents.empty()) {
      // If options are empty, or multiplexer configured in root object,
      // initialize on root object and default backend.
      auto backends = NetworkFactory::Get()->GetBackendsList();
      AddBackend(0, backends[0], weights, backend_options);
    }

    int i = 0;
    for (const auto& name : parents) {
      AddBackend(i++, name, weights, backend_options.GetSubdict(name));
    }
  }

  void AddBackend(int index, const std::string& name,
                  const std::optional<WeightsFile>& weights,
                  const OptionsDict& opts) {
    const std::string backend = opts.GetOrDefault<std::string>("backend", name);

    auto network = NetworkFactory::Get()->Create(backend, weights, opts);
    const NetworkCapabilities& caps = network->GetCapabilities();

    if (index == 0) {
      attrs_ = BackendAttributes(*network);
      input_format_ = caps.input_format;
    } else {
      attrs_ += BackendAttributes(*network);
      if (input_format_ != caps.input_format) {
        throw Exception("Incompatible input formats, " +
                        std::to_string(input_format_) + " vs " +
                        std::to_string(caps.input_format));
      }
    }
    backends_[index].Assign(std::move(network), opts, abort_);
  }

  std::unique_ptr<BackendComputation> CreateComputation() override;

  BackendAttributes GetAttributes() const override { return attrs_; }

  ~DemuxingBackend() { Abort(); }

  void Abort() {
    abort_.store(true, std::memory_order_relaxed);
    for (auto& b : backends_) {
      b.Abort();
    }
  }

  UpdateConfigurationResult UpdateConfiguration(
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
    softmax_policy_temperature_ =
        1.0f / options.Get<float>(SharedBackendParams::kPolicySoftmaxTemp);
    fill_empty_history_ = EncodeHistoryFill(
        options.Get<std::string>(SharedBackendParams::kHistoryFill));
    return UPDATE_OK;
  }

 private:
  std::vector<DemuxingChildBackend> backends_;
  BackendAttributes attrs_;
  pblczero::NetworkFormat::InputFormat input_format_;
  float softmax_policy_temperature_;
  FillEmptyHistory fill_empty_history_;
  std::atomic<int64_t> start_index_ = 0;
  std::atomic<bool> abort_ = false;

  // Cache cold variables
  const std::string backend_opts_;
  const std::string weights_path_;

  friend class DemuxingComputation;
};

class DemuxingComputation final : public BackendComputation {
  std::tuple<const std::unique_ptr<NetworkComputation>&, int> GetParent(
      int sample) const {
    auto iter =
        std::lower_bound(children_.begin(), children_.end(), sample + 1);
    assert(iter != children_.end());
    assert(sample >= iter->start_);
    assert(sample < iter->end_);
    return {iter->computation_, sample - iter->start_};
  }

 public:
  DemuxingComputation(DemuxingBackend* backend)
      : backend_(backend), entries_(backend_->attrs_.maximum_batch_size) {}
  ~DemuxingComputation() {
    // Wait for other threads to stop using this thread. It must be spinloop for
    // correct synchronization between notify_one and destructor.
    while (dataready_.load(std::memory_order_acquire) != -1) {
      SpinloopPause();
    }
  }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    int transform;
    const size_t idx = entries_.emplace_back(Entry{
        .input = EncodePositionForNN(backend_->input_format_, pos.pos, 8,
                                     backend_->fill_empty_history_, &transform),
        .legal_moves = MoveList(pos.legal_moves.begin(), pos.legal_moves.end()),
        .result = result,
        .transform = 0});
    entries_[idx].transform = transform;
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking(ComputationCallback callback) override;

  size_t UsedBatchSize() const override { return entries_.size(); }

  void NotifyComplete() {
    if (1 == dataready_.fetch_sub(1, std::memory_order_release)) {
      {
        std::lock_guard lock(mutex_);
      }
      dataready_cv_.notify_one();
      dataready_.store(-1, std::memory_order_release);
    }
  }

  void ProcessResults(const DemuxingWork& work);

  void SoftmaxPolicy(std::span<float> dst, const DemuxingWork& work,
                     size_t index);

 private:
  struct Entry {
    InputPlanes input;
    MoveList legal_moves;
    EvalResultPtr result;
    int transform;
  };

  DemuxingBackend* backend_;
  AtomicVector<Entry> entries_;
  std::vector<DemuxingWork> children_;
  ComputationCallback callback_;

  std::mutex mutex_;
  std::condition_variable dataready_cv_;
  std::atomic<int> dataready_ = -1;
  std::atomic<bool> first_done_ = false;

  friend class DemuxingChildBackend;
};

void DemuxingWork::ProcessResults() { source_->ProcessResults(*this); }

void DemuxingComputation::ProcessResults(const DemuxingWork& work) {
  size_t size = work.end_ - work.start_;
  for (size_t i = 0; i < size; ++i) {
    const EvalResultPtr& result = entries_[work.start_ + i].result;
    if (result.q) *result.q = work.computation_->GetQVal(i);
    if (result.d) *result.d = work.computation_->GetDVal(i);
    if (result.m) *result.m = work.computation_->GetMVal(i);
    if (!result.p.empty()) SoftmaxPolicy(result.p, work, i);
  }
}

void DemuxingComputation::SoftmaxPolicy(std::span<float> dst,
                                        const DemuxingWork& work,
                                        size_t index) {
  const std::vector<Move>& moves = entries_[work.start_ + index].legal_moves;
  const int transform = entries_[work.start_ + index].transform;
  // Copy the values to the destination array and compute the maximum.
  assert(dst.size() == moves.size());
  const float max_p = std::accumulate(
      moves.begin(), moves.end(), std::numeric_limits<float>::lowest(),
      [&, counter = 0](float max_p, const Move& move) mutable {
        return std::max(max_p, dst[counter++] = work.computation_->GetPVal(
                                   index, MoveToNNIndex(move, transform)));
      });
  // Compute the softmax and compute the total.
  const float temperature = backend_->softmax_policy_temperature_;
  float total = std::accumulate(
      dst.begin(), dst.end(), 0.0f, [&](float total, float& val) {
        return total + (val = FastExp((val - max_p) * temperature));
      });
  const float scale = total > 0.0f ? 1.0f / total : 1.0f;
  // Scale the values to sum to 1.0.
  std::for_each(dst.begin(), dst.end(), [&](float& val) { val *= scale; });
}

std::unique_ptr<BackendComputation> DemuxingBackend::CreateComputation() {
  return std::make_unique<DemuxingComputation>(this);
}

DemuxingChildBackend::~DemuxingChildBackend() {
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
  while (!queue_.empty()) {
    queue_.front()->source_->NotifyComplete();
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
      work->computation_ = network_->NewComputation();
      auto& entries = work->source_->entries_;
      for (int i = work->start_; i < work->end_; i++) {
        work->computation_->AddInput(std::move(entries[i].input));
      }
      work->computation_->ComputeBlocking();
      bool expected = false;
      if (work->source_->first_done_.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
        work->source_->callback_(ComputationEvent::FIRST_BACKEND_IDLE);
      }
      work->ProcessResults();
      work->source_->NotifyComplete();
    }
  }
}

void DemuxingComputation::ComputeBlocking(ComputationCallback callback) {
  assert(UsedBatchSize() != 0);
  callback_ = callback;
  // Calculate batch_step_ size split count.
  int splits =
      1 + (UsedBatchSize() - 1) / backend_->attrs_.preferred_batch_step;
  // Calculate the minimum number of splits per backend.
  int split_size_per_backend = splits / backend_->backends_.size();
  // Calculate how many backends get extra work.
  int extra_split_backends =
      splits - split_size_per_backend * backend_->backends_.size();

  // Find the first backend which got less work from the previous batch.
  size_t start_index = backend_->start_index_.fetch_add(
                           extra_split_backends, std::memory_order_relaxed) %
                       backend_->backends_.size();

  size_t end_index =
      (start_index + extra_split_backends) % backend_->backends_.size();
  size_t work_start = 0;
  int work_items = split_size_per_backend > 0 ? backend_->backends_.size()
                                              : extra_split_backends;
  // First store the work item count and reserve memory from them.
  dataready_.store(work_items, std::memory_order_relaxed);
  children_.reserve(work_items);
  size_t i = start_index;
  // First send work to backends which get extra work.
  int split_size = split_size_per_backend + 1;
  for (; i != end_index; i = (i + 1) % backend_->backends_.size()) {
    assert(work_start != UsedBatchSize());
    size_t work_end =
        work_start + split_size * backend_->attrs_.preferred_batch_step;
    work_end = std::min(work_end, UsedBatchSize());
    children_.emplace_back(this, work_start, work_end);
    backend_->backends_[i].Enqueue(&children_.back());
    work_start = work_end;
  }
  // Queue remaining work items which don't get extra work.
  split_size--;
  if (split_size > 0) {
    do {
      assert(work_start != UsedBatchSize());
      size_t work_end =
          work_start + split_size * backend_->attrs_.preferred_batch_step;
      work_end = std::min(work_end, UsedBatchSize());
      children_.emplace_back(this, work_start, work_end);
      backend_->backends_[i].Enqueue(&children_.back());
      work_start = work_end;
      i = (i + 1) % backend_->backends_.size();
    } while (i != start_index);
  }
  assert(work_start == UsedBatchSize());
  assert(work_items == (int)children_.size());
  // Wait until all backends complete their work.
  std::unique_lock<std::mutex> lock(mutex_);
  dataready_cv_.wait(lock, [this]() {
    return dataready_.load(std::memory_order_acquire) <= 0;
  });
}

class DemuxingBackendFactory : public BackendFactory {
  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    const std::string backend_options_string =
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId);
    OptionsDict backend_options;
    backend_options.AddSubdictFromString(backend_options_string);

    std::string net_path =
        options.Get<std::string>(SharedBackendParams::kWeightsId);
    std::optional<WeightsFile> weights = LoadWeights(net_path);
    return std::make_unique<DemuxingBackend>(weights, options, backend_options);
  }

  std::string_view GetName() const override {
    using namespace std::string_view_literals;
    return "demux"sv;
  }

  int GetPriority() const override { return -1001; }
};

BackendManager::Register register_demux(
    std::make_unique<DemuxingBackendFactory>());

// REGISTER_BACKEND("demux", MakeDemuxingNetwork, -1001)

}  // namespace
}  // namespace lczero

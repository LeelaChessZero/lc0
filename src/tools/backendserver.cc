/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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

#include "tools/backendserver.h"

#include <absl/cleanup/cleanup.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <utility>

#include "chess/uciloop.h"
#include "neural/backends/client/proto.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/asio.h"
#include "utils/configfile.h"
#include "utils/optionsparser.h"
#include "utils/trace.h"

namespace lczero {
namespace {

const OptionId kLogFileId{
    {.long_flag = "logfile",
     .uci_option = "LogFile",
     .help_text = "Write log to that file. Special value <stderr> to "
                  "output the log to the console.",
     .short_flag = 'l',
     .visibility = OptionId::kAlwaysVisible}};
const OptionId kMinibatchSizeOptionId{
    "minibatch-size", "MinibatchSize",
    "How many positions the engine tries to batch together for parallel NN "
    "computation. Larger batches may reduce strength a bit, especially with a "
    "small number of playouts. Set to 0 to use a backend suggested value."};
const OptionId kProtocolOptionId{
    "protocol", "Protocol",
    "Protocol to use for client connections (tcp or unix)."};
const OptionId kPipeNameOptionId{"pipe-name", "PipeName",
                                 "Named pipe allows client connections."};
const OptionId kHostOptionId{"tcp-host", "TCPHost",
                             "Host to listen on for TCP."};
const OptionId kPortOptionId{"tcp-port", "TCPPort",
                             "Port to listen on for TCP."};
const OptionId kNetworkDirectoryOptionId{
    "network-directory", "NetworkDirectory",
    "Directory where neural network files are stored."};
const OptionId kAcceptLimitOptionId{
    "accept-limit", "AcceptLimit",
    "Maximum number of accepted client connections."};
const OptionId kStatisticsIntervalOptionId{
    "statistics-interval", "StatisticsInterval",
    "Interval in seconds between statistics messages."};

const std::string kDefaultNetworkDirectory = ".";

class BackendHandler;
class ClientComputation;
using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct QueueItem {
  QueueItem(BackendHandler* backend, ClientComputation* computation,
            size_t first, size_t last)
      : backend_(backend),
        computation_(computation),
        first_(first),
        last_(last),
        enqueue_time_(Clock::now()) {}
  BackendHandler* backend_ = nullptr;
  ClientComputation* computation_ = nullptr;
  size_t first_ = 0;
  size_t last_ = 0;
  TimePoint enqueue_time_ = Clock::now();
};

class BackendHandler {
 public:
  struct Statistics {
    size_t batches_ = 0;
    size_t positions_ = 0;
    size_t queue_positions_ = 0;
    size_t minibatch_size_ = 0;
    size_t batches_in_flight_ = 0;
    double max_nps_ = 0.0;
  };

  static constexpr double kGpuIdleBufferMultiplier = 0.8;

  BackendHandler(const OptionsDict& params) : params_(params) {}

  ~BackendHandler() {
    LCTRACE_FUNCTION_SCOPE;
    {
      SpinMutex::Lock lock(mutex_);
      exit_ = true;
      cv_.notify_all();
    }
    for (auto& thread : backend_threads_) {
      thread.join();
    }
  }

  template <typename Callback>
  void EnsureLoaded(const std::string& net, Callback&& callback);

  void EnsureReady() const { SpinMutex::Lock lock(mutex_); }

  size_t Threads() const {
    SpinMutex::Lock lock(mutex_);
    return backend_threads_.size();
  }

  BackendAttributes GetAttributes() {
    assert(backend_);
    auto attrs = backend_->GetAttributes();
    int minibatch_size = params_.Get<int>(kMinibatchSizeOptionId);
    if (minibatch_size > 0) {
      attrs.recommended_batch_size = minibatch_size;
    }
    return attrs;
  }

  std::tuple<bool, unsigned> AddBatchToQueue(QueueItem& item, size_t& count) {
    LCTRACE_FUNCTION_SCOPE;
    SpinMutex::Lock lock(mutex_);
    unsigned batches = computations_in_flight_ + queued_computations_;
    assert(backend_);
    assert(batches < backend_threads_.size());
    queue_.emplace(item);
    bool flushed = false;
    size_t flushed_size = queued_computations_ * minibatch_size_;
    size_t next_full_batch_size = minibatch_size_ * (queued_computations_ + 1);
    if (queue_size_ < flushed_size) {
      // Append to the queue because backend haven't yet processed queued items.
      count = std::min(count, flushed_size - queue_size_);
      queue_.back().last_ = queue_.back().first_ + count;
    } else if (count + queue_size_ >= next_full_batch_size) {
      // Item doesn't fully fit into the queue, we need to split it.
      count = next_full_batch_size - queue_size_;
      queued_computations_++;
      batches++;
      queue_.back().last_ = queue_.back().first_ + count;
      cv_.notify_one();
      flushed = true;
    }
    item.first_ += count;
    size_t pending_size = queue_size_;
    queue_size_ += count;
    TRACE << this << " Backend queue size: " << queue_size_
          << " pending size: " << pending_size
          << " computations in flight: " << computations_in_flight_
          << " queued computations: " << queued_computations_
          << " flushed: " << flushed << " batches: " << batches;
    return {flushed, batches};
  }

  unsigned Flush() {
    LCTRACE_FUNCTION_SCOPE;
    SpinMutex::Lock lock(mutex_);
    TRACE << this << " Flush called. Pending size: " << PendingSize()
          << " computations in flight: " << computations_in_flight_
          << " queued computations: " << queued_computations_;
    unsigned batches = computations_in_flight_ + queued_computations_;
    if (PendingSize() > 0 && batches < backend_threads_.size()) {
      auto pending = PendingSize();
      auto cif = computations_in_flight_;
      auto qc = queued_computations_;
      auto qs = queue_size_;
      queued_computations_++;
      batches++;
      TRACE << this << " Forcing backend flush, new queue size: " << qs
            << " pending size: " << pending
            << " computations in flight: " << cif
            << " queued computations: " << qc;
      cv_.notify_one();
    }
    return batches;
  }

  void FlushIfIdling() {
    LCTRACE_FUNCTION_SCOPE;
    SpinMutex::Lock lock(mutex_);
    if (PendingSize() == 0 ||
        computations_in_flight_ + queued_computations_ > 0) {
      return;
    }
    auto pending = PendingSize();
    auto cif = computations_in_flight_;
    auto qc = queued_computations_;
    queued_computations_++;
    cv_.notify_one();
    lock.unlock();
    TRACE << this << " Idling flush called. Pending size: " << pending
          << " computations in flight: " << cif
          << " queued computations: " << qc;
  }

  unsigned StartBatch(size_t size) REQUIRES(mutex_) {
    queued_computations_--;
    computations_in_flight_++;
    TRACE << this << " Starting batch of size: " << size
          << " computations in flight " << computations_in_flight_
          << " queued computations " << queued_computations_;
    queue_size_ -= size;
    gpu_work_size_ += size;
    return computations_in_flight_ + queued_computations_;
  }

  std::tuple<unsigned, TimePoint> CompleteBatch(size_t size, TimePoint now) {
    SpinMutex::Lock lock(mutex_);
    statistics_.batches_++;
    statistics_.positions_ += size;
    TimePoint old = last_complete_time_;
    last_complete_time_ = now;
    gpu_work_size_ -= size;
    TRACE << this << " End of  batch " << size << " computations in flight "
          << computations_in_flight_ << " queued computations "
          << queued_computations_;
    unsigned queued_batches = computations_in_flight_--;
    queued_batches += queued_computations_;
    if (size == 0) return {queued_batches, now};
    if (old == TimePoint()) {
      return {queued_batches, now};
    }
    auto seconds = std::chrono::duration<double>(now - old).count();
    auto nps = size / seconds;
    if (nps > max_nps_) {
      TRACE << "New max NPS for backend " << this << ": " << nps
            << " previous: " << max_nps_;
      max_nps_ = nps;
    }
    TimePoint timer =
        now + std::chrono::duration_cast<Clock::duration>(
                  std::chrono::duration<double>{kGpuIdleBufferMultiplier *
                                                gpu_work_size_ / max_nps_});
    TRACE << this << " idle: " << (timer - now).count()
          << " max_nps: " << max_nps_ << " gpu_work_size: " << gpu_work_size_
          << " == " << kGpuIdleBufferMultiplier * gpu_work_size_ / max_nps_;
    return {queued_batches, timer};
  }

  size_t GetPendingSize() const {
    SpinMutex::Lock lock(mutex_);
    return PendingSize();
  }

  unsigned GetComputationsInFlight() const {
    SpinMutex::Lock lock(mutex_);
    return computations_in_flight_;
  }

  Statistics GetStatistics() {
    Statistics stats;
    SpinMutex::Lock lock(mutex_);
    std::swap(stats, statistics_);
    stats.queue_positions_ = queue_size_;
    stats.minibatch_size_ = minibatch_size_;
    stats.batches_in_flight_ = computations_in_flight_;
    stats.max_nps_ = max_nps_;
    return stats;
  }

 private:
  size_t PendingSize() const {
    size_t flushed_size = queued_computations_ * minibatch_size_;
    if (flushed_size > queue_size_) {
      return 0;
    }
    return queue_size_ - flushed_size;
  }

  void Worker();
  mutable SpinMutex mutex_;
  std::condition_variable_any cv_;
  std::unique_ptr<Backend> backend_;
  std::vector<std::thread> backend_threads_;
  std::vector<
      std::function<void(const std::error_code&, const BackendAttributes&)>>
      pending_callbacks_;
  std::queue<QueueItem> queue_;

  TimePoint last_complete_time_;

  double max_nps_ = 0.0;

  unsigned minibatch_size_ = 0;
  unsigned queue_size_ = 0;
  unsigned gpu_work_size_ = 0;
  unsigned queued_computations_ = 0;
  unsigned computations_in_flight_ = 0;
  bool exit_ = false;

  const OptionsDict& params_;

  Statistics statistics_;
};

class ClientComputation;

using BackendMap = std::map<std::string, BackendHandler, std::less<void>>;

class SharedQueue {
 public:
  static SharedQueue& Get() {
    static SharedQueue instance;
    return instance;
  }

  void Enqueue(unsigned priority, BackendHandler* backend,
               ClientComputation* computation, size_t first, size_t last) {
    assert(priority < client::kMaxComputationPriority);
    SpinMutex::Lock lock(mutex_);
    priority_queue_[priority].emplace_back(backend, computation, first, last);
    TRACE << "Enqueued work. Priority: " << priority
          << " Queue size: " << priority_queue_[priority].size()
          << " computations: " << highest_backend_computations_
          << " pending flush: " << queue_has_pending_flush_;
    if (highest_backend_computations_ == 0 ||
        (highest_backend_computations_ < max_batches_in_flight_ &&
         !queue_has_pending_flush_)) {
      PushWorkToBackend();
    }
  }

  BackendMap& GetBackendMap(const OptionsDict& options,
                            StdoutUciResponder& responder) {
    options_ = &options;
    responder_ = &responder;
    return backend_map_;
  }
  BackendMap& GetBackendMap() { return backend_map_; }
  void SetDiscovery(const std::filesystem::path& path) {
    network_discovery_ = backend_map_.find(path.filename().string());
  }

  BackendMap::iterator GetDiscovery() const { return network_discovery_; }

  void BackendThreads(size_t count) {
    SpinMutex::Lock lock(mutex_);
    max_batches_in_flight_ = count;
  }

  void ComputationDone(unsigned computations_in_flight, TimePoint idle_time) {
    LCTRACE_FUNCTION_SCOPE;
    SpinMutex::Lock lock(mutex_);
    if (highest_backend_computations_ != computations_in_flight) {
      return;
    }
    TRACE << "Computation done. Adjusting highest_backend_computations_ from "
          << highest_backend_computations_ << " based on "
          << computations_in_flight
          << " idle_time: " << idle_time.time_since_epoch();
    if (highest_backend_computations_ == max_batches_in_flight_) {
      for ([[maybe_unused]] const auto& backend : backend_map_) {
        assert(backend.second.GetComputationsInFlight() !=
               max_batches_in_flight_);
      }
      highest_backend_computations_--;
      idle_timer_target_ = idle_time;
      ScheduleFlush();
      FlushIdlingBackends();
    } else {
      unsigned computations = 0;
      for (const auto& backend : backend_map_) {
        computations =
            std::max(computations, backend.second.GetComputationsInFlight());
      }
      highest_backend_computations_ = computations;
      if (!queue_has_pending_flush_) {
        if (!Empty()) {
          PushWorkToBackend();
        }
      }
    }
  }

  void AddToFlushTimer(Clock::duration increment) {
    SpinMutex::Lock lock(mutex_);
    if (!queue_has_pending_flush_) {
      return;
    }
    idle_timer_target_ += increment;
    TRACE << "Adjusting flush timer for backend to "
          << idle_timer_target_.time_since_epoch();
    flush_timer_.expires_at(idle_timer_target_);
    WaitOnFlushTimer();
  }

  void Stop() {
    LCTRACE_FUNCTION_SCOPE;
    io_context_.stop();
  }

  void Close() {
    LCTRACE_FUNCTION_SCOPE;
    BackendMap map;
    io_context_.stop();
    {
      SpinMutex::Lock lock(mutex_);
      statistics_timer_.cancel();
      flush_timer_.cancel();
      map = std::move(backend_map_);
    }
    map.clear();
  }

  asio::io_context& GetContext() { return io_context_; }

  void StartServer();

  void EnsureReady() const {
    SpinMutex::Lock lock(mutex_);
    for (auto& backend : backend_map_) {
      backend.second.EnsureReady();
    }
  }

  void NewConnection() {
    SpinMutex::Lock lock(mutex_);
    active_connections_++;
    if (active_connections_ == 1) {
      StatisticsTimerSetup(Clock::now());
    }
  }

  void ConnectionClosed() {
    SpinMutex::Lock lock(mutex_);
    active_connections_--;
  }

 private:
  SharedQueue() = default;

  SharedQueue(const SharedQueue&) = delete;
  SharedQueue& operator=(const SharedQueue&) = delete;

  bool Empty() REQUIRES(mutex_) {
    for (const auto& q : priority_queue_) {
      if (!q.empty()) {
        return false;
      }
    }
    return true;
  }

  std::tuple<unsigned, QueueItem&> Front(TimePoint evaluation_time)
      REQUIRES(mutex_) {
    size_t selected_priority;
    TimePoint::duration preferred_delay;
    bool first = true;
    for (size_t priority = 0; priority < client::kMaxComputationPriority;
         ++priority) {
      if (priority_queue_[priority].empty()) continue;
      TimePoint::duration weighted_delay =
          evaluation_time - priority_queue_[priority].front().enqueue_time_;
      // Each priority reperesent doubling of remaining time to make a move.
      weighted_delay *= 1UL << (client::kMaxComputationPriority - priority - 1);
      if (first || weighted_delay > preferred_delay) {
        preferred_delay = weighted_delay;
        selected_priority = priority;
        first = false;
      }
    }
    return {selected_priority, priority_queue_[selected_priority].front()};
  }

  // TODO: Does this need to be dynamically tuned?
  static constexpr Clock::duration kEstimatedNetworkEvaluationTime =
      std::chrono::milliseconds(12);

  void PushWorkToBackend() REQUIRES(mutex_) {
    LCTRACE_FUNCTION_SCOPE;
    TRACE << "Pushing work to backend: " << highest_backend_computations_ << "/"
          << max_batches_in_flight_;
    bool needs_flush = true;
    auto evaluation_time = Clock::now();
    evaluation_time += kEstimatedNetworkEvaluationTime;
    while (!Empty() &&
           highest_backend_computations_ != max_batches_in_flight_) {
      auto queue_front = Front(evaluation_time);
      auto priority = std::get<0>(queue_front);
      auto& item = std::get<1>(queue_front);
      size_t batch_size = item.last_ - item.first_;
      auto [flushed, batches] =
          item.backend_->AddBatchToQueue(item, batch_size);
      if (item.first_ == item.last_) {
        priority_queue_[priority].pop_front();
      }
      if (flushed) {
        needs_flush = false;
        highest_backend_computations_ =
            std::max(highest_backend_computations_, batches);
        TRACE << "Flushed " << batches;
      }
    }
    if (highest_backend_computations_ == max_batches_in_flight_) {
      TRACE << "Backend queue full. Cancel flush timer.";
      flush_timer_.cancel();
    } else if (needs_flush) {
      ScheduleFlush();
    }
  }

  void FlushIdlingBackends() REQUIRES(mutex_) {
    LCTRACE_FUNCTION_SCOPE;
    TRACE << "Flushing idling backends.";
    for (auto& backend : backend_map_) {
      backend.second.FlushIfIdling();
    }
  }

  void ScheduleFlush() REQUIRES(mutex_) {
    if (!highest_backend_computations_) {
      TRACE << "Scheduling flush with no backends in flight!";
      DoFlush();
      return;
    }

    if (idle_timer_target_ <= Clock::now()) {
      DoFlush();
    } else {
      queue_has_pending_flush_ = true;
      TRACE << "Scheduling flush timer for backend at "
            << idle_timer_target_.time_since_epoch();
      flush_timer_.expires_at(idle_timer_target_);
      WaitOnFlushTimer();
    }
  }

  void DoFlush() REQUIRES(mutex_) {
    LCTRACE_FUNCTION_SCOPE;
    queue_has_pending_flush_ = false;
    if (highest_backend_computations_ == max_batches_in_flight_) return;
    auto iter = std::max_element(backend_map_.begin(), backend_map_.end(),
                                 [](const auto& a, const auto& b) {
                                   return a.second.GetPendingSize() <
                                          b.second.GetPendingSize();
                                 });

    if (iter == backend_map_.end()) {
      return;
    }
    highest_backend_computations_ =
        std::max(highest_backend_computations_, iter->second.Flush());
    TRACE << "Flushed backend with highest pending size: " << &iter->second
          << " highest_backend_computations_: "
          << highest_backend_computations_;
  }

  void WaitOnFlushTimer() REQUIRES(mutex_) {
    flush_timer_.async_wait([this](const std::error_code& ec) {
      if (ec) {
        if (ec != asio::error::operation_aborted) {
          CERR << "Flush timer failed: " << ec.message();
        }
        return;
      }
      TRACE << "Flush timer triggered " << Clock::now().time_since_epoch();
      SpinMutex::Lock lock(mutex_);
      PushWorkToBackend();
      DoFlush();
    });
  }

  void StatisticsTimerSetup(Clock::time_point now) {
    if (active_connections_ == 0) {
      return;
    }
    int interval = options_->Get<int>(kStatisticsIntervalOptionId);
    if (interval <= 0) {
      return;
    }
    statistics_timer_.expires_at(now + std::chrono::seconds(interval));
    statistics_timer_.async_wait([this, interval](const std::error_code& ec) {
      if (ec) {
        if (ec != asio::error::operation_aborted) {
          CERR << "Statistics timer failed: " << ec.message();
        }
        return;
      }
      std::vector<std::string> outputs;
      outputs.reserve(backend_map_.size() + 1);
      SpinMutex::Lock lock(mutex_);
      std::ostringstream oss;
      for (auto& backend : backend_map_) {
        auto statistics = backend.second.GetStatistics();
        double nps = static_cast<double>(statistics.positions_) / interval;
        oss << "info string Backend " << backend.first << " nodes "
            << statistics.positions_ << " batches " << statistics.batches_
            << " nps " << nps << " mnps " << statistics.max_nps_ << " queue "
            << statistics.queue_positions_ << "/" << statistics.minibatch_size_
            << " bif " << statistics.batches_in_flight_;
        outputs.emplace_back(oss.str());
        oss.str("");
      }
      oss << "info string Clients " << active_connections_ << " Queue";
      size_t p = 0;
      for (auto& q : priority_queue_) {
        oss << " p" << (p++) << " nodes "
            << std::accumulate(std::begin(q), std::end(q), 0UL,
                               [](size_t sum, const QueueItem& item) {
                                 return sum + (item.last_ - item.first_);
                               })
            << "/" << q.size();
      }
      outputs.emplace_back(oss.str());
      lock.unlock();
      responder_->SendRawResponses(outputs);
      StatisticsTimerSetup(statistics_timer_.expiry());
    });
  }
  BackendMap backend_map_;
  BackendMap::iterator network_discovery_;
  const OptionsDict* options_ = nullptr;
  StdoutUciResponder* responder_ = nullptr;

  mutable SpinMutex mutex_;
  std::condition_variable_any cv_;
  unsigned highest_backend_computations_ = 0;
  unsigned max_batches_in_flight_ = 0;
  bool queue_has_pending_flush_ = false;
  TimePoint idle_timer_target_{};
  std::array<std::deque<QueueItem>, client::kMaxComputationPriority>
      priority_queue_;

  size_t active_connections_ = 0;
  asio::io_context io_context_;
  asio::steady_timer flush_timer_{io_context_};
  asio::steady_timer statistics_timer_{io_context_};
};

class ClientComputation {
 public:
  using CompletionType = std::function<void(ClientComputation&)>;
  static constexpr unsigned kMaxMovesPerPosition = 218;

  ClientComputation(unsigned id, BackendHandler* backend,
                    CompletionType&& completion, unsigned priority)
      : id_(id),
        priority_(priority),
        backend_(backend),
        completion_handler_(std::move(completion)) {
    LCTRACE_FUNCTION_SCOPE;
    auto attrs = backend->GetAttributes();
    maximum_batch_size_ = attrs.maximum_batch_size;
  };

  ClientComputation(const ClientComputation&) = delete;
  ClientComputation& operator=(const ClientComputation&) = delete;

  ~ClientComputation() { LCTRACE_FUNCTION_SCOPE; }

  int ComputeBlocking(std::vector<client::InputPosition>&& inputs) {
    if (inputs.empty()) {
      CERR << "ComputeBlocking called with empty inputs.";
      return -1;
    }
    if (inputs.size() > maximum_batch_size_) {
      CERR << "ComputeBlocking called with too many inputs: " << inputs.size()
           << " maximum: " << maximum_batch_size_;
      return -1;
    }
    inputs_ = std::move(inputs);
    results_.resize(inputs_.size());
    policy_.resize(inputs_.size() * kMaxMovesPerPosition);
    SharedQueue::Get().Enqueue(priority_, backend_, this, 0, inputs_.size());
    return 0;
  }

  std::vector<client::NetworkResult>& GetResults() { return results_; }
  std::span<float> GetPolicy() { return policy_; }

  auto GetId() const { return id_; }
  auto GetInput(size_t index) { return inputs_[index]; }
  EvalResultPtr GetEvalResult(size_t index, size_t legal_moves) {
    size_t policy_offset =
        policy_reserved_.fetch_add(legal_moves, std::memory_order_relaxed);
    assert(policy_offset + legal_moves <= policy_.size());
    results_[index].policy_ =
        std::span<float>(policy_.data() + policy_offset, legal_moves);
    return {&results_[index].value_,
            &results_[index].draw_,
            &results_[index].moves_left_,
            {policy_.data() + policy_offset, legal_moves}};
  }

  void NotifyResultsReady(const QueueItem& item) {
    size_t count = item.last_ - item.first_;
    size_t done = results_ready_.fetch_add(count, std::memory_order_relaxed);
    if (done + count == inputs_.size()) {
      LCTRACE_FUNCTION_SCOPE;
      results_.resize(inputs_.size());
      policy_.resize(policy_reserved_.load(std::memory_order_relaxed));
      completion_handler_(*this);
    }
  }

 private:
  unsigned id_;
  unsigned priority_;
  unsigned maximum_batch_size_;

  BackendHandler* backend_;
  CompletionType completion_handler_;
  std::vector<client::InputPosition> inputs_;
  std::vector<client::NetworkResult> results_;
  std::vector<float> policy_;
  std::atomic<size_t> policy_reserved_{0};
  std::atomic<size_t> results_ready_{0};
};

template <typename SocketType>
class ServerConnection
    : public client::Connection<SocketType>,
      public std::enable_shared_from_this<ServerConnection<SocketType>> {
  using Base = client::Connection<SocketType>;
  using ComputationMapType = std::map<unsigned long, ClientComputation>;

 public:
  ServerConnection(SocketType&& socket)
      : Base(std::forward<SocketType>(socket)) {
    // Initialize connection.
    TRACE << "New client connection.";
    SharedQueue::Get().NewConnection();
  }

  ~ServerConnection() {
    TRACE << "Client connection closed.";
    SharedQueue::Get().ConnectionClosed();
  }

  void Start() { Read(); }

 private:
  void Read() {
    auto self = this->shared_from_this();
    Base::ReadHeader([this, self](auto& message, auto& ar) {
      // Clang warns about unused this if not using this for the call.
      return this->HandleMessage(message, ar);
    });
  }

  template <typename MessageType, typename Archive>
  typename Archive::ResultType HandleMessage(const MessageType& message,
                                             Archive&) {
    // Handle different message types here.
    CERR << "Received unexpected message of type: " << message.header_.type_;
    return Unexpected(client::ArchiveError::UnknownType);
  }

  template <typename Archive>
  typename Archive::ResultType HandleMessage(const client::Handshake& message,
                                             Archive& ar) {
    LCTRACE_FUNCTION_SCOPE;
    assert(message.header_.type_ == client::MessageType::HANDSHAKE);
    if (backend_) {
      CERR << "Received duplicate handshake message.";
      return Unexpected(client::ArchiveError::InvalidData);
    }
    std::string error;
    auto& backends = SharedQueue::Get().GetBackendMap();
    auto iter = message.network_name_ == SharedBackendParams::kAutoDiscover
                    ? SharedQueue::Get().GetDiscovery()
                    : backends.find(message.network_name_);
    if (iter == backends.end()) {
      error = "Requested network not found: ";
      error.append(message.network_name_);
      client::HandshakeReply reply;
      reply.error_message_ = error;

      Base::SendMessage(this->shared_from_this(), reply);
      return Unexpected(client::ArchiveError::InvalidData);
    }
    auto self = this->shared_from_this();

    TRACE << "Received handshake for network: " << iter->first
          << " backend: " << &iter->second;

    iter->second.EnsureLoaded(
        iter->first, [this, self, iter](const std::error_code& ec,
                                        const BackendAttributes& attr) {
          client::HandshakeReply reply;
          if (ec) {
            CERR << "Error loading backend for network: " << iter->first;
            std::string error_message = "Error loading backend for network: ";
            error_message.append(iter->first);
            reply.error_message_ = error_message;
            Base::SendMessage(std::move(self), reply);
            Close();
            return;
          }
          this->Dispatch([this, self, iter]() { backend_ = &iter->second; });
          reply.attributes_ = attr;
          Base::SendMessage(std::move(self), reply);
        });
    this->Defer([self = std::move(self), this] { Read(); });
    return {ar};
  }

  template <typename Archive>
  typename Archive::ResultType HandleMessage(client::ComputeBlocking& message,
                                             Archive& ar) {
    LCTRACE_FUNCTION_SCOPE;
    assert(message.header_.type_ == client::MessageType::COMPUTE_BLOCKING);
    if (!backend_) {
      CERR << "Received ComputeBlocking message before handshake.";
      return Unexpected(client::ArchiveError::InvalidData);
    }
    client::ComputeBlockingReply reply;
    unsigned priority = message.priority_;
    if (priority >= client::kMaxComputationPriority) {
      std::string error_message =
          "Invalid computation priority: " + std::to_string(priority);
      reply.error_message_ = error_message;
      Base::SendMessage(this->shared_from_this(), reply);
      return Unexpected(client::ArchiveError::InvalidData);
    }
    auto self = this->shared_from_this();
    size_t id = message.computation_id_;
    auto iter = computations_.try_emplace(
        id, id, backend_,
        [self = std::move(self), this](ClientComputation& computation) {
          this->Defer([self = std::move(self), this, &computation]() {
            CompleteComputation(computation);
          });
        },
        priority);
    if (!iter.second) {
      std::string error_message =
          "Duplicate computation ID: " + std::to_string(id);
      reply.error_message_ = error_message;
      Base::SendMessage(this->shared_from_this(), reply);
      return Unexpected(client::ArchiveError::InvalidData);
    }

    if (iter.first->second.ComputeBlocking(std::move(message.inputs_))) {
      std::string error_message =
          "ComputeBlocking failed for Computation ID: " +
          std::to_string(message.computation_id_);
      reply.error_message_ = error_message;
      Base::SendMessage(this->shared_from_this(), reply);
      return Unexpected(client::ArchiveError::InvalidData);
    }
    this->Defer([self = this->shared_from_this(), this] { Read(); });
    return {ar};
  }

  void CompleteComputation(ClientComputation& computation) {
    LCTRACE_FUNCTION_SCOPE;
    auto iter = std::find_if(
        computations_.begin(), computations_.end(),
        [&](const auto& pair) { return &pair.second == &computation; });
    assert(iter != computations_.end());
    TRACE << "Computation completed, sending results. " << iter->first;
    client::ComputeBlockingReply message;
    message.computation_id_ = iter->first;
    message.results_ = std::move(computation.GetResults());
    Base::SendMessage(this->shared_from_this(), message);
    computations_.erase(iter);
  }

  void Close() { Base::Close(); }

  BackendHandler* backend_ = nullptr;
  ComputationMapType computations_;
};

template <typename Proto>
class BackendServer {
 public:
  using AcceptorType = typename Proto::acceptor;
  using SocketType = typename Proto::socket;
  using Endpoint = typename Proto::endpoint;

  // Windows bind returns not supported if we request reuse_address on local
  // stream. The library default is to set the option but it won't do anything
  // for local streams.
  // https://stackoverflow.com/questions/68791319
  BackendServer(asio::io_context& ctx, const OptionsDict& params,
                StdoutUciResponder& responder)
      : acceptor_(ctx, GetEndpoint(ctx, params),
                  !std::is_same_v<Proto, asio::local::stream_protocol>),
        params_(const_cast<OptionsDict&>(params)) {
    do_accept();
    std::ostringstream oss;
    oss << "info string Backend server listening on "
        << params.Get<std::string>(kProtocolOptionId) << "://"
        << acceptor_.local_endpoint();
    responder.SendRawResponse(oss.str());
  }

  ~BackendServer() {
    if constexpr (std::is_same_v<Proto, asio::local::stream_protocol>) {
      // Remove the named pipe file.
      std::filesystem::remove(params_.Get<std::string>(kPipeNameOptionId));
    }
  }

 private:
  static Endpoint GetEndpoint(asio::io_context& ctx,
                              const OptionsDict& params) {
    if constexpr (std::is_same_v<Proto, asio::local::stream_protocol>) {
      std::string pipe_name = params.Get<std::string>(kPipeNameOptionId);
      std::filesystem::remove(pipe_name);
      return client::GetEndpoint<Endpoint>(pipe_name);
    } else {
      std::string host = params.Get<std::string>(kHostOptionId);
      std::string port = std::to_string(params.Get<int>(kPortOptionId));
      return client::GetEndpoint<Endpoint>(ctx, host, port);
    }
  }

  void do_accept() {
    LCTRACE_FUNCTION_SCOPE;
    acceptor_.async_accept([this](std::error_code ec, SocketType socket) {
      if (ec) {
        CERR << "Accept error: " << ec.message();
        return;
      }
      std::make_shared<ServerConnection<SocketType>>(std::move(socket))
          ->Start();
      if (params_.Get<int>(kAcceptLimitOptionId) == 0 ||
          ++accepted_connections_ < static_cast<unsigned long>(params_.Get<int>(
                                        kAcceptLimitOptionId))) {
        do_accept();
      } else {
        acceptor_.close();
      }
    });
  }
  AcceptorType acceptor_;
  OptionsDict& params_;
  unsigned long accepted_connections_ = 0;
};

template <typename Callback>
void BackendHandler::EnsureLoaded(const std::string& net, Callback&& callback) {
  SpinMutex::Lock lock(mutex_);
  if (backend_) {
    // Already loaded.
    lock.unlock();
    std::error_code ec{};
    callback(ec, GetAttributes());
    return;
  }

  pending_callbacks_.emplace_back(std::forward<Callback>(callback));

  if (backend_threads_.empty()) {
    backend_threads_.emplace_back([this, net] {
      try {
        const std::string name =
            params_.Get<std::string>(SharedBackendParams::kBackendId);
        auto factory = BackendManager::Get()->GetFactoryByName(name);
        auto backend = factory->Create(params_, net);
        {
          LCTRACE_FUNCTION_SCOPE;
          SpinMutex::Lock lock(mutex_);
          backend_ = std::move(backend);
          std::error_code ec{};
          BackendAttributes attrs = GetAttributes();
          minibatch_size_ = attrs.recommended_batch_size;
          size_t threads =
              attrs.suggested_num_search_threads + !attrs.runs_on_cpu;
          while (backend_threads_.size() < threads) {
            backend_threads_.emplace_back([this] { Worker(); });
          }
          for (auto& cb : pending_callbacks_) {
            cb(ec, attrs);
          }
          pending_callbacks_.clear();
        }
        SharedQueue::Get().BackendThreads(backend_threads_.size());
        Worker();
      } catch (const Exception& ex) {
        CERR << "Error loading backend: " << ex.what();
        SpinMutex::Lock lock(mutex_);
        auto err = std::make_error_code(std::errc::function_not_supported);
        for (auto& cb : pending_callbacks_) {
          cb(err, BackendAttributes{});
        }
        pending_callbacks_.clear();
      }
    });
  }
}

void BackendHandler::Worker() {
  assert(backend_);
  try {
    while (true) {
      size_t size = 0;
      Clock::duration increment_flush_timer;
      bool update_flush_timer = false;
      std::vector<QueueItem> batch;
      {
        SpinMutex::Lock lock(mutex_);
        cv_.wait(lock, [this] { return queued_computations_ || exit_; });
        if (exit_) {
          TRACE << this << " Backend worker exiting.";
          return;
        }
        while (size < minibatch_size_ && !queue_.empty()) {
          auto& item = queue_.front();
          size += item.last_ - item.first_;
          batch.emplace_back(item);
          queue_.pop();
        }
        update_flush_timer = StartBatch(size) < backend_threads_.size();
        if (update_flush_timer) {
          increment_flush_timer = std::chrono::duration_cast<Clock::duration>(
              std::chrono::duration<double>(kGpuIdleBufferMultiplier * size /
                                            max_nps_));
        }
      }
      LCTRACE_FUNCTION_SCOPE;
      if (update_flush_timer) {
        SharedQueue::Get().AddToFlushTimer(increment_flush_timer);
      }
      TRACE << this << " Backend worker picked batch of size: " << size;
      auto computation = backend_->CreateComputation(0);
      PositionHistory history;
      history.Reserve(kMoveHistory);
      for (const auto& item : batch) {
        TRACE << this << " Backend adding inputs from " << item.first_ << " to "
              << item.last_;
        for (size_t i = item.first_; i < item.last_; ++i) {
          auto position = item.computation_->GetInput(i);
          history.Reset(position.base_);
          for (size_t i = 0; i < position.history_length_; ++i) {
            history.Append(position.history_[i]);
          }
          auto pos = history.GetPositions();
          auto legal_moves = pos.back().GetBoard().GenerateLegalMoves();
          auto results =
              item.computation_->GetEvalResult(i, legal_moves.size());
          TRACE << this << " Adding input " << item.computation_->GetId() << "["
                << i << "] legal moves: " << legal_moves.size() << " fen "
                << pos.back().DebugString();
          computation->AddInput({pos, legal_moves}, results);
        }
      }
      if (size != 0) computation->ComputeBlocking();
      TRACE << this << " Backend worker completed batch of size: " << size;
      TimePoint now = Clock::now();
      auto [queued_batches, idle] = CompleteBatch(size, now);
      SharedQueue::Get().ComputationDone(queued_batches, idle);
      for (auto& item : batch) {
        item.computation_->NotifyResultsReady(item);
      }
    }
  } catch (const Exception& ex) {
    CERR << "Backend worker loop exited with error: " << ex.what();
  }
}

void SharedQueue::StartServer() {
  if (options_->Get<std::string>(kProtocolOptionId) == "unix") {
    BackendServer<asio::local::stream_protocol> server(io_context_, *options_,
                                                       *responder_);
    io_context_.run();
  } else if (options_->Get<std::string>(kProtocolOptionId) == "tcp") {
    BackendServer<asio::ip::tcp> server(io_context_, *options_, *responder_);
    io_context_.run();
  } else {
    CERR << "Unknown protocol: "
         << options_->Get<std::string>(kProtocolOptionId);
  }
}

class BackendserverEngine : public EngineControllerBase {
 public:
  BackendserverEngine() = default;
  ~BackendserverEngine() override = default;

  void EnsureReady() override { SharedQueue::Get().EnsureReady(); }
  void NewGame() override {}
  void SetPosition(const std::string&,
                   const std::vector<std::string>&) override {}
  void Go(const GoParams&) override {}
  void PonderHit() override {}
  void Stop() override {}
  void Wait() override {}

  void RegisterUciResponder(UciResponder*) override {}
  void UnregisterUciResponder(UciResponder*) override {}
};

class ConsoleThread : public std::thread {
 public:
  using std::thread::thread;
  ~ConsoleThread() {
#ifdef _WIN32
    _close(_fileno(stdin));
    GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0);
#else
    close(STDIN_FILENO);
    pthread_kill(native_handle(), SIGINT);
#endif
    if (joinable()) {
      join();
    }
  }
};

}  // namespace

void RunBackendServer() {
  OptionsParser options_parser;
  options_parser.Add<StringOption>(kLogFileId);
  ConfigFile::PopulateOptions(&options_parser);
  SharedBackendParams::Populate(&options_parser);
  options_parser.Add<IntOption>(kMinibatchSizeOptionId, 0, 1024) = 0;
  options_parser.Add<StringOption>(kProtocolOptionId) =
      client::kDefaultProtocol;
  options_parser.Add<StringOption>(kPipeNameOptionId) =
      client::kDefaultPipeName;
  options_parser.Add<StringOption>(kHostOptionId) = client::kDefaultHost;
  options_parser.Add<IntOption>(kPortOptionId, 1, 65535) = client::kDefaultPort;
  options_parser.Add<StringOption>(kNetworkDirectoryOptionId) =
      kDefaultNetworkDirectory;
  options_parser.Add<IntOption>(kAcceptLimitOptionId, 0, 1024) = 0;
  options_parser.Add<IntOption>(kStatisticsIntervalOptionId, 0, 3600) = 60;
  if (!ConfigFile::Init() || !options_parser.ProcessAllFlags()) return;
  auto options = options_parser.GetOptionsDict();
  Logging::Get().SetFilename(options.Get<std::string>(kLogFileId));
  try {
    CERR << "Using network directory: "
         << options.Get<std::string>(kNetworkDirectoryOptionId);
    StdoutUciResponder uci_responder;
    BackendMap& backends =
        SharedQueue::Get().GetBackendMap(options, uci_responder);
    absl::Cleanup cleanup = [] { SharedQueue::Get().Close(); };
    std::filesystem::path network_dir(
        options.Get<std::string>(kNetworkDirectoryOptionId));
    if (!std::filesystem::exists(network_dir)) {
      CERR << "Network directory does not exist: " << network_dir;
      return;
    }
    if (!std::filesystem::is_directory(network_dir)) {
      CERR << "Network directory is not a directory: " << network_dir;
      return;
    }

    std::filesystem::directory_entry newest;
    bool first = true;

    for (const auto& entry : std::filesystem::directory_iterator(network_dir)) {
      if (IsPathWeightsFile(entry)) {
        if (first || entry.last_write_time() > newest.last_write_time()) {
          first = false;
          newest = entry;
        }
        CERR << entry.path().filename();
        backends.try_emplace(entry.path().filename().string(), options);
      }
    }

    SharedQueue::Get().SetDiscovery(newest.path());

    ConsoleThread console([&] {
      absl::Cleanup cleanup = [] { SharedQueue::Get().Stop(); };
      std::cout.setf(std::ios::unitbuf);
      std::string line;
      BackendserverEngine engine{};
      UciLoop loop(&uci_responder, &options_parser, &engine);
      while (std::getline(std::cin, line)) {
        LOGFILE << ">> " << line;
        try {
          if (!loop.ProcessLine(line)) break;

          Logging::Get().SetFilename(options.Get<std::string>(kLogFileId));
        } catch (const Exception& ex) {
          uci_responder.SendRawResponse(std::string("error ") + ex.what());
        }
      }
    });

    SharedQueue::Get().StartServer();
  } catch (Exception& ex) {
    CERR << "Error: " << ex.what();
  }
}
}  // namespace lczero

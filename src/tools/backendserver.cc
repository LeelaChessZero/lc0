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

// clang-format off
#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#endif
// clang-format on

#include <absl/cleanup/cleanup.h>

#include <asio.hpp>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <map>
#include <memory>
#include <regex>
#include <utility>

#include "neural/backends/client/proto.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/optionsparser.h"
#include "utils/trace.h"

namespace lczero {
namespace {

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
const OptionId kHostOptionId{"tcp-host", "TCPHost", "Host to listen on for TCP."};
const OptionId kPortOptionId{"tcp-port", "TCPPort", "Port to listen on for TCP."};
const OptionId kNetworkDirectoryOptionId{
    "network-directory", "NetworkDirectory",
    "Directory where neural network files are stored."};
const OptionId kAcceptLimitOptionId{
    "accept-limit", "AcceptLimit",
    "Maximum number of accepted client connections."};

const std::string kDefaultNetworkDirectory = ".";

class BackendHandler;
class ClientComputation;

struct QueueItem {
  BackendHandler* backend_ = nullptr;
  ClientComputation* computation_ = nullptr;
  size_t first_ = 0;
  size_t last_ = 0;
};

class BackendHandler {
 public:
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
    batch_size_ = attrs.recommended_batch_size;
    // Half size minibatches are the smallest batch which is still fast.
    attrs.recommended_batch_size /= 2;
    return attrs;
  }

  std::tuple<bool, unsigned> AddBatchToQueue(QueueItem& item, size_t& count) {
    LCTRACE_FUNCTION_SCOPE;
    SpinMutex::Lock lock(mutex_);
    assert(backend_);
    assert(computations_in_flight_ < backend_threads_.size());
    queue_.emplace(item);
    bool flushed = false;
    auto pending_size = PendingSize();
    if (count + pending_size >= batch_size_) {
      count = batch_size_ - pending_size;
      computations_in_flight_++;
      queue_.back().last_ = queue_.back().first_ + count;
      item.first_ += count;
      cv_.notify_one();
      flushed = true;
    }
    queue_size_ += count;
    TRACE << this << " Backend queue size: " << queue_size_
          << " pending size: " << pending_size
          << " computations in flight: " << computations_in_flight_
          << " flushed: " << flushed;
    return {flushed, computations_in_flight_};
  }

  unsigned Flush() {
    LCTRACE_FUNCTION_SCOPE;
    // TODO: Do a delayed flush based on predicted end of an active evaluation.
    SpinMutex::Lock lock(mutex_);
    if (PendingSize() > 0 &&
        computations_in_flight_ < backend_threads_.size()) {
      computations_in_flight_++;
      queue_size_ = computations_in_flight_ * batch_size_;
      TRACE << this << " Forcing backend flush, new queue size: " << queue_size_
            << " pending size: " << PendingSize()
            << " computations in flight: " << computations_in_flight_;
      cv_.notify_one();
    }
    return computations_in_flight_;
  }

  size_t GetPendingSize() const {
    SpinMutex::Lock lock(mutex_);
    return PendingSize();
  }

 private:
  size_t PendingSize() const {
    size_t flushed_sizes = computations_in_flight_ * batch_size_;
    size_t pending_size = queue_size_ - flushed_sizes;
    return pending_size;
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

  unsigned batch_size_ = 0;
  unsigned queue_size_ = 0;
  unsigned computations_in_flight_ = 0;
  bool exit_ = false;

  const OptionsDict& params_;
};

class ClientComputation;

using BackendMap = std::map<std::string, BackendHandler, std::less<void>>;

class SharedQueue {
 public:
  static SharedQueue& Get() {
    static SharedQueue instance;
    return instance;
  }

  void NewComputation() {
    SpinMutex::Lock lock(mutex_);
    active_computations_++;
  }

  void ComputeBlocking() {
    SpinMutex::Lock lock(mutex_);
    active_computations_--;
    if (active_computations_ == 0) {
      PushWorkToBackend();
    }
  }

  void Enqueue(unsigned priority, BackendHandler* backend,
               ClientComputation* computation, size_t first, size_t last) {
    // TODO: Use priority for preferred fair scheduling.
    assert(priority < client::kMaxComputationPriority);
    SpinMutex::Lock lock(mutex_);
    queue_.emplace(backend, computation, first, last);
    if (!backend_evaluations_) {
      PushWorkToBackend();
    }
  }

  BackendMap& GetBackendMap() { return backend_map_; }

  void BackendThreads(size_t count) {
    SpinMutex::Lock lock(mutex_);
    max_batches_in_flight_ = count;
  }

  void ComputationDone(unsigned computations_in_flight) {
    LCTRACE_FUNCTION_SCOPE;
    SpinMutex::Lock lock(mutex_);
    if (backend_evaluations_ == computations_in_flight) {
      backend_evaluations_--;
      if (!queue_.empty()) {
        PushWorkToBackend();
      }
    }
  }

  void Close() {
    LCTRACE_FUNCTION_SCOPE;
    SpinMutex::Lock lock(mutex_);
    assert(active_computations_ == 0);
    assert(queue_.empty());
    backend_map_.clear();
  }

 private:
  SharedQueue() = default;

  SharedQueue(const SharedQueue&) = delete;
  SharedQueue& operator=(const SharedQueue&) = delete;

  void PushWorkToBackend() {
    LCTRACE_FUNCTION_SCOPE;
    TRACE << "Pushing work to backend: " << queue_.size() << " items in queue.";
    bool needs_flush = true;
    while (!queue_.empty() && backend_evaluations_ != max_batches_in_flight_) {
      auto& item = queue_.front();
      size_t batch_size = item.last_ - item.first_;
      auto [flushed, batches] =
          item.backend_->AddBatchToQueue(item, batch_size);
      if (flushed) {
        needs_flush = false;
        if (item.first_ == item.last_) {
          queue_.pop();
        }
        backend_evaluations_ = std::max(backend_evaluations_, batches);
      } else {
        queue_.pop();
      }
    }
    if (needs_flush) {
      auto iter = std::max_element(backend_map_.begin(), backend_map_.end(),
                                   [](const auto& a, const auto& b) {
                                     return a.second.GetPendingSize() <
                                            b.second.GetPendingSize();
                                   });
      if (iter != backend_map_.end()) {
        TRACE << "Forcing flush to backend: " << iter->first;
        auto batches = iter->second.Flush();
        backend_evaluations_ = std::max(backend_evaluations_, batches);
      }
    }
  }

  BackendMap backend_map_;

  SpinMutex mutex_;
  std::condition_variable_any cv_;
  unsigned active_computations_ = 0;
  unsigned backend_evaluations_ = 0;
  unsigned max_batches_in_flight_ = 0;
  std::queue<QueueItem> queue_;
};

class ClientComputation {
 public:
  using CompletionType = std::function<void(ClientComputation&)>;
  static constexpr unsigned kMaxMovesPerPosition = 218;

  ClientComputation(BackendHandler* backend, CompletionType&& completion,
                    unsigned priority)
      : priority_(priority),
        backend_(backend),
        completetion_(std::move(completion)) {
    LCTRACE_FUNCTION_SCOPE;
    SharedQueue::Get().NewComputation();
    auto attrs = backend->GetAttributes();
    inputs_.reserve(attrs.maximum_batch_size);
    results_.resize(attrs.maximum_batch_size);
    policy_.resize(attrs.maximum_batch_size * kMaxMovesPerPosition);
    maximum_batch_size_ = attrs.maximum_batch_size;
  };

  ClientComputation(const ClientComputation&) = delete;
  ClientComputation& operator=(const ClientComputation&) = delete;

  ~ClientComputation() {
    LCTRACE_FUNCTION_SCOPE;
    if (!computed_) {
      SharedQueue::Get().ComputeBlocking();
    }
  }

  int ComputeBlocking(const std::vector<std::span<const Position>>& inputs) {
    if (inputs.empty()) {
      CERR << "ComputeBlocking called with empty inputs.";
      return -1;
    }
    if (inputs.size() > maximum_batch_size_) {
      CERR << "ComputeBlocking called with too many inputs: " << inputs.size()
           << " maximum: " << maximum_batch_size_;
      return -1;
    }
    for (const auto& input : inputs) {
      inputs_.emplace_back(input.begin(), input.end());
    }
    SharedQueue::Get().Enqueue(priority_, backend_, this, 0, inputs_.size());
    computed_ = true;
    SharedQueue::Get().ComputeBlocking();
    return 0;
  }

  std::vector<client::NetworkResult>& GetResults() { return results_; }
  std::span<float> GetPolicy() { return policy_; }

  auto GetInput(size_t index) { return inputs_[index]; }
  EvalResultPtr GetEvalResult(size_t index, size_t legal_moves) {
    size_t policy_offset =
        policy_reserved_.fetch_add(legal_moves, std::memory_order_relaxed);
    assert(policy_offset + legal_moves <= policy_.size());
    TRACE << "Allocating eval result for index " << index
          << " legal moves: " << legal_moves
          << " policy offset: " << policy_offset;
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
      completetion_(*this);
    }
  }

 private:
  unsigned priority_;
  unsigned maximum_batch_size_;
  bool computed_ = false;

  BackendHandler* backend_;
  CompletionType completetion_;
  std::vector<std::vector<Position>> inputs_;
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
  }

  ~ServerConnection() { TRACE << "Client connection closed."; }

  void Start() { Read(); }

 private:
  void Read() {
    auto self = this->shared_from_this();
    Base::ReadHeader([this, self](const auto& message) {
      // Clang warns about unused this if not using this for the call.
      return this->HandleMessage(message);
    });
  }

  template <typename MessageType>
  int HandleMessage(const MessageType& message) {
    // Handle different message types here.
    CERR << "Received unexpected message of type: " << message.header_.type_;
    return -1;
  }

  int HandleMessage(const client::Handshake& message) {
    LCTRACE_FUNCTION_SCOPE;
    assert(message.header_.type_ == client::MessageType::HANDSHAKE);
    if (backend_) {
      CERR << "Received duplicate handshake message.";
      return -1;
    }
    std::string error;
    auto& backends = SharedQueue::Get().GetBackendMap();
    auto iter = backends.find(message.network_name_);
    if (iter == backends.end()) {
      error = "Requested network not found: ";
      error.append(message.network_name_);
      client::HandshakeReply reply;
      reply.error_message_ = error;

      Base::SendMessage(this->shared_from_this(), reply);
      return -1;
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
    return 0;
  }

  int HandleMessage([[maybe_unused]] const client::ComputeBlocking& message) {
    LCTRACE_FUNCTION_SCOPE;
    assert(message.header_.type_ == client::MessageType::COMPUTE_BLOCKING);
    if (!backend_) {
      CERR << "Received ComputeBlocking message before handshake.";
      return -1;
    }
    client::ComputeBlockingReply reply;
    auto self = this->shared_from_this();
    size_t id = message.computation_id_;
    unsigned priority = 0;
    auto iter = computations_.try_emplace(
        id, backend_,
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
      return -1;
    }

    if (iter.first->second.ComputeBlocking(message.inputs_)) {
      std::string error_message =
          "ComputeBlocking failed for Computation ID: " +
          std::to_string(message.computation_id_);
      reply.error_message_ = error_message;
      Base::SendMessage(this->shared_from_this(), reply);
      return -1;
    }
    this->Defer([self = this->shared_from_this(), this] { Read(); });
    return 0;
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
  BackendServer(asio::io_context& ctx, const OptionsDict& params)
      : acceptor_(ctx, GetEndpoint(ctx, params)),
        params_(const_cast<OptionsDict&>(params)) {
    do_accept();
    COUT << "info string Backend server listening on " << acceptor_.local_endpoint();
  }

  ~BackendServer() {
    std::filesystem::remove(params_.Get<std::string>(kPipeNameOptionId));
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
      std::vector<QueueItem> batch;
      {
        SpinMutex::Lock lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || exit_; });
        if (exit_) {
          TRACE << this << " Backend worker exiting.";
          return;
        }
        size_t size = 0;
        while (size < batch_size_ && !queue_.empty()) {
          auto& item = queue_.front();
          size += item.last_ - item.first_;
          batch.emplace_back(item);
          queue_.pop();
        }
        TRACE << this << " Backend worker picked batch of size: " << size;
        if (batch.empty()) {
          continue;
        }
      }
      LCTRACE_FUNCTION_SCOPE;
      auto computation = backend_->CreateComputation();
      for (const auto& item : batch) {
        TRACE << this << " Backend adding inputs from " << item.first_ << " to "
              << item.last_;
        for (size_t i = item.first_; i < item.last_; ++i) {
          auto position = item.computation_->GetInput(i);
          auto legal_moves = position.back().GetBoard().GenerateLegalMoves();
          auto results =
              item.computation_->GetEvalResult(i, legal_moves.size());
          computation->AddInput({position, legal_moves}, results);
        }
      }
      computation->ComputeBlocking();
      unsigned cif;
      {
        SpinMutex::Lock lock(mutex_);
        queue_size_ -= batch_size_;
        cif = computations_in_flight_--;
      }
      SharedQueue::Get().ComputationDone(cif);
      for (auto& item : batch) {
        item.computation_->NotifyResultsReady(item);
      }
    }
  } catch (const Exception& ex) {
    CERR << "Backend worker loop exited with error: " << ex.what();
  }
}

}  // namespace

void RunBackendServer() {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  options.Add<IntOption>(kMinibatchSizeOptionId, 0, 1024) = 0;
  options.Add<StringOption>(kProtocolOptionId) = client::kDefaultProtocol;
  options.Add<StringOption>(kPipeNameOptionId) = client::kDefaultPipeName;
  options.Add<StringOption>(kHostOptionId) = client::kDefaultHost;
  options.Add<IntOption>(kPortOptionId, 1, 65535) = client::kDefaultPort;
  options.Add<StringOption>(kNetworkDirectoryOptionId) =
      kDefaultNetworkDirectory;
  options.Add<IntOption>(kAcceptLimitOptionId, 0, 1024) = 0;
  if (!options.ProcessAllFlags()) return;
  try {
    auto option_dict = options.GetOptionsDict();

    CERR << "Using network directory: "
         << option_dict.Get<std::string>(kNetworkDirectoryOptionId);
    BackendMap& backends = SharedQueue::Get().GetBackendMap();
    absl::Cleanup cleanup = [] { SharedQueue::Get().Close(); };
    std::filesystem::path network_dir(
        option_dict.Get<std::string>(kNetworkDirectoryOptionId));
    if (!std::filesystem::exists(network_dir)) {
      CERR << "Network directory does not exist: " << network_dir;
      return;
    }
    if (!std::filesystem::is_directory(network_dir)) {
      CERR << "Network directory is not a directory: " << network_dir;
      return;
    }
    std::regex kNetworkFileRegex(R"(\.pb(\.gz)?$)");

    for (const auto& entry : std::filesystem::directory_iterator(network_dir)) {
      if (entry.is_regular_file() &&
          std::regex_search(entry.path().filename().string(),
                            kNetworkFileRegex)) {
        CERR << entry.path().filename();
        backends.try_emplace(entry.path().filename().string(), option_dict);
      }
    }

    asio::io_context io_context;
    if (option_dict.Get<std::string>(kProtocolOptionId) == "unix") {
      BackendServer<asio::local::stream_protocol> server(io_context,
                                                         option_dict);
      io_context.run();
      return;
    } else if (option_dict.Get<std::string>(kProtocolOptionId) == "tcp") {
      BackendServer<asio::ip::tcp> server(io_context, option_dict);
      io_context.run();
      return;
    } else {
      CERR << "Unknown protocol: "
           << option_dict.Get<std::string>(kProtocolOptionId);
      return;
    }

  } catch (Exception& ex) {
    CERR << ex.what();
  }
}
}  // namespace lczero

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

#include <stdio.h>

#include <atomic>
#include <list>

#include "neural/backend.h"
#include "neural/backends/client/proto.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/asio.h"
#include "utils/atomic.h"
#include "utils/atomic_vector.h"
#include "utils/commandline.h"
#include "utils/trace.h"

// clang-format off
#ifdef _WIN32
#include <io.h>
#define popen _popen
#define pclose _pclose
#define close _close
#endif
// clang-format on

namespace lczero::client {

namespace {

// Parent class to hold asio context because it must be allocated before socket
// inside Connection.
class Context {
 public:
  Context() {}
  ~Context() {}
  asio::io_context& io_context() { return io_context_; }

 private:
  asio::io_context io_context_;
};

template <typename Proto>
class BackendClientComputation;

// ClientConnection manages a single connection to a backend server. It uses
// synchronous networing. Each search thread must have its own ClientConnection.
// Backend will allocate connections for computations. It caches allocated
// connections for reuse.
template <typename Proto>
class ClientConnection final : public Context,
                               public Connection<typename Proto::socket> {
  using Base = client::Connection<typename Proto::socket>;
  using ComputationMap = std::map<size_t, BackendClientComputation<Proto>*>;
  using SocketType = typename Proto::socket;
  using Endpoint = typename Proto::endpoint;

 public:
  // Connect to backend server based on backend options. If user gives
  // server-arguments option, it will start a new server process after a
  // failing to connect to an existing server.
  ClientConnection(const OptionsDict& options, const std::string& network)
      : Context(), Base(SocketType{this->io_context()}) {
    // Initialize connection.
    LCTRACE_FUNCTION_SCOPE;
    std::string user_arguments =
        options.GetOrDefault("server-arguments", std::string());
    std::string args = " backendserver ";
    if constexpr (std::is_same_v<Proto, asio::local::stream_protocol>) {
      const std::string pipe =
          options.GetOrDefault("pipe_name", kDefaultPipeName);
      endpoint_ = GetEndpoint<Endpoint>(pipe);
      args += "--protocol=unix --pipe-name=" + pipe;
    } else {
      const std::string host = options.GetOrDefault("tcp-host", kDefaultHost);
      const int port = options.GetOrDefault("tcp-port", kDefaultPort);
      const std::string port_str = std::to_string(port);
      endpoint_ = GetEndpoint<Endpoint>(this->io_context(), host, port_str);
      args += "--protocol=tcp --tcp-host=" + host + " --tcp-port=" + port_str;
    }
    try {
      this->Connect(endpoint_);
    } catch (const std::exception& e) {
      if (user_arguments.empty()) {
        CERR << "Failed to connect to backend server at " << endpoint_ << ": "
             << e.what();
        throw Exception("Failed to connect to backend server");
      }
      std::string command =
          CommandLine::BinaryName() + args + " " + user_arguments;
      FILE* pipe_ = popen(command.c_str(), "r");
      if (!pipe_) {
        CERR << "Failed to start backend server with command: " << command;
        throw Exception("Failed to start backend server");
      }
      const std::string ready_message(
          "info string Backend server listening on ");
      std::string buffer;
      buffer.resize(512);

      // Wait until server has initialized.
      while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe_)) {
        if (buffer.find(ready_message) != std::string::npos) {
          break;
        }
      }

      // Close the pipe to avoid need to read from it further. Keep the server
      // process running.
      close(fileno(pipe_));

      // Connect to the newly started server.
      this->Connect(endpoint_);
    }
    Start(network);
    CERR << "Connected to backend server " << endpoint_;
  }

  // Create a secondary connection to the same backend server as primary.
  ClientConnection(const ClientConnection& primary, const std::string& network)
      : Context(),
        Base(SocketType{this->io_context()}),
        endpoint_(primary.endpoint_) {
    // Initialize connection.
    LCTRACE_FUNCTION_SCOPE;
    this->Connect(endpoint_);
    Start(network);
    CERR << "Connected to backend server " << endpoint_;
  }

  ~ClientConnection() {
    this->io_context().stop();
    Base::Close();
    if (pipe_) {
      pclose(pipe_);
    }
  }

  // Close the connection if there is any errors.
  void Close() override {
    Base::Close();
    // TODO: Try reconnecting to a restarted server.
    throw Exception("Backend client connection closed");
  }

  // Start a new connection. A client sends a handshake message first.
  void Start(const std::string& network) { WriteHandshake(network); }

  // A fake self type for synchronous send/receive. Asynchronous operations
  // require a shared pointer to keep the object alive during the operation.
  // Synchronus operations won't access the object after a call returns.
  struct FakeSelf {};

  // Get backend attributes after the server responds to a handshake.
  BackendAttributes GetAttributes() const { return attrs_; }

  // Send a compute blocking request to the server and wait for reply.
  void ComputeBlocking(size_t computation_id, size_t priority,
                       std::vector<InputPosition>& inputs) {
    client::ComputeBlocking message;
    message.computation_id_ = computation_id;
    message.priority_ = priority;
    message.inputs_ = std::move(inputs);
    Base::template SendMessage<false>(Self(), message);
    Read();
  }

  // Reserve this connection for a computation. Returns true if successful.
  bool Reserve(BackendClientComputation<Proto>* computation) {
    BackendClientComputation<Proto>* old = nullptr;
    return reserved_.compare_exchange_strong(
        old, computation, std::memory_order_acquire, std::memory_order_relaxed);
  }

  // Release the connection from a computation.
  void Release() { reserved_.store(nullptr, std::memory_order_release); }

 private:
  // Get the owning computation for this connection.
  BackendClientComputation<Proto>* GetReservedComputation() {
    return reserved_.load(std::memory_order_relaxed);
  }

  // Get a fake self object for synchronous operations.
  FakeSelf Self() { return {}; }

  // Read a message from the server and dispatch to appropriate handler.
  void Read() {
    Base::template ReadHeader<false>(
        [this](const auto& message, auto& archive) {
          // Clang warns about unused this if not using this for the call.
          return this->HandleMessage(message, archive);
        });
  }

  // Send a handshake message to the server and wait for reply.
  void WriteHandshake(const std::string& network) {
    Handshake message;
    message.network_name_ = network;

    Base::template SendMessage<false>(Self(), message);
    Read();
    assert(!Base::IsOpen() || attrs_.maximum_batch_size);
  }

  // Default handler for unexpected message types.
  template <typename MessageType, typename Archive>
  typename Archive::ResultType HandleMessage(const MessageType& message,
                                             Archive&) {
    // Handle different message types here.
    CERR << "Received unexpected message of type: " << message.header_.type_;
    return Unexpected(ArchiveError::UnknownType);
  }

  // Handler for handshake reply message.
  template <typename Archive>
  typename Archive::ResultType HandleMessage(const HandshakeReply& message,
                                             Archive& ar) {
    LCTRACE_FUNCTION_SCOPE;
    assert(message.header_.type_ == MessageType::HANDSHAKE_REPLY);
    if (!message.error_message_.empty()) {
      CERR << "Handshake error: " << message.error_message_;
      return Unexpected(ArchiveError::RemoteError);
    }
    if ((message.attributes_.has_mlh & 1) != message.attributes_.has_mlh) {
      CERR << "has_mlh support has unexpected value: " << std::hex
           << static_cast<unsigned>(message.attributes_.has_mlh);
      return Unexpected(ArchiveError::InvalidData);
    }
    if ((message.attributes_.has_wdl & 1) != message.attributes_.has_wdl) {
      CERR << "has_wdl support has unexpected value: " << std::hex
           << static_cast<unsigned>(message.attributes_.has_wdl);
      return Unexpected(ArchiveError::InvalidData);
    }
    if ((message.attributes_.runs_on_cpu & 1) !=
        message.attributes_.runs_on_cpu) {
      CERR << "runs_on_cpu has unexpected value: " << std::hex
           << static_cast<unsigned>(message.attributes_.runs_on_cpu);
      return Unexpected(ArchiveError::InvalidData);
    }
    if (message.attributes_.maximum_batch_size >
            static_cast<int>(kMaxMinibatchSizes) ||
        message.attributes_.maximum_batch_size <= 0) {
      CERR << "maximum_batch_size outside accepted range: "
           << message.attributes_.maximum_batch_size;
      return Unexpected(ArchiveError::InvalidData);
    }
    if (message.attributes_.recommended_batch_size >
            message.attributes_.maximum_batch_size ||
        message.attributes_.recommended_batch_size <= 0) {
      CERR << "recommended_batch_size outside 0 < "
           << message.attributes_.recommended_batch_size
           << " <= " << message.attributes_.maximum_batch_size;
      return Unexpected(ArchiveError::InvalidData);
    }
    if (message.attributes_.suggested_num_search_threads >
            static_cast<int>(kMaxSearchThreads) ||
        message.attributes_.suggested_num_search_threads <= 0) {
      CERR << "suggested_num_search_threads outside accepted range: "
           << message.attributes_.suggested_num_search_threads;
      return Unexpected(ArchiveError::InvalidData);
    }
    attrs_ = message.attributes_;
    // Handshake complete, ready to proceed.
    return {ar};
  }

  // Handler for compute blocking reply message.
  template <typename Archive>
  typename Archive::ResultType HandleMessage(
      const ComputeBlockingReply& message, Archive& ar);

  FILE* pipe_ = nullptr;
  size_t computation_id_ = -1;
  std::vector<char> pipe_input_buffer_;
  BackendAttributes attrs_{};
  std::atomic<BackendClientComputation<Proto>*> reserved_{nullptr};
  Endpoint endpoint_;
};

// BakendClient forwards evaluation requests to a backend server via a TCP or
// UNIX socket connection.
template <typename Proto>
class BackendClient final : public Backend {
 public:
  BackendClient(const std::string& network, const OptionsDict& options)
      : network_(network) {
    LCTRACE_FUNCTION_SCOPE;
    connections_.emplace_back(options, network_);
    size_t fixed_priority = options.GetOrDefault("fixed-priority", -1);

    if (fixed_priority >= kMaxComputationPriority &&
        fixed_priority != (size_t)-1) {
      CERR << "fixed-priority option " << fixed_priority
           << " is out of range, must be less than " << kMaxComputationPriority;
      fixed_priority = -1;
    }
    fixed_priority_ = fixed_priority;
  }

  BackendAttributes GetAttributes() const override {
    return connections_.front().GetAttributes();
  }
  std::unique_ptr<BackendComputation> CreateComputation(
      size_t time_remaining) override;

  // A smart pointer providing RAII locking for a ClientConnection.
  struct ClientConnectionReference {
    ClientConnectionReference(ClientConnection<Proto>& connection,
                              BackendClient<Proto>& backend)
        : connection_(connection), backend_{backend} {}

    ~ClientConnectionReference() {
      SpinMutex::Lock lock(backend_.mutex_);
      connection_.Release();
    }

    ClientConnection<Proto>* operator->() { return &connection_; }

    ClientConnection<Proto>& connection_;
    BackendClient<Proto>& backend_;
  };

  // Reserve a connection for a computation. If there is no free connections, it
  // creates a new connection.
  ClientConnectionReference GetConnection(
      BackendClientComputation<Proto>* computation) {
    SpinMutex::Lock lock(mutex_);
    for (auto& conn : connections_) {
      if (conn.Reserve(computation)) {
        return {conn, *this};
      }
    }
    connections_.emplace_back(connections_.front(), network_);
    connections_.back().Reserve(computation);
    return {connections_.back(), *this};
  }

  size_t GetFixedPriority() const { return fixed_priority_; }

 private:
  mutable SpinMutex mutex_;
  const std::string network_;
  OptionsDict options_;
  std::atomic<size_t> next_computation_id_ = 0;
  std::list<ClientConnection<Proto>> connections_;
  size_t fixed_priority_ = -1;
};

template <typename Proto>
class BackendClientComputation final : public BackendComputation {
 public:
  BackendClientComputation(BackendClient<Proto>& backend, size_t id,
                           size_t time_remaining)
      : backend_(backend),
        id_(id),
        priority_(TimeToPriority(time_remaining)),
        entries_(backend_.GetAttributes().maximum_batch_size) {}

  ~BackendClientComputation() {}

  size_t UsedBatchSize() const override { return entries_.size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    InputPosition input_pos;
    size_t len = std::min(pos.pos.size(), (size_t)kMoveHistory);
    input_pos.history_length_ = len - 1;
    input_pos.base_ = pos.pos[pos.pos.size() - len];
    for (size_t i = 1; i < len; ++i) {
      // Convert history planes to moves for smaller message size.
      Move move = pos.pos[pos.pos.size() - len + i - 1].GetNextMove(
          pos.pos[pos.pos.size() - len + i]);
      input_pos.history_[i - 1] = move;
    }
    entries_.emplace_back(input_pos, result);
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    LCTRACE_FUNCTION_SCOPE;
    std::vector<InputPosition> inputs;
    inputs.reserve(entries_.size());
    for (const auto& entry : entries_) {
      inputs.emplace_back(entry.pos_);
    }
    auto connection = backend_.GetConnection(this);
    connection->ComputeBlocking(GetId(), priority_, inputs);
    assert(!connection->IsOpen() || entries_.size() == 0);
  }

  // Clears entries when the server reply has been processed.
  void NotifyComputationCompleted() {
    entries_.clear();
  }

  size_t GetId() const { return id_; }

  const EvalResultPtr& GetResult(size_t index) const {
    return entries_[index].result_;
  }

 private:
  // Compute priority based on time remaining.
  size_t TimeToPriority(size_t time_remaining) {
    auto fixed = backend_.GetFixedPriority();
    if (fixed < kMaxComputationPriority) {
      return fixed;
    }
    size_t max_limit = 125;
    size_t priority = 0;
    while (time_remaining > max_limit &&
           priority < kMaxComputationPriority - 1) {
      max_limit *= 2;
      ++priority;
    }
    return priority;
  }

  struct Entry {
    Entry(const InputPosition& pos, const EvalResultPtr& result)
        : pos_(pos), result_(result) {}
    const InputPosition pos_;
    const EvalResultPtr result_;
  };

  BackendClient<Proto>& backend_;
  size_t id_;
  size_t priority_;
  AtomicVector<Entry> entries_;
};

template <typename Proto>
template <typename Archive>
typename Archive::ResultType ClientConnection<Proto>::HandleMessage(
    const ComputeBlockingReply& message, Archive& ar) {
  LCTRACE_FUNCTION_SCOPE;
  if (!message.error_message_.empty()) {
    CERR << "Compute blocking error: " << message.error_message_;
    return Unexpected(ArchiveError::RemoteError);
  }
  auto* computation = GetReservedComputation();
  if (message.computation_id_ != computation->GetId()) {
    CERR << "Received ComputeBlockingReply for unknown computation ID "
         << message.computation_id_ << ", expected " << computation_id_;
    return Unexpected(ArchiveError::InvalidData);
  }
  if (message.results_.size() != computation->UsedBatchSize()) {
    CERR << "Received ComputeBlockingReply with unexpected number of "
            "results: "
         << message.results_.size() << " expected "
         << computation->UsedBatchSize();
    return Unexpected(ArchiveError::InvalidData);
  }

  for (size_t i = 0; i < message.results_.size(); ++i) {
    const auto& net_result = message.results_[i];
    const auto& result_ptr = computation->GetResult(i);
    if (result_ptr.q) *result_ptr.q = net_result.value_;
    if (result_ptr.d) *result_ptr.d = net_result.draw_;
    if (result_ptr.m) *result_ptr.m = net_result.moves_left_;
    if (!result_ptr.p.empty()) {
      if (net_result.policy_.size() != result_ptr.p.size()) {
        CERR << "Received ComputeBlockingReply with unexpected policy size: "
             << net_result.policy_.size() << " expected "
             << result_ptr.p.size();
        return Unexpected(ArchiveError::InvalidData);
      }
      std::copy(net_result.policy_.begin(), net_result.policy_.end(),
                result_ptr.p.begin());
    }
  }
  computation->NotifyComputationCompleted();

  return {ar};
}

template <typename Proto>
std::unique_ptr<BackendComputation> BackendClient<Proto>::CreateComputation(
    size_t time_remaining) {
  LCTRACE_FUNCTION_SCOPE;
  return std::make_unique<BackendClientComputation<Proto>>(
      *this,
      (uint16_t)next_computation_id_.fetch_add(1, std::memory_order_relaxed),
      time_remaining);
}

}  // namespace

class ClientFactory final : public BackendFactory {
 public:
  static BackendManager::Register reg_;
  virtual int GetPriority() const override { return -1; }
  virtual std::string_view GetName() const override { return "client"; }
  virtual std::unique_ptr<Backend> Create(const OptionsDict& options,
                                          const std::string& network) override {
    const std::string backend_options =
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId);
    OptionsDict client_options;
    client_options.AddSubdictFromString(backend_options);
#ifdef ASIO_HAS_LOCAL_SOCKETS
    if (client_options.GetOrDefault("protocol", kDefaultProtocol) == "unix") {
      return std::make_unique<BackendClient<asio::local::stream_protocol>>(
          network, client_options);
    }
#endif
    return std::make_unique<BackendClient<asio::ip::tcp>>(network,
                                                          client_options);
  }
};

BackendManager::Register ClientFactory::reg_(std::make_unique<ClientFactory>());

}  // namespace lczero::client

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

// clang-format off
#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#include <io.h>
#define popen _popen
#define pclose _pclose
#endif
// clang-format on

#include <asio.hpp>
#include <asio/io_context.hpp>
#include <atomic>
#include <regex>
#include <stdio.h>
#include <thread>

#include "neural/backend.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "proto.h"
#include "utils/atomic_vector.h"
#include "utils/commandline.h"
#include "utils/trace.h"

namespace lczero::client {

namespace {

class Context {
 public:
  // TODO: Use a shared io thread for multiple connections.
  Context() : io_thread_([this] { this->io_context().run(); }) {}
  ~Context() {
    work_guard_.reset();
    io_thread_.join();
  }
  asio::io_context& io_context() { return io_context_; }

  void Connected() { work_guard_.reset(); }

 private:
  asio::io_context io_context_;
  asio::executor_work_guard<asio::io_context::executor_type> work_guard_{
      io_context_.get_executor()};
  std::thread io_thread_;
};

template <typename Proto>
class BackendClientComputation;

template <typename Proto>
class ClientConnection final : public Context,
                               public Connection<typename Proto::socket> {
  using Base = client::Connection<typename Proto::socket>;
  using ComputationMap = std::map<size_t, BackendClientComputation<Proto>*>;
  using SocketType = typename Proto::socket;
  using Endpoint = typename Proto::endpoint;

 public:
  ClientConnection(const OptionsDict& options)
      : Context(),
        Base(SocketType{this->io_context()}),
        wraped_pipe_(this->io_context()) {
    // Initialize connection.
    LCTRACE_FUNCTION_SCOPE;
    Endpoint endpoint;
    std::string args = " backendserver --accept-limit=1 ";
    if constexpr (std::is_same_v<Proto, asio::local::stream_protocol>) {
      const std::string pipe =
          options.GetOrDefault("pipe_name", kDefaultPipeName);
      endpoint = GetEndpoint<Endpoint>(pipe);
      args += "--protocol=unix --pipe-name='" + pipe + "'";
    } else {
      const std::string host = options.GetOrDefault("tcp-host", kDefaultHost);
      const int port = options.GetOrDefault("tcp-port", kDefaultPort);
      const std::string port_str = std::to_string(port);
      endpoint = GetEndpoint<Endpoint>(this->io_context(), host, port_str);
      args += "--protocol=tcp --tcp-host='" + host + "' --tcp-port='" +
              port_str + "'";
    }
    try {
      this->Connect(endpoint);
    } catch (const std::exception& e) {
      std::string command = CommandLine::BinaryName() + args;
      FILE* pipe_ = popen(command.c_str(), "r");
      if (!pipe_) {
        CERR << "Failed to start backend server with command: " << command;
        throw Exception("Failed to start backend server");
      }
#ifdef _WIN32
      wraped_pipe_.assign((asio::windows::stream_handle::native_handle_type)_get_osfhandle(fileno(pipe_)));
#else
      wraped_pipe_.assign(fileno(pipe_));
#endif
      std::regex ready_message("^info string Backend server listening on ");
      do {
        std::string buffer;
        size_t count =
            asio::read_until(wraped_pipe_, asio::dynamic_buffer(buffer), '\n');
        if (!count || std::regex_search(buffer, ready_message)) {
          break;
        }
      } while (true);
      pipe_input_buffer_.resize(512);
      ReadPipe();
      this->Connect(endpoint);
    }
    CERR << "Connected to backend server " << endpoint;
  }

  ~ClientConnection() {
    wraped_pipe_.release();
    Base::Close();
    this->io_context().stop();
    if (pipe_) {
      pclose(pipe_);
    }
  }

  void Close() override {
    Base::Close();
    throw Exception("Backend client connection closed");
  }

  void Start(const std::string& network) { WriteHandshake(network); }

  struct FakeSelf {};

  BackendAttributes GetAttributes() const { return attrs_; }

  void ComputeBlocking(size_t computation_id,
                       std::vector<std::span<const Position>>& inputs) {
    client::ComputeBlocking message;
    message.computation_id_ = computation_id;
    message.inputs_ = std::move(inputs);
    Base::SendMessage(Self(), message);
  }

  ComputationMap& GetComputations() { return computations_; }

  SpinMutex::Lock Lock() ACQUIRE(mutex_) { return {mutex_}; }

  void WaitForHandshake() {
    handshake_completed_.wait(false, std::memory_order_acquire);
  }

 private:
  FakeSelf Self() { return {}; }

  void Read() {
    Base::ReadHeader([this](const auto& message) {
      // Clang warns about unused this if not using this for the call.
      return this->HandleMessage(message);
    });
  }

  void ReadPipe() {
    wraped_pipe_.async_read_some(
        asio::buffer(pipe_input_buffer_), [this](std::error_code ec, size_t) {
          if (ec) {
            if (ec != asio::error::eof &&
                ec != asio::error::operation_aborted) {
              CERR << "Pipe read error: " << ec.message();
            }
            return;
          }
          ReadPipe();
        });
  }

  void WriteHandshake(const std::string& network) {
    Handshake message;
    message.network_name_ = network;

    Base::SendMessage(Self(), message);
    Read();
    Context::Connected();
  }

  template <typename MessageType>
  int HandleMessage(const MessageType& message) {
    // Handle different message types here.
    CERR << "Received unexpected message of type: " << message.header_.type_;
    return -1;
  }

  int HandleMessage(const HandshakeReply& message) {
    LCTRACE_FUNCTION_SCOPE;
    assert(message.header_.type_ == MessageType::HANDSHAKE_REPLY);
    if (!message.error_message_.empty()) {
      CERR << "Handshake error: " << message.error_message_;
      return -1;
    }
    if ((message.attributes_.has_mlh & 1) != message.attributes_.has_mlh) {
      CERR << "has_mlh support has unexpected value: " << std::hex
           << static_cast<unsigned>(message.attributes_.has_mlh);
      return -1;
    }
    if ((message.attributes_.has_wdl & 1) != message.attributes_.has_wdl) {
      CERR << "has_wdl support has unexpected value: " << std::hex
           << static_cast<unsigned>(message.attributes_.has_wdl);
      return -1;
    }
    if ((message.attributes_.runs_on_cpu & 1) !=
        message.attributes_.runs_on_cpu) {
      CERR << "runs_on_cpu has unexpected value: " << std::hex
           << static_cast<unsigned>(message.attributes_.runs_on_cpu);
      return -1;
    }
    if (message.attributes_.maximum_batch_size >=
            static_cast<int>(kMaxMessageSize) &&
        message.attributes_.maximum_batch_size < 0) {
      CERR << "maximum_batch_size outside accepted range: "
           << message.attributes_.maximum_batch_size;
      return -1;
    }
    if (message.attributes_.recommended_batch_size >=
            message.attributes_.maximum_batch_size &&
        message.attributes_.recommended_batch_size < 0) {
      CERR << "recommended_batch_size outside 0 < "
           << message.attributes_.recommended_batch_size << " < "
           << message.attributes_.maximum_batch_size;
      return -1;
    }
    if (message.attributes_.suggested_num_search_threads >=
            static_cast<int>(kMaxSearchThreads) &&
        message.attributes_.suggested_num_search_threads < 0) {
      CERR << "suggested_num_search_threads outside accepted range: "
           << message.attributes_.suggested_num_search_threads;
      return -1;
    }
    attrs_ = message.attributes_;
    handshake_completed_.store(true, std::memory_order_release);
    handshake_completed_.notify_one();
    TRACE << "Received handshake response. Batch size: "
         << attrs_.recommended_batch_size << "/" << attrs_.maximum_batch_size;
    // Handshake complete, ready to proceed.
    this->Defer([this] { Read(); });
    return 0;
  }

  int HandleMessage(const ComputeBlockingReply& message);

  SpinMutex mutex_;
  FILE* pipe_ = nullptr;
  asio::readable_pipe wraped_pipe_;
  std::vector<char> pipe_input_buffer_;
  BackendAttributes attrs_{};
  std::atomic<bool> handshake_completed_ = false;
  ComputationMap computations_;
};

template <typename Proto>
class BackendClient final : public Backend {
 public:
  BackendClient(const std::string& network, const OptionsDict& options)
      : connection_{options} {
    LCTRACE_FUNCTION_SCOPE;
    connection_.Start(network);

    connection_.WaitForHandshake();
    assert(connection_.GetAttributes().maximum_batch_size != 0 ||
           connection_.io_context().stopped());
    CERR << "Connected to backend server using " << network;
  }

  BackendAttributes GetAttributes() const override {
    return connection_.GetAttributes();
  }
  std::unique_ptr<BackendComputation> CreateComputation() override;

  void InsertComputation(size_t id,
                         BackendClientComputation<Proto>* computation) {
    auto lock = connection_.Lock();
    connection_.GetComputations().emplace(id, computation);
  }
  void EraseComputation(size_t id) {
    auto lock = connection_.Lock();
    connection_.GetComputations().erase(id);
  }

  void ComputeBlocking(size_t id,
                       std::vector<std::span<const Position>>& inputs) {
    connection_.ComputeBlocking(id, inputs);
  }

 private:
  std::atomic<size_t> next_computation_id_ = 0;
  ClientConnection<Proto> connection_;
};

template <typename Proto>
class BackendClientComputation final : public BackendComputation {
 public:
  BackendClientComputation(BackendClient<Proto>& backend, size_t id)
      : backend_(backend),
        id_(id),
        entries_(backend_.GetAttributes().maximum_batch_size) {}

  ~BackendClientComputation() { backend_.EraseComputation(id_); }

  size_t UsedBatchSize() const override { return entries_.size(); }

  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override {
    entries_.emplace_back(Entry{pos, result});
    return ENQUEUED_FOR_EVAL;
  }

  void ComputeBlocking() override {
    LCTRACE_FUNCTION_SCOPE;
    std::vector<std::span<const Position>> inputs;
    inputs.reserve(entries_.size());
    for (const auto& entry : entries_) {
      if (entry.pos_.pos.size() <= kMoveHistory) {
        inputs.emplace_back(entry.pos_.pos);
      } else {
        inputs.emplace_back(
            entry.pos_.pos.begin() + (entry.pos_.pos.size() - kMoveHistory),
            entry.pos_.pos.end());
      }
    }
    backend_.InsertComputation(GetId(), this);
    backend_.ComputeBlocking(GetId(), inputs);
    results_ready_.wait(false, std::memory_order_relaxed);
    TRACE << "Computation ID " << id_ << " completed. "
          << results_ready_.load(std::memory_order_relaxed);
  }

  size_t GetId() const { return id_; }

  const EvalResultPtr& GetResult(size_t index) const {
    return entries_[index].result_;
  }

  void NotifyResultsReady() {
    TRACE << "Results ready for computation ID " << id_;
    results_ready_.store(true, std::memory_order_relaxed);
    results_ready_.notify_one();
  }

 private:
  struct Entry {
    const EvalPosition pos_;
    const EvalResultPtr result_;
  };

  std::atomic<bool> results_ready_ = false;
  BackendClient<Proto>& backend_;
  size_t id_;
  AtomicVector<Entry> entries_;
};

template <typename Proto>
int ClientConnection<Proto>::HandleMessage(
    const ComputeBlockingReply& message) {
  LCTRACE_FUNCTION_SCOPE;
  if (!message.error_message_.empty()) {
    CERR << "Compute blocking error: " << message.error_message_;
    return -1;
  }
  auto lock = Lock();
  auto iter = computations_.find(message.computation_id_);
  if (iter == computations_.end()) {
    CERR << "Received ComputeBlockingReply for unknown computation ID "
         << message.computation_id_;
    return -1;
  }
  if (message.results_.size() != iter->second->UsedBatchSize()) {
    CERR << "Received ComputeBlockingReply with unexpected number of "
            "results: "
         << message.results_.size() << " expected "
         << iter->second->UsedBatchSize();
    return -1;
  }

  for (size_t i = 0; i < message.results_.size(); ++i) {
    const auto& net_result = message.results_[i];
    const auto& result_ptr = iter->second->GetResult(i);
    if (result_ptr.q) *result_ptr.q = net_result.value_;
    if (result_ptr.d) *result_ptr.d = net_result.draw_;
    if (result_ptr.m) *result_ptr.m = net_result.moves_left_;
    if (!result_ptr.p.empty()) {
      if (net_result.policy_.size() != result_ptr.p.size()) {
        CERR << "Received ComputeBlockingReply with unexpected policy size: "
             << net_result.policy_.size() << " expected "
             << result_ptr.p.size();
        return -1;
      }
      std::copy(net_result.policy_.begin(), net_result.policy_.end(),
                result_ptr.p.begin());
    }
  }
  iter->second->NotifyResultsReady();

  this->Defer([this] { Read(); });
  return 0;
}

template <typename Proto>
std::unique_ptr<BackendComputation> BackendClient<Proto>::CreateComputation() {
  LCTRACE_FUNCTION_SCOPE;
  return std::make_unique<BackendClientComputation<Proto>>(
      *this,
      (uint16_t)next_computation_id_.fetch_add(1, std::memory_order_relaxed));
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

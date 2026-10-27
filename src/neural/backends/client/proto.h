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

#pragma once

#include <array>
#include <cstdint>
#include <queue>
#include <span>
#include <string_view>

#include "neural/backend.h"
#include "neural/backends/client/archive.h"
#include "neural/encoder.h"
#include "utils/asio.h"

namespace lczero::client {

#ifdef ASIO_HAS_LOCAL_SOCKETS
const std::string kDefaultProtocol = "unix";
#else
const std::string kDefaultProtocol = "tcp";
#endif

const std::string kDefaultHost = "localhost";
const int kDefaultPort = 9433;
#ifdef _WIN32
const std::string kDefaultPipeName = "lczero_backend_pipe";
#else
const std::string kDefaultPipeName = "/tmp/lczero_backend_pipe";
#endif

// Message types.
enum MessageType : uint8_t {
  HANDSHAKE = 0,
  HANDSHAKE_REPLY = 1,
  COMPUTE_BLOCKING = 2,
  COMPUTE_BLOCKING_REPLY = 3,
};

using MagicType = uint32_t;

static constexpr MagicType kMagic = 'L' << 0 | 'C' << 8 | 'Z' << 16 | 'B' << 24;
// Must be incremented when any structure changes.
static constexpr uint16_t kBackendApiVersion = 0;
static constexpr unsigned kMaxComputationPriority = 12;
static constexpr size_t kMaxSearchThreads = 2;
static constexpr size_t kMaxMinibatchSizes = 1024;
static constexpr size_t kLegalMovesInPosition = 120;

// Shared message header structure for all messages.
struct MessageHeader {
  MagicType magic_ = kMagic;  // "LCZB"
  uint32_t size_;
  uint8_t type_;

  template <typename Archive>
  typename Archive::ResultType Serialize(
      Archive& ar, [[maybe_unused]] const unsigned version) {
    auto r = ar & FixedInteger(magic_);
    r = r.and_then([this](Archive& ar) { return ar & size_; });
    r = r.and_then([this](Archive& ar) { return ar & type_; });
    return r;
  }

  static constexpr size_t Size() {
    return sizeof(MagicType) + sizeof(uint32_t) + sizeof(MessageType);
  }

  size_t PredictedSize() const { return Size(); }
};

// Handshake message sent by the client to the backend.
struct Handshake {
  MessageHeader header_ = {kMagic, 0, MessageType::HANDSHAKE};
  uint16_t backend_api_version_ = kBackendApiVersion;
  std::string_view network_name_{};

  template <typename Archive>
  typename Archive::ResultType Serialize(
      Archive& ar, [[maybe_unused]] const unsigned version) {
    typename Archive::ResultType r{ar};
    r = Archive::is_saving ? ar & header_ : r;
    r = r.and_then([this](Archive& ar) { return ar & backend_api_version_; });
    r = r.and_then([this](Archive& ar) { return ar & network_name_; });
    return r;
  }
};

// Handshake reply message sent by the backend to the client.
struct HandshakeReply {
  MessageHeader header_ = {kMagic, 0, MessageType::HANDSHAKE_REPLY};
  BackendAttributes attributes_{};
  std::string_view error_message_{};

  template <typename Archive>
  typename Archive::ResultType Serialize(
      Archive& ar, [[maybe_unused]] const unsigned version) {
    typename Archive::ResultType r{ar};
    r = Archive::is_saving ? ar & header_ : r;
    r = r.and_then([this](Archive& ar) { return ar & attributes_; });
    r = r.and_then([this](Archive& ar) { return ar & error_message_; });
    return r;
  }
};

// Input position with move history.
struct InputPosition {
  Position base_{};
  unsigned char history_length_{0};
  std::array<Move, kMoveHistory - 1> history_{};

  template <typename Archive>
  typename Archive::ResultType Serialize(
      Archive& ar, [[maybe_unused]] const unsigned version) {
    auto r = ar & base_;
    r = r.and_then([this](Archive& ar) { return ar & history_length_; });
    const unsigned len = history_length_;
    if (len > history_.size()) {
      return Unexpected{ArchiveError::InvalidData};
    }
    for (unsigned i = 0; i < len; ++i) {
      r = r.and_then([this, i](Archive& ar) { return ar & history_[i]; });
    }
    return r;
  }
};

// Compute blocking message sent by the client to the backend.
struct ComputeBlocking {
  MessageHeader header_ = {kMagic, 0, MessageType::COMPUTE_BLOCKING};
  uint16_t computation_id_{};
  unsigned char priority_{};
  std::vector<InputPosition> inputs_{};

  template <typename Archive>
  typename Archive::ResultType Serialize(
      Archive& ar, [[maybe_unused]] const unsigned version) {
    typename Archive::ResultType r{ar};
    r = Archive::is_saving ? ar & header_ : r;
    r = r.and_then([this](Archive& ar) { return ar & computation_id_; });
    r = r.and_then([this](Archive& ar) { return ar & priority_; });
    r = r.and_then([this](Archive& ar) {
      return ar & VectorLimits(inputs_, 1, kMaxMinibatchSizes);
    });
    return r;
  }
};

// Network result for a position.
// TODO: maybe allow transmitting fp16 values.
struct NetworkResult {
  float value_{};
  float draw_{};
  float moves_left_{};
  std::span<float> policy_{};

  template <typename Archive>
  typename Archive::ResultType Serialize(
      Archive& ar, [[maybe_unused]] const unsigned version) {
    auto r = ar & value_;
    r = r.and_then([this](Archive& ar) { return ar & draw_; });
    r = r.and_then([this](Archive& ar) { return ar & moves_left_; });
    r = r.and_then([this](Archive& ar) { return ar & policy_; });
    return r;
  }
};

// Compute blocking reply message sent by the backend server to the client.
struct ComputeBlockingReply {
  MessageHeader header_ = {kMagic, 0, MessageType::COMPUTE_BLOCKING_REPLY};
  uint16_t computation_id_{};
  std::vector<NetworkResult> results_{};
  std::string_view error_message_{};

  template <typename Archive>
  typename Archive::ResultType Serialize(
      Archive& ar, [[maybe_unused]] const unsigned version) {
    typename Archive::ResultType r{ar};
    r = Archive::is_saving ? ar & header_ : r;
    r = r.and_then([this](Archive& ar) { return ar & computation_id_; });
    r = r.and_then([this](Archive& ar) {
      return ar & VectorLimits(results_, 1, kMaxMinibatchSizes);
    });
    r = r.and_then([this](Archive& ar) { return ar & error_message_; });
    return r;
  }
};

static constexpr size_t kMaxBytesPerPosition =
    sizeof(float) * kLegalMovesInPosition + sizeof(NetworkResult);
static constexpr size_t kMaxMessageSize =
    kMaxBytesPerPosition * kMaxMinibatchSizes + sizeof(ComputeBlockingReply);
static constexpr size_t kBufferSize = std::bit_ceil(kMaxMessageSize);

// Serialize a message to an output archive. The function fills the size field
// of header.
template <typename Archive, typename T>
[[nodiscard]]
typename Archive::ResultType SerializeMessage(Archive& oa, T& message);

// Parse an incoming message header.
template <typename Archive>
[[nodiscard]]
typename Archive::ResultType ParseMessageHeader(Archive& ia,
                                                MessageHeader& header);

// Parse an incoming message body. It will call the callback function if parsing
// succeeds.
template <typename Archive, typename T, typename Callback>
[[nodiscard]]
typename Archive::ResultType ParseMessageType(Archive& ia,
                                              const MessageHeader& header,
                                              T& out, Callback&& callback) {
  assert(ia.Size() >= header.size_);
  out.header_ = header;
  return (ia & out).and_then(
      [&out, &callback](Archive& ar) { return callback(out, ar); });
}

// Helper to select the correct body parsing function based on message type.
template <typename Archive, typename Callback>
[[nodiscard]]
typename Archive::ResultType ParseMessage(Archive& ia,
                                          const MessageHeader& header,
                                          Callback&& callback) {
  assert(ia.Size() >= header.size_);
  switch (header.type_) {
    case MessageType::HANDSHAKE: {
      Handshake msg;
      return ParseMessageType(ia, header, msg,
                              std::forward<Callback>(callback));
    }
    case MessageType::HANDSHAKE_REPLY: {
      HandshakeReply msg;
      return ParseMessageType(ia, header, msg,
                              std::forward<Callback>(callback));
    }
    case MessageType::COMPUTE_BLOCKING: {
      ComputeBlocking msg;
      return ParseMessageType(ia, header, msg,
                              std::forward<Callback>(callback));
    }
    case MessageType::COMPUTE_BLOCKING_REPLY: {
      ComputeBlockingReply msg;
      return ParseMessageType(ia, header, msg,
                              std::forward<Callback>(callback));
    }
    default:
      CERR << "Unknown message type received: " << header.type_;
      return Unexpected{ArchiveError::InvalidData};
  }
}

#ifdef ASIO_HAS_LOCAL_SOCKETS
// Get a local socket endpoint.
// TODO: These could parse a full URI to allow more flexible addressing.
// Examples are unix:///path/to/socket, tcp://localhost:port or
// tls://hostname:port.
template <typename Endpoint>
Endpoint GetEndpoint(const std::string& pipe_name) {
  return {pipe_name};
}
#endif
template <typename Endpoint>
Endpoint GetEndpoint(asio::io_context& ctx, const std::string& host,
                     const std::string& port) {
  asio::ip::tcp::resolver resolver(ctx);
  auto addrs = resolver.resolve(host, port);
  auto iter = addrs.begin();
  auto endpoint = iter->endpoint();
  return {endpoint.address(), endpoint.port()};
}

// Shared connection implementation for server and client. It handles socket
// options, parsing and writing messages.
template <typename SocketType>
class Connection {
 public:
  explicit Connection(SocketType&& socket) : socket_(std::move(socket)) {
    input_.resize(kMaxMessageSize);
    SetSocketOptions();
  }
  virtual ~Connection() = default;

  bool IsOpen() const { return socket_.is_open(); }

 protected:
  // Set common socket options.
  void SetSocketOptions() {
    if (socket_.is_open()) {
      if constexpr (std::is_same_v<SocketType, asio::ip::tcp::socket>) {
        socket_.set_option(asio::ip::tcp::no_delay(true));
      }
      typename SocketType::keep_alive keep_alive_option(true);
      socket_.set_option(keep_alive_option);
    }
  }

  // Starting point for receiving a new message. It first reads and parses a
  // header structure. Then it call ReadBody to receive and parsee the message
  // body.
  // The callback function is called when a full message has been parsed. If
  // there are connection errors or parsing errors, the connection is closed and
  // the callback is not called.
  template <bool async = true, bool new_request = true,
            typename MessageCallback>
  void ReadHeader(MessageCallback&& callback) {
    if (new_request) {
      auto rv = ParseHeader<async>(std::forward<MessageCallback>(callback));
      if (rv == ArchiveError::None) return;
      if (rv != ArchiveError::BufferOverflow) {
        CERR << "Error parsing message header: " << rv;
        Close();
        return;
      }
      if (ReadyBytes() > 0) {
        // Move unparsed data to the front.
        std::memmove(input_.data(), input_.data() + parsed_bytes_,
                     ReadyBytes());
      }
      input_read_bytes_ = ReadyBytes();
      parsed_bytes_ = 0;
    }
    // Shared message handling code for async and sync reads.
    auto handler = [this, callback = std::move(callback)](std::error_code ec,
                                                          size_t length) {
      if (ec) {
        // Handle error or disconnection.
        if (ec != asio::error::eof && ec != asio::error::operation_aborted) {
          CERR << "Connection error in ReadHeader: " << ec.message();
        }
        Close();
        return;
      }
      input_read_bytes_ += length;
      // Process data.
      auto rv = ParseHeader<async>(std::move(callback));
      switch (rv) {
        case ArchiveError::None:
          return;
        case ArchiveError::BufferOverflow:
          // Need more data.
          ReadHeader<async, false>(std::move(callback));
          return;
        default:
          CERR << "Error parsing message header: " << rv;
          Close();
          return;
      }
    };
    if (async) {
      socket_.async_read_some(InputBuffer(), handler);
    } else {
      asio::error_code ec;
      size_t length = socket_.read_some(InputBuffer(), ec);
      handler(ec, length);
    }
  }

  // Serialize and send a message.
  template <bool async = true, typename SelfType, typename MessageType>
  void SendMessage(SelfType&& self, MessageType& message) {
    BinaryOArchive output(kBackendApiVersion);
    if (!SerializeMessage(output, message)) {
      CERR << "Error serializing message for sending.";
      Close();
      return;
    }
    if (async) {
      Dispatch([this, self = std::move(self), output = std::move(output)] {
        bool empty = queue_.empty();
        queue_.push(std::move(output));
        if (!empty) {
          // A write is already in progress.
          return;
        }
        Write(std::move(self));
      });
    } else {
      asio::error_code ec;
      // Synchronous write. asio::write restarts system calls if interrupted.
      asio::write(socket_, asio::buffer(output.GetVector()), ec);
      if (ec) {
        CERR << "Connection error in SendMessage: " << ec.message();
        Close();
        return;
      }
    }
  }

  // Dispatch a function to be run on the socket's executor. If called from
  // executing thread, the function is run immediately from asio::dispatch.
  template <typename FunctionType>
  void Dispatch(FunctionType&& func) {
    asio::dispatch(socket_.get_executor(), std::forward<FunctionType>(func));
  }

  // Defer a function to be run on the socket's executor later. It typically
  // will execute after the current handlers return to the event loop.
  template <typename FunctionType>
  void Defer(FunctionType&& func) {
    asio::defer(socket_.get_executor(), std::forward<FunctionType>(func));
  }

  // Write an asynchronous queued message to the socket.
  template <typename SelfType>
  void Write(SelfType&& self) {
    asio::async_write(
        socket_, asio::buffer(queue_.front().GetVector()),
        [this, self](std::error_code ec, [[maybe_unused]] size_t length) {
          if (ec) {
            if (ec != asio::error::eof) {
              CERR << "Connection error in SendMessage: " << ec.message();
            }
            Close();
            return;
          }
          queue_.pop();
          if (queue_.empty()) {
            return;
          }
          Write(std::forward<SelfType>(self));
        });
  }

  // Synchronously connect to an endpoint.
  template <typename Endpoint>
  void Connect(const Endpoint& endpoint) {
    socket_.connect(endpoint);
    SetSocketOptions();
  }

  // Close the connection.
  virtual void Close() {
    if (!socket_.is_open()) {
      return;
    }
    // Make sure the socket is closed gracefully.
    socket_.shutdown(SocketType::shutdown_both);
    socket_.close();
  }

 private:
  // Read the message body based on the pending header.
  template <bool async, bool new_request = true, typename MessageCallback>
  void ReadBody(MessageCallback&& callback) {
    // Check if we already have the full body.
    if (ReadyBytes() >= pending_header_.size_) {
      ParseInput(std::move(callback));
      return;
    }
    // Move unparsed data to the front if this is a new request and there is
    // not enough space for full body.
    if (new_request && pending_header_.size_ >= input_.size() - parsed_bytes_) {
      if (ReadyBytes() > 0) {
        // Move unparsed data to the front.
        std::memmove(input_.data(), input_.data() + parsed_bytes_,
                     ReadyBytes());
      }
      input_read_bytes_ = ReadyBytes();
      parsed_bytes_ = 0;
    }
    auto handler = [this, callback = std::move(callback)](std::error_code ec,
                                                          size_t length) {
      if (ec) {
        // Handle error or disconnection.
        if (ec != asio::error::eof) {
          CERR << "Connection error in ReadBody: " << ec.message();
        }
        Close();
        return;
      }
      input_read_bytes_ += length;
      if (ReadyBytes() < pending_header_.size_) {
        // Need more data.
        ReadBody<false>(std::move(callback));
        return;
      }
      // Process data.
      ParseInput(std::move(callback));
    };
    if (async) {
      socket_.async_read_some(InputBuffer(), handler);
    } else {
      asio::error_code ec;
      // Synchronous read. It can return less data than requested.
      size_t length = socket_.read_some(InputBuffer(), ec);
      handler(ec, length);
    }
  }

  const char* InputBegin() { return input_.data() + parsed_bytes_; }
  const char* InputEnd() { return input_.data() + input_read_bytes_; }

  // Get an io buffer which can receive network data.
  auto InputBuffer() {
    return asio::buffer(input_.data() + input_read_bytes_,
                        input_.size() - input_read_bytes_);
  }

  // Parse a message header from the input buffer.
  template <bool async, typename MessageCallback>
  ArchiveError ParseHeader(MessageCallback&& callback) {
    if (ReadyBytes() == 0) return ArchiveError::BufferOverflow;

    auto first = InputBegin();
    BinaryIArchive ia(std::span<const char>(first, InputEnd()), input_.data(),
                      0, kBackendApiVersion);
    size_t size = ia.Size();
    auto rv = ParseMessageHeader(ia, pending_header_);
    if (!rv) {
      return rv.error();
    }
    parsed_bytes_ += size - ia.Size();
    ReadBody<async>(std::forward<MessageCallback>(callback));
    return ArchiveError::None;
  }

  // Parse a message body from the input buffer.
  template <typename MessageCallback>
  void ParseInput(MessageCallback&& callback) {
    assert(ReadyBytes() >= pending_header_.size_);
    auto first = InputBegin();
    BinaryIArchive ia(std::span<const char>(first, InputEnd()), input_.data(),
                      0, kBackendApiVersion);
    size_t size = ia.Size();
    if (!ParseMessage(ia, pending_header_,
                      std::forward<MessageCallback>(callback))) {
      CERR << "Error parsing message "
           << static_cast<uint32_t>(pending_header_.type_) << " body.";
      Close();
      return;
    }
    parsed_bytes_ += size - ia.Size();
  }

  // Get number of unparsed bytes in the input buffer.
  size_t ReadyBytes() const { return input_read_bytes_ - parsed_bytes_; }

  // Queue of outgoing messages for asynchronous sending. It must only be
  // accessed from the socket's executor. If executor runs in a thread pool,
  // then a strand must be used to synchronize access.
  std::queue<BinaryOArchive> queue_;

  SocketType socket_;

  size_t input_read_bytes_ = 0;
  size_t parsed_bytes_ = 0;
  MessageHeader pending_header_;
  std::vector<char> input_;
};

}  // namespace lczero::client

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

// clang-format off
#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#endif
// clang-format on

#include <array>
#include <asio.hpp>
#include <cstdint>
#include <queue>
#include <span>
#include <string_view>

#include "archive.h"
#include "neural/backend.h"
#include "neural/encoder.h"

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

struct null_stream {
  template <typename T>
  null_stream& operator<<(const T&) {
    return *this;
  }
};

#if 1
#define TRACE \
  lczero::client::null_stream {}
#else
#define TRACE CERR
#endif

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

struct MessageHeader {
  MagicType magic_ = kMagic;  // "LCZB"
  uint32_t size_;
  uint8_t type_;

  template <typename Archive>
  Archive::ResultType Serialize(Archive& ar,
                                [[maybe_unused]] const unsigned version) {
    auto r = ar & FixedInteger(magic_);
    r = r.and_then([this](Archive& ar) { return ar & size_; });
    r = r.and_then([this](Archive& ar) { return ar & type_; });
    TRACE << "MessageHeader::Serialize(" << ar.Size() << "): size=" << size_
          << " type=" << static_cast<uint32_t>(type_);
    return r;
  }

  static constexpr size_t Size() {
    return sizeof(MagicType) + sizeof(uint32_t) + sizeof(MessageType);
  }

  size_t PredictedSize() const { return Size(); }
};

struct Handshake {
  MessageHeader header_ = {kMagic, 0, MessageType::HANDSHAKE};
  uint16_t backend_api_version_ = kBackendApiVersion;
  std::string_view network_name_{};

  template <typename Archive>
  Archive::ResultType Serialize(Archive& ar,
                                [[maybe_unused]] const unsigned version) {
    typename Archive::ResultType r{ar};
    r = Archive::is_saving ? ar & header_ : r;
    r = r.and_then([this](Archive& ar) { return ar & backend_api_version_; });
    r = r.and_then([this](Archive& ar) { return ar & network_name_; });
    TRACE << "Handshake::Serialize(" << ar.Size()
          << "): backend_api_version=" << backend_api_version_
          << " network_name=" << network_name_ << " size=" << ar.Size();
    return r;
  }
};

struct HandshakeReply {
  MessageHeader header_ = {kMagic, 0, MessageType::HANDSHAKE_REPLY};
  BackendAttributes attributes_{};
  std::string_view error_message_{};

  template <typename Archive>
  Archive::ResultType Serialize(Archive& ar,
                                [[maybe_unused]] const unsigned version) {
    TRACE << "HandshakeReply::Serialize(" << ar.Size()
          << "): error_message=" << error_message_;
    typename Archive::ResultType r{ar};
    r = Archive::is_saving ? ar & header_ : r;
    r = r.and_then([this](Archive& ar) { return ar & attributes_; });
    r = r.and_then([this](Archive& ar) { return ar & error_message_; });
    return r;
  }
};

struct InputPosition {
  Position base_{};
  unsigned char history_length_{0};
  std::array<Move, kMoveHistory - 1> history_{};

  template <typename Archive>
  Archive::ResultType Serialize(Archive& ar,
                                [[maybe_unused]] const unsigned version) {
    auto r = ar & base_;
    r = r.and_then([this](Archive& ar) { return ar & history_length_; });
    const unsigned len = history_length_;
    assert(len <= history_.size());
    for (unsigned i = 0; i < len; ++i) {
      r = r.and_then([this, i](Archive& ar) { return ar & history_[i]; });
    }
    return r;
  }
};

struct ComputeBlocking {
  MessageHeader header_ = {kMagic, 0, MessageType::COMPUTE_BLOCKING};
  uint16_t computation_id_{};
  unsigned char priority_{};
  std::vector<InputPosition> inputs_{};

  template <typename Archive>
  Archive::ResultType Serialize(Archive& ar,
                                [[maybe_unused]] const unsigned version) {
    typename Archive::ResultType r{ar};
    r = Archive::is_saving ? ar & header_ : r;
    r = r.and_then([this](Archive& ar) { return ar & computation_id_; });
    r = r.and_then([this](Archive& ar) { return ar & priority_; });
    r = r.and_then([this](Archive& ar) { return ar & inputs_; });
    return r;
  }
};

struct NetworkResult {
  float value_{};
  float draw_{};
  float moves_left_{};
  std::span<float> policy_{};

  template <typename Archive>
  Archive::ResultType Serialize(Archive& ar,
                                [[maybe_unused]] const unsigned version) {
    auto r = ar & value_;
    r = r.and_then([this](Archive& ar) { return ar & draw_; });
    r = r.and_then([this](Archive& ar) { return ar & moves_left_; });
    r = r.and_then([this](Archive& ar) { return ar & policy_; });
    return r;
  }
};

struct ComputeBlockingReply {
  MessageHeader header_ = {kMagic, 0, MessageType::COMPUTE_BLOCKING_REPLY};
  uint16_t computation_id_{};
  std::vector<NetworkResult> results_{};
  std::string_view error_message_{};

  template <typename Archive>
  Archive::ResultType Serialize(Archive& ar,
                                [[maybe_unused]] const unsigned version) {
    typename Archive::ResultType r{ar};
    r = Archive::is_saving ? ar & header_ : r;
    r = r.and_then([this](Archive& ar) { return ar & computation_id_; });
    r = r.and_then([this](Archive& ar) { return ar & results_; });
    r = r.and_then([this](Archive& ar) { return ar & error_message_; });
    return r;
  }
};

static constexpr size_t kMaxSearchThreads = 2;
static constexpr size_t kMaxMinibatchSizes = 1024;
static constexpr size_t kLegalMovesInPosition = 64;
static constexpr size_t kMaxBytesPerPosition =
    sizeof(float) * kLegalMovesInPosition;
static constexpr size_t kMaxMessageSize =
    kMaxBytesPerPosition * kMaxMinibatchSizes + sizeof(ComputeBlockingReply);
static constexpr size_t kBufferSize = std::bit_ceil(kMaxMessageSize);

using InputBufferType = std::array<char, kBufferSize>;

template <typename Archive, typename T>
[[nodiscard]]
Archive::ResultType SerializeMessage(Archive& oa, T& message);

template <typename Archive>
[[nodiscard]]
Archive::ResultType ParseMessageHeader(Archive& ia, MessageHeader& header);

template <typename Archive, typename T>
[[nodiscard]]
Archive::ResultType ParseMessageType(Archive& ia, const MessageHeader& header,
                                     T& out);

template <typename Archive, typename Callback>
[[nodiscard]]
Archive::ResultType ParseMessage(Archive& ia, const MessageHeader& header,
                                 Callback&& callback) {
  TRACE << "Parsing message " << ia.Size() << " bytes available";
  assert(ia.Size() >= header.size_);
  typename Archive::ResultType r;
  switch (header.type_) {
    case MessageType::HANDSHAKE: {
      Handshake msg;
      if (!(r = ParseMessageType(ia, header, msg))) {
        TRACE << "Parsed Handshake message failed " << r.error();
        return r;
      }
      if (callback(msg)) {
        return Unexpected{ArchiveError::InvalidData};
      }
      break;
    }
    case MessageType::HANDSHAKE_REPLY: {
      HandshakeReply msg;
      if (!(r = ParseMessageType(ia, header, msg))) {
        TRACE << "Parsed HandshakeReply message failed " << r.error();
        return r;
      }
      if (callback(msg)) {
        return Unexpected{ArchiveError::InvalidData};
      }
      break;
    }
    case MessageType::COMPUTE_BLOCKING: {
      ComputeBlocking msg;
      if (!(r = ParseMessageType(ia, header, msg))) {
        TRACE << "Parsed ComputeBlocking message failed " << r.error();
        return r;
      }
      if (callback(msg)) {
        return Unexpected{ArchiveError::InvalidData};
      }
      break;
    }
    case MessageType::COMPUTE_BLOCKING_REPLY: {
      ComputeBlockingReply msg;
      if (!(r = ParseMessageType(ia, header, msg))) {
        TRACE << "Parsed ComputeBlockingReply message failed " << r.error();
        return r;
      }
      if (callback(msg)) {
        return Unexpected{ArchiveError::InvalidData};
      }
      break;
    }
    default:
      CERR << "Unknown message type received: " << header.type_;
      return Unexpected{ArchiveError::InvalidData};
  }
  return r;
}

#ifdef ASIO_HAS_LOCAL_SOCKETS
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

template <typename SocketType>
class Connection {
 public:
  explicit Connection(SocketType&& socket) : socket_(std::move(socket)) {
    input_.resize(kMaxMessageSize);
    if (std::is_same_v<SocketType, asio::ip::tcp::socket> && socket_.is_open()) {
      socket_.set_option(asio::ip::tcp::no_delay(true));
    }
  }
  virtual ~Connection() = default;

 protected:
  template <bool new_request = true, typename MessageCallback>
  void ReadHeader(MessageCallback&& callback) {
    if (new_request) {
      if (ParseHeader(std::move(callback))) return;
      if (ReadyBytes() > 0) {
        // Move unparsed data to the front.
        std::memmove(input_.data(), input_.data() + parsed_bytes_,
                     ReadyBytes());
      }
      input_read_bytes_ = ReadyBytes();
      parsed_bytes_ = 0;
    }
    socket_.async_read_some(
        InputBuffer(), [this, callback = std::move(callback)](
                           std::error_code ec, size_t length) {
          if (ec) {
            // Handle error or disconnection.
            if (ec != asio::error::eof &&
                ec != asio::error::operation_aborted) {
              CERR << "Connection error in ReadHeader: " << ec.message();
            }
            Close();
            return;
          }
          input_read_bytes_ += length;
          // Process data.
          if (!ParseHeader(std::move(callback))) {
            ReadBody<false>(std::move(callback));
          }
        });
  }

  template <typename SelfType, typename MessageType>
  void SendMessage(SelfType&& self, MessageType& message) {
    BinaryOArchive output(kBackendApiVersion);
    if (!SerializeMessage(output, message)) {
      CERR << "Error serializing message for sending.";
      Close();
      return;
    }
    Dispatch([this, self = std::move(self), output = std::move(output)] {
      bool empty = queue_.empty();
      TRACE << "Queueing message<" << typeid(MessageType).name() << "> of size "
            << output.Size() << " for sending. Queue size was "
            << queue_.size();
      queue_.push(std::move(output));
      if (!empty) {
        // A write is already in progress.
        return;
      }
      Write(std::move(self));
    });
  }

  template <typename FunctionType>
  void Dispatch(FunctionType&& func) {
    asio::dispatch(socket_.get_executor(), std::forward<FunctionType>(func));
  }

  template <typename FunctionType>
  void Defer(FunctionType&& func) {
    asio::defer(socket_.get_executor(), std::forward<FunctionType>(func));
  }

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

  template <typename Endpoint>
  void Connect(const Endpoint& endpoint) {
    socket_.connect(endpoint);
    if constexpr (std::is_same_v<SocketType, asio::ip::tcp::socket>) {
      socket_.set_option(asio::ip::tcp::no_delay(true));
    }
  }

  virtual void Close() {
    if (!socket_.is_open()) {
      return;
    }
    socket_.shutdown(SocketType::shutdown_both);
    socket_.close();
  }

 private:
  template <bool new_request = true, typename MessageCallback>
  void ReadBody(MessageCallback&& callback) {
    if (ReadyBytes() >= pending_header_.size_) {
      ParseInput(std::move(callback));
      return;
    }
    if (new_request && pending_header_.size_ >= input_.size() - parsed_bytes_) {
      if (ReadyBytes() > 0) {
        // Move unparsed data to the front.
        std::memmove(input_.data(), input_.data() + parsed_bytes_,
                     ReadyBytes());
      }
      input_read_bytes_ = ReadyBytes();
      parsed_bytes_ = 0;
    }
    socket_.async_read_some(
        InputBuffer(), [this, callback = std::move(callback)](
                           std::error_code ec, size_t length) {
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
        });
  }

  const char* InputBegin() { return input_.data() + parsed_bytes_; }
  const char* InputEnd() { return input_.data() + input_read_bytes_; }

  auto InputBuffer() {
    return asio::buffer(input_.data() + input_read_bytes_,
                        input_.size() - input_read_bytes_);
  }

  template <typename MessageCallback>
  bool ParseHeader(MessageCallback&& callback) {
    TRACE << "Parsing header data " << ReadyBytes() << " bytes available. ";
    if (ReadyBytes() == 0) return false;

    auto first = InputBegin();
    BinaryIArchive ia(std::span<const char>(first, InputEnd()), input_.data(),
                      kBackendApiVersion);
    size_t size = ia.Size();
    if (!ParseMessageHeader(ia, pending_header_)) {
      return false;
    }
    parsed_bytes_ += size - ia.Size();
    ReadBody(std::forward<MessageCallback>(callback));
    return true;
  }

  template <typename MessageCallback>
  void ParseInput(MessageCallback&& callback) {
    TRACE << "Parsing client data " << ReadyBytes() << " bytes available. "
          << "Expected " << pending_header_.size_ << " bytes. ";

    assert(ReadyBytes() >= pending_header_.size_);
    auto first = InputBegin();
    BinaryIArchive ia(std::span<const char>(first, InputEnd()), input_.data(),
                      kBackendApiVersion);
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

  size_t ReadyBytes() const { return input_read_bytes_ - parsed_bytes_; }

  std::queue<BinaryOArchive> queue_;

  SocketType socket_;

  size_t input_read_bytes_ = 0;
  size_t parsed_bytes_ = 0;
  MessageHeader pending_header_;
  std::vector<char> input_;
};

}  // namespace lczero::client

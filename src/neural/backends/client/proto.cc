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

#include "proto.h"

namespace lczero::client {

namespace {

template <typename T>
int read(T& out, const char*& data, size_t& left) {
  if (sizeof(T) > left) {
    CERR << "Not enough data to read type of size " << sizeof(T);
    return -1;
  }
  std::memcpy(&out, data, sizeof(T));
  data += sizeof(T);
  left -= sizeof(T);
  return 0;
}
template <typename T>
int write(char*& data, const T& value, size_t& left) {
  if (sizeof(T) > left) {
    CERR << "Not enough space to write type of size " << sizeof(T);
    return -1;
  }
  std::memcpy(data, &value, sizeof(T));
  data += sizeof(T);
  left -= sizeof(T);
  return 0;
}
template <typename T>
int write(OutputBufferType& os, const T& value, size_t& offset) {
  size_t reserve = sizeof(T);
  if (os.size() < offset + reserve) {
    TRACE << "Resizing output buffer to " << (offset + reserve);
    os.resize(offset + reserve);
  }
  char* data = os.data() + offset;
  size_t left = os.size() - offset;
  offset += reserve;
  return write(data, value, left);
}

template <typename T>
int read(std::span<T>& out, const char*& data, size_t& left);
template <typename T>
int write(OutputBufferType& os, const std::span<T>& value, size_t& offset);

template <>
int read<std::string_view>(std::string_view& out, const char*& data,
                           size_t& left) {
  TRACE << "Reading string from data, " << left << " bytes left";
  uint16_t str_len;
  if (read(str_len, data, left)) {
    CERR << "Failed to read string length";
    return -1;
  }
  if (str_len > left) {
    CERR << "Not enough data to read string of size " << str_len;
    return -1;
  }
  out = std::string_view(data, str_len);
  data += str_len;
  left -= str_len;
  return 0;
}
template <>
int write<std::string_view>(OutputBufferType& os, const std::string_view& value,
                            size_t& offset) {
  size_t reserve = sizeof(uint16_t) + value.size();
  if (os.size() < offset + reserve) {
    TRACE << "Resizing output buffer to " << (offset + reserve);
    os.resize(offset + reserve);
  }
  char* data = os.data() + offset;
  size_t left = os.size() - offset;
  TRACE << "Writing string of size " << value.size() << ", " << left
        << " bytes left";
  uint16_t str_len = static_cast<uint16_t>(value.size());
  if (write(data, str_len, left)) {
    CERR << "Failed to write string length";
    return -1;
  }
  if (str_len > left) {
    CERR << "Not enough space to write string of size " << str_len;
    return -1;
  }
  std::memcpy(data, value.data(), str_len);
  offset += reserve;
  return 0;
}

template <>
int read<NetworkResult>(NetworkResult& out, const char*& data, size_t& left) {
  TRACE << "Reading NetworkResult from data, " << left << " bytes left";
  if (read(out.value_, data, left)) {
    CERR << "Failed to read NetworkResult value";
    return -1;
  }
  if (read(out.draw_, data, left)) {
    CERR << "Failed to read NetworkResult draw";
    return -1;
  }
  if (read(out.moves_left_, data, left)) {
    CERR << "Failed to read NetworkResult moves_left";
    return -1;
  }
  if (read(out.policy_, data, left)) {
    CERR << "Failed to read NetworkResult policy";
    return -1;
  }
  return 0;
}
template <>
int write<NetworkResult>(OutputBufferType& os, const NetworkResult& value,
                        size_t& offset) {
  TRACE << "Writing NetworkResult to output buffer";
  if (write(os, value.value_, offset)) {
    CERR << "Failed to write NetworkResult value";
    return -1;
  }
  if (write(os, value.draw_, offset)) {
    CERR << "Failed to write NetworkResult draw";
    return -1;
  }
  if (write(os, value.moves_left_, offset)) {
    CERR << "Failed to write NetworkResult moves_left";
    return -1;
  }
  if (write(os, value.policy_, offset)) {
    CERR << "Failed to write NetworkResult policy";
    return -1;
  }
  return 0;
}

template <typename T>
int read(std::span<T>& out, const char*& data, size_t& left) {
  TRACE << "Reading span<" << typeid(T).name() << "> from data, " << left
        << " bytes left";
  uint16_t vec_len;
  if (read(vec_len, data, left)) {
    CERR << "Failed to read span length";
    return -1;
  }

  TRACE << "Span length: " << vec_len;

  size_t byte_size = vec_len * sizeof(T);
  if (byte_size > left) {
    CERR << "Not enough data to read span of size " << vec_len;
    return -1;
  }
  T* first = reinterpret_cast<T*>(const_cast<char*>(data));
  T* last = first + vec_len;
  out = std::span<T>(first, last);
  data += byte_size;
  left -= byte_size;
  return 0;
}
template <typename T>
int write(OutputBufferType& os, const std::span<T>& value, size_t& offset) {
  size_t reserve = sizeof(uint16_t) + value.size() * sizeof(T);
  if (os.size() < offset + reserve) {
    TRACE << "Resizing output buffer to " << (offset + reserve);
    os.resize(offset + reserve);
  }
  char* data = os.data() + offset;
  size_t left = os.size() - offset;
  TRACE << "Writing span<" << typeid(T).name() << "> of size " << value.size()
        << ", " << left;
  uint16_t vec_len = static_cast<uint16_t>(value.size());
  if (write(data, vec_len, left)) {
    CERR << "Failed to write span length";
    return -1;
  }
  size_t byte_size = vec_len * sizeof(T);
  if (byte_size > left) {
    CERR << "Not enough space to write span of size " << vec_len;
    return -1;
  }
  std::memcpy(data, value.data(), byte_size);
  offset += reserve;
  return 0;
}

template <typename T>
int read(std::vector<T>& out, const char*& data, size_t& left) {
  TRACE << "Reading vector<" << typeid(T).name() << "> from data, " << left
        << " bytes left";
  uint16_t vec_len;
  if (read(vec_len, data, left)) {
    CERR << "Failed to read vector length";
    return -1;
  }

  TRACE << "Vector length: " << vec_len;

  out.resize(vec_len);
  for (auto& item : out) {
    if (read(item, data, left)) {
      CERR << "Failed to read vector item";
      return -1;
    }
  }
  return 0;
}
template <typename T>
int write(OutputBufferType& os, const std::vector<T>& value, size_t& offset) {
  size_t reserve = sizeof(uint16_t);
  if (os.size() < offset + reserve) {
    TRACE << "Resizing output buffer to " << (offset + reserve);
    os.resize(offset + reserve);
  }
  char* data = os.data() + offset;
  size_t left = os.size() - offset;
  TRACE << "Writing vector<" << typeid(T).name() << "> of size " << value.size()
        << ", " << left;
  uint16_t vec_len = static_cast<uint16_t>(value.size());
  if (write(data, vec_len, left)) {
    CERR << "Failed to write vector length";
    return -1;
  }
  offset += reserve;
  for (const auto& item : value) {
    if (write(os, item, offset)) {
      CERR << "Failed to write vector item";
      return -1;
    }
  }
  return 0;
}

int ParseMessage(const char*& first, const char* const last, Handshake& out) {
  TRACE << "Parsing Handshake message";
  size_t left = std::distance(first, last);
  if (read(out.backend_api_version_, first, left)) {
    return -1;
  }
  TRACE << "Backend API version: " << out.backend_api_version_ << " left "
        << left;
  return read(out.network_name_, first, left);
}
int SerializeMessage(OutputBufferType& os, const Handshake& message,
                     size_t& offset) {
  TRACE << "Serializing Handshake message";
  if (write(os, message.backend_api_version_, offset)) {
    return -1;
  }
  return write(os, message.network_name_, offset);
}

int ParseMessage(const char*& first, const char* const last,
                 HandshakeReply& out) {
  TRACE << "Parsing HandshakeReply message";
  size_t left = std::distance(first, last);
  if (read(out.attributes_, first, left)) {
    return -1;
  }
  return read(out.error_message_, first, left);
}
int SerializeMessage(OutputBufferType& os, const HandshakeReply& message,
                     size_t& offset) {
  TRACE << "Serializing HandshakeReply message";
  if (write(os, message.attributes_, offset)) {
    return -1;
  }
  return write(os, message.error_message_, offset);
}

int ParseMessage(const char*& first, const char* const last,
                 ComputeBlocking& out) {
  TRACE << "Parsing ComputeBlocking message";
  size_t left = std::distance(first, last);
  if (read(out.computation_id_, first, left)) {
    return -1;
  }
  return read(out.inputs_, first, left);
}
int SerializeMessage(OutputBufferType& os, const ComputeBlocking& message,
                     size_t& offset) {
  TRACE << "Serializing ComputeBlocking message " << message.computation_id_;
  if (write(os, message.computation_id_, offset)) {
    return -1;
  }
  return write(os, message.inputs_, offset);
}

int ParseMessage(const char*& first, const char* const last,
                 ComputeBlockingReply& out) {
  TRACE << "Parsing ComputeBlockingReply message";
  size_t left = std::distance(first, last);
  if (read(out.computation_id_, first, left)) {
    return -1;
  }
  if (read(out.results_, first, left)) {
    return -1;
  }
  return read(out.error_message_, first, left);
}
int SerializeMessage(OutputBufferType& os, const ComputeBlockingReply& message,
                     size_t& offset) {
  TRACE << "Serializing ComputeBlockingReply message "
        << message.computation_id_;
  if (write(os, message.computation_id_, offset)) {
    return -1;
  }
  if (write(os, message.results_, offset)) {
    return -1;
  }
  return write(os, message.error_message_, offset);
}

}  // namespace

int ParseMessageHeader(const char*& first, const char* const last,
                       MessageHeader& header) {
  TRACE << "Parsing MessageHeader";
  size_t left = std::distance(first, last);
  if (read(header.magic_, first, left)) {
    return -1;
  }
  if (read(header.size_, first, left)) {
    return -1;
  }
  if (read(header.type_, first, left)) {
    return -1;
  }
  if (header.magic_ != kMagic) {
    CERR << "Invalid message magic while parsing header: " << std::hex
         << header.magic_;
    return -1;
  }
  if (header.size_ > kMaxMessageSize) {
    CERR << "Message size too large: " << header.size_;
    return -1;
  }
  return 0;
}

template <typename T>
int ParseMessageType(const char*& first, const char* const last,
                     const MessageHeader& header, T& out) {
  assert(std::distance(first, last) >= header.size_);
  out.header_ = header;
  return ParseMessage(first, last, out);
}

template <typename T>
int SerializeMessage(OutputBufferType& os, const T& message) {
  size_t start = MessageHeader::Size();
  if (SerializeMessage(os, message, start)) {
    return -1;
  }
  size_t offset = 0;
  MessageHeader header = message.header_;
  header.size_ = static_cast<uint32_t>(os.size() - MessageHeader::Size());
  TRACE << "Serializing MessageHeader with size " << header.size_ << " type "
        << header.type_;
  if (write(os, header.magic_, offset)) {
    return -1;
  }
  if (write(os, header.size_, offset)) {
    return -1;
  }
  if (write(os, header.type_, offset)) {
    return -1;
  }
  return 0;
}

template int ParseMessageType<Handshake>(const char*& first,
                                         const char* const last,
                                         const MessageHeader& header,
                                         Handshake& out);
template int ParseMessageType<HandshakeReply>(const char*& first,
                                              const char* const last,
                                              const MessageHeader& header,
                                              HandshakeReply& out);

template int ParseMessageType<ComputeBlocking>(const char*& first,
                                               const char* const last,
                                               const MessageHeader& header,
                                               ComputeBlocking& out);
template int ParseMessageType<ComputeBlockingReply>(const char*& first,
                                                    const char* const last,
                                                    const MessageHeader& header,
                                                    ComputeBlockingReply& out);

template int SerializeMessage<Handshake>(OutputBufferType& os,
                                         const Handshake& message);
template int SerializeMessage<HandshakeReply>(OutputBufferType& os,
                                              const HandshakeReply& message);
template int SerializeMessage<ComputeBlocking>(OutputBufferType& os,
                                               const ComputeBlocking& message);
template int SerializeMessage<ComputeBlockingReply>(
    OutputBufferType& os, const ComputeBlockingReply& message);

}  // namespace lczero::client

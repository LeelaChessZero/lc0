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

#include "neural/backends/client/proto.h"

#include "neural/backends/client/archive.h"

namespace lczero::client {

template <typename Archive>
Archive::ResultType ParseMessageHeader(Archive& ia, MessageHeader& header) {
  TRACE << "Parsing MessageHeader";

  auto r = ia & header;
  if (!r) return r;
  if (header.magic_ != kMagic) {
    CERR << "Invalid message magic while parsing header: " << std::hex
         << header.magic_;
    return Unexpected{ArchiveError::InvalidData};
  }
  if (header.size_ > kMaxMessageSize) {
    CERR << "Message size too large: " << header.size_;
    return Unexpected{ArchiveError::InvalidData};
  }
  TRACE << "Parsed MessageHeader: size=" << header.size_
        << " type=" << static_cast<uint32_t>(header.type_);
  return r;
}

template <typename Archive, typename T>
Archive::ResultType ParseMessageType(Archive& ia, const MessageHeader& header,
                                     T& out) {
  assert(ia.Size() >= header.size_);
  out.header_ = header;
  return ia & out;
}

template <typename Archive, typename T>
Archive::ResultType SerializeMessage(Archive& os, T& message) {
  return os.StartSerialize(message);
}

template BinaryIArchive::ResultType ParseMessageHeader<BinaryIArchive>(
    BinaryIArchive& ia, MessageHeader& header);

template BinaryIArchive::ResultType ParseMessageType<BinaryIArchive, Handshake>(
    BinaryIArchive& ia, const MessageHeader& header, Handshake& out);
template BinaryIArchive::ResultType
ParseMessageType<BinaryIArchive, HandshakeReply>(BinaryIArchive& ia,
                                                 const MessageHeader& header,
                                                 HandshakeReply& out);

template BinaryIArchive::ResultType
ParseMessageType<BinaryIArchive, ComputeBlocking>(BinaryIArchive& ia,
                                                  const MessageHeader& header,
                                                  ComputeBlocking& out);
template BinaryIArchive::ResultType
ParseMessageType<BinaryIArchive, ComputeBlockingReply>(
    BinaryIArchive& ia, const MessageHeader& header, ComputeBlockingReply& out);

template BinaryOArchive::ResultType SerializeMessage<BinaryOArchive, Handshake>(
    BinaryOArchive& oa, Handshake& message);
template BinaryOArchive::ResultType
SerializeMessage<BinaryOArchive, HandshakeReply>(BinaryOArchive& oa,
                                                 HandshakeReply& message);
template BinaryOArchive::ResultType
SerializeMessage<BinaryOArchive, ComputeBlocking>(BinaryOArchive& os,
                                                  ComputeBlocking& message);
template BinaryOArchive::ResultType
SerializeMessage<BinaryOArchive, ComputeBlockingReply>(
    BinaryOArchive& os, ComputeBlockingReply& message);

}  // namespace lczero::client

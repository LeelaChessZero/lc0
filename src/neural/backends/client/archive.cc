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

#include "neural/backends/client/archive.h"

#include <ostream>
#include <type_traits>
#include <vector>

#include "neural/backends/client/proto.h"
#include "utils/bit.h"

namespace lczero::client {

namespace {

template <typename R, typename T>
auto SaveImpl(R&& successed, std::vector<char>& buffer, const T& value)
    -> std::enable_if_t<std::is_integral_v<T>, R> {
  using UnsignedT = std::make_unsigned_t<T>;
  UnsignedT uvalue = static_cast<UnsignedT>(value);
  if (!std::is_same_v<T, UnsignedT>) {
    // For signed types, use zigzag encoding.
    uvalue = (uvalue << 1) ^ static_cast<UnsignedT>(value < 0 ? -1 : 0);
  }
  do {
    char byte = static_cast<char>(uvalue & 0x7f);
    uvalue >>= 7;
    if (uvalue) byte |= 0x80;
    if (buffer.size() == buffer.capacity()) {
      return Unexpected{ArchiveError::BufferOverflow};
    }
    buffer.push_back(byte);
  } while (uvalue);
  return std::forward<R>(successed);
}
template <typename R, typename T>
auto LoadImpl(R&& successed, std::span<const char>& buffer, T& value)
    -> std::enable_if_t<std::is_integral_v<T>, R> {
  using UnsignedT = std::make_unsigned_t<T>;
  UnsignedT uvalue = 0;
  unsigned shift = 0;
  while (true) {
    if (buffer.size() == 0) {
      return Unexpected{ArchiveError::BufferOverflow};
    }
    char byte = buffer[0];
    buffer = buffer.subspan(1);
    uvalue |= static_cast<UnsignedT>(byte & 0x7f) << shift;
    if (!(byte & 0x80)) break;
    shift += 7;
    if (shift >= sizeof(UnsignedT) * 8) {
      return Unexpected{ArchiveError::ValueOverflow};
    }
  }
  if (!std::is_same_v<T, UnsignedT>) {
    // Decode zigzag encoding for signed types.
    value =
        static_cast<T>((uvalue >> 1) ^ static_cast<UnsignedT>(-(uvalue & 1)));
  } else {
    value = uvalue;
  }
  return std::forward<R>(successed);
}
template <typename R, typename T>
auto SizeImpl(R&& successed, size_t& size_archive, const T& value)
    -> std::enable_if_t<std::is_integral_v<T>, R> {
  using UnsignedT = std::make_unsigned_t<T>;
  UnsignedT uvalue = static_cast<UnsignedT>(value);
  if (!std::is_same_v<T, UnsignedT>) {
    uvalue = (uvalue << 1) ^ static_cast<UnsignedT>(value < 0 ? -1 : 0);
  }
  size_t size = 0;
  do {
    ++size;
    uvalue >>= 7;
  } while (uvalue);
  size_archive += size;
  return std::forward<R>(successed);
}

template <typename R, typename T>
auto SaveImpl(R&& successed, std::vector<char>& buffer,
              const FixedInteger<T>& value)
    -> std::enable_if_t<std::is_integral_v<T>, R> {
  T v = static_cast<T>(value.value);
  if (buffer.size() + sizeof(T) > buffer.capacity()) {
    return Unexpected{ArchiveError::BufferOverflow};
  }
  for (size_t i = 0; i < sizeof(T); ++i) {
    buffer.push_back(static_cast<char>(v & 0xff));
    if constexpr (sizeof(T) > 1) v >>= 8;
  }
  return std::forward<R>(successed);
}
template <typename R, typename T>
auto LoadImpl(R&& successed, std::span<const char>& buffer,
              FixedInteger<T> value)
    -> std::enable_if_t<std::is_integral_v<T>, R> {
  T v = 0;
  if (buffer.size() < sizeof(T)) {
    return Unexpected{ArchiveError::BufferOverflow};
  }
  for (size_t i = 0; i < sizeof(T); ++i) {
    v |= static_cast<T>(static_cast<unsigned char>(buffer[0])) << (i * 8);
    buffer = buffer.subspan(1);
  }
  value.value = v;
  return std::forward<R>(successed);
}
template <typename R, typename T>
auto SizeImpl(R&& successed, size_t& size_archive, const FixedInteger<T>&)
    -> std::enable_if_t<std::is_integral_v<T>, R> {
  size_archive += sizeof(T);
  return std::forward<R>(successed);
}

template <typename R>
auto SaveImpl(R&& successed, std::vector<char>& buffer, const bool& value)
    -> R {
  uint8_t v = value ? 1 : 0;
  return SaveImpl(std::forward<R>(successed), buffer, FixedInteger{v});
}
template <typename R>
auto LoadImpl(R&& successed, std::span<const char>& buffer, bool& value) -> R {
  uint8_t v;
  auto res = LoadImpl(std::forward<R>(successed), buffer, FixedInteger{v});
  if (!res) return res;
  value = !!v;
  return res;
}
template <typename R>
auto SizeImpl(R&& successed, size_t& size_archive, const bool&) -> R {
  size_archive += 1;
  return std::forward<R>(successed);
}

template <typename R, typename T>
auto SaveImpl(R&& successed, std::vector<char>& buffer, const T& fvalue)
    -> std::enable_if_t<std::is_floating_point_v<T>, R> {
  using IntType = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
  static_assert(sizeof(T) == sizeof(IntType), "Size mismatch");
  IntType ivalue = bit_cast<IntType>(fvalue);
  return SaveImpl(std::forward<R>(successed), buffer, FixedInteger{ivalue});
}
template <typename R, typename T>
auto LoadImpl(R&& successed, std::span<const char>& buffer, T& fvalue)
    -> std::enable_if_t<std::is_floating_point_v<T>, R> {
  using IntType = std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>;
  static_assert(sizeof(T) == sizeof(IntType), "Size mismatch");
  IntType ivalue;
  auto res = LoadImpl(std::forward<R>(successed), buffer, FixedInteger{ivalue});
  if (!res) return res;
  fvalue = bit_cast<T>(ivalue);
  return res;
}
template <typename R, typename T>
auto SizeImpl(R&& successed, size_t& size_archive, const T&)
    -> std::enable_if_t<std::is_floating_point_v<T>, R> {
  size_archive += sizeof(T);
  return std::forward<R>(successed);
}

struct BinaryOSizeArchive {
  using ResultType = Expected<BinaryOSizeArchive&, ArchiveError>;
  static constexpr bool is_loading = false;
  static constexpr bool is_saving = true;

  template <typename T>
  ResultType operator&(const T& value) {
    return Size(value);
  }

  template <typename T>
  std::enable_if_t<!std::is_integral_v<T> && !std::is_floating_point_v<T>,
                   ResultType>
  Size(const T& value) {
    return Serialize(const_cast<T&>(value), *this, kBackendApiVersion);
  }

  size_t Size() const { return 0; }

  template <typename T>
  std::enable_if_t<std::is_integral_v<T> || std::is_floating_point_v<T>,
                   ResultType>
  Size(const T& value) {
    return SizeImpl(ResultType{*this}, total_size_, value);
  }

  ResultType Size(const bool& value) {
    return SizeImpl(ResultType{*this}, total_size_, value);
  }

  template <typename T>
  std::enable_if_t<std::is_integral_v<T>, ResultType> Size(
      const FixedInteger<T>& value) {
    return SizeImpl(ResultType{*this}, total_size_, value);
  }

  template <typename T>
  ResultType Size(const std::vector<T>& value) {
    ResultType r = SizeImpl(ResultType{*this}, total_size_,
                            static_cast<uint64_t>(value.size()));
    if (!r) return r;
    for (const auto& item : value) {
      r = Size(item);
      if (!r) return r;
    }
    return r;
  }
  template <typename T>
  ResultType Size(const std::span<T>& value) {
    ResultType r = SizeImpl(ResultType{*this}, total_size_,
                            static_cast<uint64_t>(value.size()));
    if (!r) return r;
    for (const auto& item : value) {
      r = Size(item);
      if (!r) return r;
    }
    return r;
  }

  ResultType Size(const std::string_view& value) {
    auto r = SizeImpl(ResultType{*this}, total_size_,
                      static_cast<uint64_t>(value.size()));
    if (!r) return r;
    total_size_ += value.size();
    return r;
  }

  size_t total_size_ = 0;
};

}  // namespace

std::ostream& operator<<(std::ostream& os, ArchiveError error) {
  switch (error) {
    case ArchiveError::None:
      os << "None";
      break;
    case ArchiveError::BufferOverflow:
      os << "BufferOverflow";
      break;
    case ArchiveError::ValueOverflow:
      os << "ValueOverflow";
      break;
    case ArchiveError::SizeCalculationFailed:
      os << "SizeCalculationFailed";
      break;
    case ArchiveError::InvalidData:
      os << "InvalidData";
      break;
    case ArchiveError::UnknownType:
      os << "UnknownType";
      break;
    case ArchiveError::RemoteError:
      os << "RemoteError";
      break;
  }
  return os;
}

BinaryOArchive::ResultType BinaryOArchive::Save(const uint64_t& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(const int64_t& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(const uint32_t& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(const int32_t& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(const uint16_t& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(const int16_t& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(const uint8_t& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(const int8_t& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}

BinaryOArchive::ResultType BinaryOArchive::Save(
    const FixedInteger<uint64_t>& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(
    const FixedInteger<int64_t>& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(
    const FixedInteger<uint32_t>& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(
    const FixedInteger<int32_t>& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(
    const FixedInteger<uint16_t>& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(
    const FixedInteger<int16_t>& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(
    const FixedInteger<uint8_t>& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(
    const FixedInteger<int8_t>& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}

BinaryOArchive::ResultType BinaryOArchive::Save(const bool& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}

BinaryOArchive::ResultType BinaryOArchive::Save(const double& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(const float& value) {
  return SaveImpl(ResultType{*this}, buffer_, value);
}
BinaryOArchive::ResultType BinaryOArchive::Save(const std::string_view& value) {
  ResultType r = Save(static_cast<uint64_t>(value.size()));
  if (!r) return r;
  if (buffer_.size() + value.size() > buffer_.capacity()) {
    return Unexpected{ArchiveError::BufferOverflow};
  }
  buffer_.insert(buffer_.end(), value.begin(), value.end());
  return r;
}

template <typename T>
BinaryOArchive::ResultType BinaryOArchive::StartSerialize(T& message) {
  BinaryOSizeArchive header_size;
  BinaryOSizeArchive size_archive;
  if (!(header_size & message.header_)) {
    return Unexpected{ArchiveError::SizeCalculationFailed};
  }
  if (!(size_archive & message)) {
    return Unexpected{ArchiveError::SizeCalculationFailed};
  }
  size_t size = size_archive.total_size_ - header_size.total_size_;
  message.header_.size_ = size;
  while (size >= 0x80) {
    size_archive.total_size_++;
    size >>= 7;
  }
  buffer_.reserve(size_archive.total_size_);
  return Save(message);
}

template BinaryOArchive::ResultType BinaryOArchive::StartSerialize<Handshake>(
    Handshake& message);
template BinaryOArchive::ResultType
BinaryOArchive::StartSerialize<HandshakeReply>(HandshakeReply& message);
template BinaryOArchive::ResultType
BinaryOArchive::StartSerialize<ComputeBlocking>(ComputeBlocking& message);
template BinaryOArchive::ResultType BinaryOArchive::StartSerialize<
    ComputeBlockingReply>(ComputeBlockingReply& message);

BinaryIArchive::ResultType BinaryIArchive::Load(uint64_t& value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(int64_t& value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(uint32_t& value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(int32_t& value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(uint16_t& value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(int16_t& value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(uint8_t& value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(int8_t& value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(FixedInteger<uint64_t> value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(FixedInteger<int64_t> value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(FixedInteger<uint32_t> value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(FixedInteger<int32_t> value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(FixedInteger<uint16_t> value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(FixedInteger<int16_t> value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(FixedInteger<uint8_t> value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(FixedInteger<int8_t> value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(bool& value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(double& value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(float& value) {
  return LoadImpl(ResultType{*this}, buffer_, value);
}
BinaryIArchive::ResultType BinaryIArchive::Load(std::string_view& value) {
  uint64_t size;
  auto res = Load(size);
  if (!res) return res;
  if (buffer_.size() < size) {
    return Unexpected{ArchiveError::BufferOverflow};
  }
  value = std::string_view(buffer_.data(), size);
  buffer_ = buffer_.subspan(size);
  return res;
}

}  // namespace lczero::client

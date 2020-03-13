#include "utils/protomessage.h"

#include "utils/exception.h"

namespace lczero {
namespace {

uint64_t ReadVarInt(const char** iter, const char* const end) {
  uint64_t res = 0;
  uint64_t multiplier = 1;
  while (*iter < end) {
    unsigned char x = **iter;
    ++*iter;
    res += (x & 0x7f) * multiplier;
    if ((x & 0x80) == 0) return res;
    multiplier *= 0x80;
  }
  throw Exception("The file seems truncated.");
}

std::string EncodeVarInt(std::uint64_t val) {
  std::string res;
  while (true) {
    char c = (val & 0x7f);
    val >>= 7;
    if (val) c |= 0x80;
    res += c;
    if (!val) return res;
  }
}

}  // namespace

void ProtoMessage::ParseFromString(const std::string& str) {
  // Making the buffer "owned", e.g. copy the string contents.
  buffer_ = str;
  data_ = buffer_;
  RebuildOffsets();
}

ProtoMessage::ProtoMessage(std::string_view serialized_proto)
    : data_(serialized_proto) {
  // Not owned.
  RebuildOffsets();
}

void ProtoMessage::RebuildOffsets() {
  // Builds offsets, e.g. mapping from wire file id to list of field offsets in
  // data_.
  offsets_.clear();
  const char* const begin = data_.data();
  const char* iter = data_.data();
  const char* const end = data_.data() + data_.size();
  while (iter < end) {
    uint64_t field_id = ReadVarInt(&iter, end);
    auto offset = iter;
    switch (field_id & 0x7) {
      case 0:
        // Varint field, so read one more varint.
        ReadVarInt(&iter, end);
        break;
      case 1:
        // Fixed64, skip 8 bytes.
        iter += 8;
        break;
      case 2: {
        // String/submessage. Varint length and then buffer of that length.
        size_t size = ReadVarInt(&iter, end);
        iter += size;
        break;
      }
      case 5:
        // Fixed32, skip 4 bytes.
        iter += 4;
        break;
      default:
        throw Exception("The file seems to be unparseable.");
    }
    offsets_[field_id].push_back({static_cast<size_t>(offset - begin),
                                  static_cast<size_t>(iter - offset)});
  }
  if (iter != end) {
    throw Exception("The file is truncated.");
  }
}

void ProtoMessage::operator=(ProtoMessage&& other) {
  buffer_ = std::move(other.buffer_);
  offsets_ = std::move(other.offsets_);
  if (!buffer_.empty()) {
    // If owned, make data_ point to a new buffer_ (the underlying data was
    // probably moved though, so probably data_ == other.data_.
    data_ = buffer_;
  } else {
    // Not owned, copy buffer.
    data_ = std::move(other.data_);
  }
}

ProtoMessage::ProtoMessage(ProtoMessage&& other) {
  operator=(std::move(other));
}

size_t ProtoMessage::WireFieldCount(int wire_field_id) const {
  auto iter = offsets_.find(wire_field_id);
  if (iter == offsets_.end()) return 0;
  return iter->second.size();
}

const char* ProtoMessage::GetFieldPtr(int wire_field_id, size_t index) const {
  auto iter = offsets_.find(wire_field_id);
  if (iter == offsets_.end()) return nullptr;
  if (index == kLast) return data_.data() + iter->second.back().offset;
  return data_.data() + iter->second.at(index).offset;
}

std::uint64_t ProtoMessage::GetVarintVal(int wire_field_id,
                                         size_t index) const {
  auto x = GetFieldPtr(wire_field_id, index);
  if (x == nullptr) return 0;
  return ReadVarInt(&x, data_.data() + data_.size());
}

float ProtoMessage::GetFloatVal(int wire_field_id, size_t index) const {
  auto x = GetFieldPtr(wire_field_id, index);
  if (x == nullptr) return 0.0f;
  float res;
  std::memcpy(&res, x, sizeof(res));
  return res;
}
double ProtoMessage::GetDoubleVal(int wire_field_id, size_t index) const {
  auto x = GetFieldPtr(wire_field_id, index);
  if (x == nullptr) return 0.0;
  double res;
  std::memcpy(&res, x, sizeof(res));
  return res;
}
std::uint32_t ProtoMessage::GetFixed32Val(int wire_field_id,
                                          size_t index) const {
  // WARNING: Doesn't support big-endian.
  auto x = GetFieldPtr(wire_field_id, index);
  if (x == nullptr) return 0;
  std::uint32_t res;
  std::memcpy(&res, x, sizeof(res));
  return res;
}
std::uint64_t ProtoMessage::GetFixed64Val(int wire_field_id,
                                          size_t index) const {
  // WARNING: Doesn't support big-endian.
  auto x = GetFieldPtr(wire_field_id, index);
  if (x == nullptr) return 0;
  std::uint64_t res;
  std::memcpy(&res, x, sizeof(res));
  return res;
}
std::string_view ProtoMessage::GetBytesVal(int wire_field_id,
                                           size_t index) const {
  auto x = GetFieldPtr(wire_field_id, index);
  if (x == nullptr) return {};
  size_t size = ReadVarInt(&x, data_.data() + data_.size());
  return std::string_view(x, size);
}

ProtoMessage::Builder::Builder(const ProtoMessage& msg) {
  for (const auto& iter : msg.offsets_) {
    auto& bucket = fields_[iter.first];
    for (const auto& entry : iter.second) {
      bucket.emplace_back(msg.data_.data() + entry.offset, entry.size);
    }
  }
}

void ProtoMessage::Builder::WireFieldSetVarint(int wire_field_id,
                                               std::uint64_t value) {
  fields_[wire_field_id] = {EncodeVarInt(value)};
}

ProtoMessage::ProtoMessage(const ProtoMessage::Builder& builder) {
  buffer_ = builder.AsString();
  data_ = buffer_;
  RebuildOffsets();
}

std::string ProtoMessage::Builder::AsString() const {
  std::string res;
  for (const auto& iter : fields_) {
    for (const auto& entry : iter.second) {
      res += EncodeVarInt(iter.first);
      res += entry;
    }
  }
  return res;
}

void ProtoMessage::Builder::WireFieldSetMessage(int wire_field_id,
                                                const ProtoMessage& msg) {
  fields_[wire_field_id] = {EncodeVarInt(msg.data_.size()) +
                            std::string(msg.data_)};
}

}  // namespace lczero
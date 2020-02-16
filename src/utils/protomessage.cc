#include "utils/protomessage.h"

#include "utils/exception.h"

namespace lczero {

void ProtoMessage::ParseFromString(const std::string& str) {
  buf_owned_ = str;
  buf_unowned_ = buf_owned_;
  RebuildOffsets();
}

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

}  // namespace

void ProtoMessage::RebuildOffsets() {
  offsets_.clear();
  const char* iter = buf_unowned_.data();
  const char* const end = buf_unowned_.data() + buf_unowned_.size();
  while (iter < end) {
    uint64_t field_id = ReadVarInt(&iter, end);
    offsets_[field_id].push_back(iter);
    switch (field_id & 0x7) {
      case 0:
        ReadVarInt(&iter, end);
        break;
      case 1:
        iter += 8;
        break;
      case 2: {
        size_t size = ReadVarInt(&iter, end);
        iter += size;
        break;
      }
      case 5:
        iter += 4;
        break;
      default:
        throw Exception("The file seems to be unparseable.");
    }
  }
  if (iter != end) {
    throw Exception("The file is truncated.");
  }
}

ProtoMessage::ProtoMessage(ProtoMessage&& other) {
  buf_owned_ = std::move(other.buf_owned_);
  if (!buf_owned_.empty() && other.buf_unowned_.data() != buf_owned_.data()) {
    buf_unowned_ = buf_owned_;
    RebuildOffsets();
  } else {
    buf_unowned_ = std::move(other.buf_unowned_);
    offsets_ = std::move(other.offsets_);
  }
}

ProtoMessage::ProtoMessage(std::string_view serialized_proto)
    : buf_unowned_(serialized_proto) {
  RebuildOffsets();
}

size_t ProtoMessage::WireFieldCount(int wire_field_id) const {
  auto iter = offsets_.find(wire_field_id);
  if (iter == offsets_.end()) return 0;
  return iter->second.size();
}

const char* ProtoMessage::GetFieldPtr(int wire_field_id, size_t index) const {
  auto iter = offsets_.find(wire_field_id);
  if (iter == offsets_.end()) return nullptr;
  if (index == kLast) return iter->second.back();
  return iter->second.at(index);
}

std::uint64_t ProtoMessage::GetVarintVal(int wire_field_id,
                                         size_t index) const {
  auto x = GetFieldPtr(wire_field_id, index);
  if (x == nullptr) return 0;
  return ReadVarInt(&x, buf_unowned_.data() + buf_unowned_.size());
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
  size_t size = ReadVarInt(&x, buf_unowned_.data() + buf_unowned_.size());
  return std::string_view(x, size);
}

}  // namespace lczero
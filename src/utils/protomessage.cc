#include "utils/protomessage.h"

#include <cstdint>

#include "utils/exception.h"

namespace lczero {

namespace {
uint64_t ReadVarInt(const std::uint8_t** iter, const std::uint8_t* const end) {
  uint64_t res = 0;
  uint64_t multiplier = 1;
  while (*iter < end) {
    std::uint8_t x = **iter;
    ++*iter;
    res += (x & 0x7f) * multiplier;
    if ((x & 0x80) == 0) return res;
    multiplier *= 0x80;
  }
  throw Exception("The file seems truncated.");
}

void CheckOutOfBounds(const std::uint8_t* const iter, size_t size,
                      const std::uint8_t* const end) {
  if (iter + size > end) {
    throw Exception("The file is truncated.");
  }
}

uint64_t ReadFixed(const std::uint8_t** iter, size_t size,
                   const std::uint8_t* const end) {
  CheckOutOfBounds(*iter, size, end);
  uint64_t multiplier = 1;
  uint64_t result = 0;
  for (; size != 0; --size, multiplier *= 256, ++*iter) {
    result += multiplier * **iter;
  }
  return result;
}

void WriteFixed(uint64_t value, size_t size, std::string* out) {
  out->reserve(out->size() + size);
  for (size_t i = 0; i < size; ++i) {
    out->push_back(static_cast<char>(static_cast<uint8_t>(value)));
    value /= 256;
  }
}

// // Kept for serialization part.
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

void ProtoMessage::ParseFromString(std::string_view str) {
  Clear();
  return MergeFromString(str);
}

void ProtoMessage::MergeFromString(std::string_view str) {
  const std::uint8_t* iter = reinterpret_cast<const std::uint8_t*>(str.data());
  const std::uint8_t* const end = iter + str.size();
  while (iter < end) {
    uint64_t wire_field_id = ReadVarInt(&iter, end);
    uint64_t field_id = wire_field_id >> 3;
    switch (wire_field_id & 0x7) {
      case 0:
        // Varint field, so read one more varint.
        SetVarInt(field_id, ReadVarInt(&iter, end));
        break;
      case 1:
        // Fixed64, read 8 bytes.
        SetInt64(field_id, ReadFixed(&iter, 8, end));
        break;
      case 2: {
        // String/submessage. Varint length and then buffer of that length.
        size_t size = ReadVarInt(&iter, end);
        CheckOutOfBounds(iter, size, end);
        SetString(field_id,
                  std::string_view(reinterpret_cast<const char*>(iter), size));
        iter += size;
        break;
      }
      case 5:
        // Fixed32, read 4 bytes.
        SetInt32(field_id, ReadFixed(&iter, 4, end));
        break;
      default:
        throw Exception("The file seems to be unparseable.");
    }
  }
}

void ProtoMessage::AppendVarInt(int field_id, std::uint64_t value,
                                std::string* out) const {
  *out += EncodeVarInt(field_id << 3);
  *out += EncodeVarInt(value);
}
void ProtoMessage::AppendInt64(int field_id, std::uint64_t value,
                               std::string* out) const {
  *out += EncodeVarInt(1 + (field_id << 3));
  WriteFixed(value, 8, out);
}
void ProtoMessage::AppendInt32(int field_id, std::uint32_t value,
                               std::string* out) const {
  *out += EncodeVarInt(5 + (field_id << 3));
  WriteFixed(value, 4, out);
}

void ProtoMessage::AppendString(int field_id, std::string_view value,
                                std::string* out) const {
  *out += EncodeVarInt(2 + (field_id << 3));
  *out += EncodeVarInt(value.size());
  *out += value;
}

}  // namespace lczero
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

// // Kept for serialization part.
// std::string EncodeVarInt(std::uint64_t val) {
//   std::string res;
//   while (true) {
//     char c = (val & 0x7f);
//     val >>= 7;
//     if (val) c |= 0x80;
//     res += c;
//     if (!val) return res;
//   }
// }

void CheckOutOfBounds(const char* const iter, size_t size,
                      const char* const end) {
  if (iter + size > end) {
    throw Exception("The file is truncated.");
  }
}

}  // namespace

void ProtoMessage::ParseFromString(std::string_view str) {
  Clear();
  return MergeFromString(str);
}

void ProtoMessage::MergeFromString(std::string_view str) {
  const char* iter = str.data();
  const char* const end = str.data() + str.size();
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
        CheckOutOfBounds(iter, 8, end);
        SetInt64(field_id, reinterpret_cast<const uint64_t*>(iter));
        iter += 8;
        break;
      case 2: {
        // String/submessage. Varint length and then buffer of that length.
        size_t size = ReadVarInt(&iter, end);
        CheckOutOfBounds(iter, size, end);
        SetString(field_id, std::string_view(iter, size));
        iter += size;
        break;
      }
      case 5:
        // Fixed32, read 4 bytes.
        CheckOutOfBounds(iter, 4, end);
        SetInt32(field_id, reinterpret_cast<const uint32_t*>(iter));
        iter += 4;
        break;
      default:
        throw Exception("The file seems to be unparseable.");
    }
  }
}

}  // namespace lczero
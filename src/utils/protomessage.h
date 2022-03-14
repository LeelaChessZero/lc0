#pragma once

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <array>
#include <vector>

// Undef g++ macros to ged rid of warnings.
#ifdef minor
#undef minor
#endif
#ifdef major
#undef major
#endif

namespace lczero {

class ProtoMessage {
 public:
  virtual ~ProtoMessage() {}
  virtual void Clear() = 0;

  void ParseFromString(std::string_view);
  void MergeFromString(std::string_view);
  virtual std::string OutputAsString() const = 0;

 protected:
  template <class To, class From>
  static To bit_cast(From from) {
    if constexpr (std::is_same_v<From, To>) {
      return from;
    } else {
      To to;
      std::memcpy(&to, &from, sizeof(to));
      return to;
    }
  }

  void AppendVarInt(int field_id, std::uint64_t value, std::string* out) const;
  void AppendInt64(int field_id, std::uint64_t value, std::string* out) const;
  void AppendInt32(int field_id, std::uint32_t value, std::string* out) const;
  void AppendString(int field_id, std::string_view value,
                    std::string* out) const;

 private:
  virtual void SetVarInt(int /* field_id */, uint64_t /* value */) {}
  virtual void SetInt64(int /* field_id */, uint64_t /* value */) {}
  virtual void SetInt32(int /* field_id */, uint32_t /* value */) {}
  virtual void SetString(int /* field_id */, std::string_view /* value */) {}
};

}  // namespace lczero
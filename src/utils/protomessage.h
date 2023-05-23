#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <string_view>
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
  virtual std::string OutputAsJson() const = 0;

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

  static void AppendVarInt(int field_id, std::uint64_t value, std::string* out);
  static void AppendInt64(int field_id, std::uint64_t value, std::string* out);
  static void AppendInt32(int field_id, std::uint32_t value, std::string* out);
  static void AppendString(int field_id, std::string_view value,
                           std::string* out);
  template <typename T>
  static void AppendJsonRepeatedField(const std::string& name,
                                      const std::vector<T>& val, bool* is_first,
                                      std::string* out) {
    AppendJsonFieldPrefix(name, is_first, out);
    out->append("[");
    for (std::size_t i = 0; i < val.size(); ++i) {
      if (i > 0) out->append(",");
      AppendJsonValue(val[i], out);
    }
    out->append("]");
  }
  template <typename T>
  static void AppendJsonField(const std::string& name, const T& val,
                              bool* is_first, std::string* out) {
    AppendJsonFieldPrefix(name, is_first, out);
    AppendJsonValue(val, out);
  }

 private:
  virtual void SetVarInt(int /* field_id */, uint64_t /* value */) {}
  virtual void SetInt64(int /* field_id */, uint64_t /* value */) {}
  virtual void SetInt32(int /* field_id */, uint32_t /* value */) {}
  virtual void SetString(int /* field_id */, std::string_view /* value */) {}

  static void AppendJsonFieldPrefix(const std::string& name, bool* is_first,
                                    std::string* out);
  static void AppendJsonValue(const std::string& val, std::string* out);
  static void AppendJsonValue(bool val, std::string* out);
  static void AppendJsonValue(double val, std::string* out);
  static void AppendJsonValue(uint64_t val, std::string* out);
  static void AppendJsonValue(int64_t val, std::string* out);
  static void AppendJsonValue(uint32_t val, std::string* out);
  static void AppendJsonValue(int32_t val, std::string* out);
  static void AppendJsonValue(const ProtoMessage& val, std::string* out);
};

}  // namespace lczero
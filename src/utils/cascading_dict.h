/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2025 The LCZero Authors

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

#include <absl/container/flat_hash_map.h>

#include <variant>

#include "utils/exception.h"

namespace lczero {

template <typename K, typename... V>
class CascadingDict {
 public:
  explicit CascadingDict(const CascadingDict* parent = nullptr)
      : parent_(parent) {}

  // e.g. dict.Get<int>(&"threads")
  // Returns value of given type. Throws exception if not found.
  template <typename T>
  T Get(const K& key) const;

  // Returns the own value of given type (doesn't fall back to querying parent).
  // Returns nullopt if doesn't exist.
  template <typename T>
  std::optional<T> OwnGet(const K& key) const;

  // Checks whether the given key exists for any type.
  bool KeyExists(const K& key) const;

  // Checks whether the given key exists for given type.
  template <typename T>
  bool Exists(const K& key) const;

  // Checks whether the given key exists for given type, and throws an exception
  // if not.
  template <typename T>
  void EnsureExists(const K& key) const;

  // Checks whether the given key exists for any type. Does not fall back to
  // check parents.
  bool OwnKeyExists(const K& key) const;

  // Checks whether the given key exists for given type. Does not fall back to
  // check parents.
  template <typename T>
  bool OwnExists(const K& key) const;

  // Returns value of given type. Returns default if not found.
  template <typename T>
  T GetOrDefault(const K& key, const T& default_val) const;

  // Sets value for a given type.
  template <typename T>
  void Set(const K& key, const T& value);

  // Get reference to assign value to. It's considered a write operation (i.e.
  // resets read-after-set flag).
  template <typename T>
  T& GetOwnRef(const K& key);

  // Returns true when the value is not set anywhere maybe except the root
  // dictionary;
  bool IsDefault(const K& key) const;

  // Returns subdictionary. Throws exception if doesn't exist.
  const CascadingDict& GetSubdict(const std::string& name) const;

  // Returns subdictionary. Throws exception if doesn't exist.
  CascadingDict* GetMutableSubdict(const std::string& name);

  // Creates subdictionary. Throws exception if already exists.
  CascadingDict* AddSubdict(const std::string& name);

  // Returns list of subdictionaries.
  std::vector<std::string> ListSubdicts() const;

  // Adds alias dictionary.
  void AddAliasDict(const CascadingDict* dict);

  // Throws an exception for the first option in the dict that has not been read
  // to find syntax errors in options added using AddSubdictFromString.
  void CheckAllOptionsRead(const std::string& path_from_parent) const;

  // Returns true if the subdictionary with the given name exists.
  bool HasSubdict(const std::string& name) const;

 private:
  struct ValueType {
    std::variant<V...> value;
    mutable bool was_read_after_set = false;
  };

  // Tries various ways to get printable string representation of the key.
  static std::string KeyAsString(const K& key);

  const CascadingDict* parent_ = nullptr;
  absl::flat_hash_map<K, ValueType> dict_;
  absl::flat_hash_map<std::string, CascadingDict> subdicts_;
  std::vector<const CascadingDict*> aliases_;
};

template <typename K, typename... V>
template <typename T>
T CascadingDict<K, V...>::Get(const K& key) const {
  for (const auto* alias : aliases_) {
    const auto value = alias->template OwnGet<T>(key);
    if (value) return *value;
  }
  if (parent_) return parent_->Get<T>(key);
  throw Exception("Key [" + KeyAsString(key) + "] was not set in options.");
};

template <typename K, typename... V>
template <typename T>
std::optional<T> CascadingDict<K, V...>::OwnGet(const K& key) const {
  const auto it = dict_.find(key);
  if (it == dict_.end()) return std::nullopt;
  if (!std::holds_alternative<T>(it->second.value)) {
    throw Exception("Key [" + KeyAsString(key) + "] is not of expected type.");
  }
  it->second.was_read_after_set = true;
  return std::get<T>(it->second.value);
}

template <typename K, typename... V>
bool CascadingDict<K, V...>::KeyExists(const K& key) const {
  for (const auto* alias : aliases_) {
    if (alias->OwnKeyExists(key)) return true;
  }
  return parent_ && parent_->KeyExists(key);
}

template <typename K, typename... V>
template <typename T>
bool CascadingDict<K, V...>::Exists(const K& key) const {
  for (const auto* alias : aliases_) {
    if (alias->template OwnExists<T>(key)) return true;
  }
  return parent_ && parent_->Exists<T>(key);
}

template <typename K, typename... V>
template <typename T>
void CascadingDict<K, V...>::EnsureExists(const K& key) const {
  if (!Exists<T>(key)) {
    throw Exception("Key [" + KeyAsString(key) + "] is not set.");
  }
}

template <typename K, typename... V>
bool CascadingDict<K, V...>::OwnKeyExists(const K& key) const {
  return dict_.find(key) != dict_.end();
}

template <typename K, typename... V>
template <typename T>
bool CascadingDict<K, V...>::OwnExists(const K& key) const {
  const auto it = dict_.find(key);
  return it != dict_.end() && std::holds_alternative<T>(it->second.value);
}

template <typename K, typename... V>
template <typename T>
T CascadingDict<K, V...>::GetOrDefault(const K& key,
                                       const T& default_val) const {
  for (const auto* alias : aliases_) {
    const auto value = alias->template OwnGet<T>(key);
    if (value) return *value;
  }
  if (parent_) return parent_->GetOrDefault<T>(key, default_val);
  return default_val;
}

template <typename K, typename... V>
template <typename T>
void CascadingDict<K, V...>::Set(const K& key, const T& value) {
  GetOwnRef<T>(key) = value;
}

template <typename K, typename... V>
template <typename T>
T& CascadingDict<K, V...>::GetOwnRef(const K& key) {
  auto it = dict_.find(key);
  if (it != dict_.end() && !std::holds_alternative<T>(it->second.value)) {
    throw Exception("Key [" + KeyAsString(key) + "] is not of expected type.");
  }
  if (it == dict_.end()) {
    it = dict_.emplace(key, ValueType{T{}}).first;
  }
  it->second.was_read_after_set = false;
  return std::get<T>(it->second.value);
}

template <typename K, typename... V>
bool CascadingDict<K, V...>::IsDefault(const K& key) const {
  if (!parent_) return true;
  for (const auto* alias : aliases_) {
    if (alias->OwnKeyExists(key)) return false;
  }
  return parent_->IsDefault(key);
}

template <typename K, typename... V>
const CascadingDict<K, V...>& CascadingDict<K, V...>::GetSubdict(
    const std::string& name) const {
  const auto it = subdicts_.find(name);
  if (it == subdicts_.end()) {
    throw Exception("Subdictionary [" + name + "] doesn't exist.");
  }
  return it->second;
}

template <typename K, typename... V>
CascadingDict<K, V...>* CascadingDict<K, V...>::GetMutableSubdict(
    const std::string& name) {
  auto it = subdicts_.find(name);
  if (it == subdicts_.end()) {
    throw Exception("Subdictionary [" + name + "] doesn't exist.");
  }
  return &it->second;
}

template <typename K, typename... V>
CascadingDict<K, V...>* CascadingDict<K, V...>::AddSubdict(
    const std::string& name) {
  auto it = subdicts_.find(name);
  if (it != subdicts_.end()) {
    throw Exception("Subdictionary [" + name + "] already exists.");
  }
  return &subdicts_.emplace(name, this).first->second;
}

template <typename K, typename... V>
std::vector<std::string> CascadingDict<K, V...>::ListSubdicts() const {
  std::vector<std::string> result;
  result.reserve(subdicts_.size());
  for (const auto& [name, _] : subdicts_) result.push_back(name);
  return result;
}

template <typename K, typename... V>
void CascadingDict<K, V...>::AddAliasDict(const CascadingDict* dict) {
  aliases_.push_back(dict);
}

template <typename K, typename... V>
void CascadingDict<K, V...>::CheckAllOptionsRead(
    const std::string& path_from_parent) const {
  std::string prefix = path_from_parent.empty() ? "" : path_from_parent + '.';
  for (const auto& [key, value] : dict_) {
    if (!value.was_read_after_set) {
      throw Exception("Option [" + prefix + KeyAsString(key) +
                      "] was not accessed after being set.");
    }
  }
  for (const auto& [name, subdict] : subdicts_) {
    subdict.CheckAllOptionsRead(prefix + name);
  }
}

template <typename K, typename... V>
bool CascadingDict<K, V...>::HasSubdict(const std::string& name) const {
  return subdicts_.find(name) != subdicts_.end();
}

// Type trait to check if T is convertible to std::string
template <typename T>
struct is_string_convertible {
  template <typename U>
  static auto test(U* u) -> decltype(std::string(*u), std::true_type{});
  static std::false_type test(...);
  static constexpr bool value = decltype(test(static_cast<T*>(nullptr)))::value;
};

// Type trait to check if T has an AsString() method
template <typename T>
struct has_as_string {
 private:
  template <typename U>
  static auto test(U* u) -> decltype(u->AsString(), std::true_type{});
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test(static_cast<T*>(nullptr)))::value;
};

// Type trait to check if std::to_string works with T
template <typename T>
struct has_std_to_string {
 private:
  template <typename U>
  static auto test(U* u) -> decltype(std::to_string(*u), std::true_type{});
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test(static_cast<T*>(nullptr)))::value;
};

template <typename K, typename... V>
std::string CascadingDict<K, V...>::KeyAsString(const K& key) {
  if constexpr (is_string_convertible<K>::value) {
    return std::string(key);
  } else if constexpr (has_as_string<K>::value) {
    return key.AsString();
  } else if constexpr (has_std_to_string<K>::value) {
    return std::to_string(key);
  } else {
    std::ostringstream oss;
    oss << &key;
    return oss.str();
  }
}

/*
// Creates options dict from string. Example of a string:
// option1=1, option_two = "string val", subdict(option3=3.14)
//
// the sub dictionary is containing a parent pointer refering
// back to this object. You need to ensure, that this object
// is still in scope, when the parent pointer is used
void AddSubdictFromString(const std::string& str);

// Throws an exception for the first option in the dict that has not been read
// to find syntax errors in options added using AddSubdictFromString.
void CheckAllOptionsRead(const std::string& path_from_parent) const;

bool HasSubdict(const std::string& name) const;

private:
static std::string GetOptionId(const OptionId& option_id) {
  return std::to_string(reinterpret_cast<intptr_t>(&option_id));
}

const OptionsDict* parent_ = nullptr;
std::map<std::string, OptionsDict> subdicts_;
// Dictionaries where to search for "own" parameters. By default contains only
// this.
std::vector<const OptionsDict*> aliases_;
};

template <typename T>
T OptionsDict::Get(const std::string& key) const {
  for (const auto* alias : aliases_) {
    const auto value = alias->OwnGet<T>(&key);
    if (value) return *value;
  }
  if (parent_) return parent_->Get<T>(&key);
  throw Exception("Key [" + key + "] was not set in options.");
}
template <typename T>
T OptionsDict::Get(const OptionId& option_id) const {
  return Get<T>(&GetOptionId(option_id));
}
template <typename T>
std::optional<T> OptionsDict::OwnGet(const std::string& key) const {
  const auto& dict = TypeDict<T>::dict();
  auto iter = dict.find(key);
  if (iter != dict.end()) {
    return iter->second.Get();
  }
  return {};
}
template <typename T>
std::optional<T> OptionsDict::OwnGet(const OptionId& option_id) const {
  return OwnGet<T>(&GetOptionId(option_id));
}

template <typename T>
bool OptionsDict::Exists(const std::string& key) const {
  for (const auto* alias : aliases_) {
    if (alias->OwnExists<T>(key)) return true;
  }
  return parent_ && parent_->Exists<T>(key);
}
template <typename T>
bool OptionsDict::Exists(const OptionId& option_id) const {
  return Exists<T>(GetOptionId(option_id));
}
template <typename T>
void OptionsDict::EnsureExists(const OptionId& option_id) const {
  if (!OwnExists<T>(option_id)) {
    throw Exception(std::string("The flag --") + option_id.long_flag() +
                    " must be specified.");
  }
}

template <typename T>
bool OptionsDict::OwnExists(const std::string& key) const {
  const auto& dict = TypeDict<T>::dict();
  auto iter = dict.find(key);
  return iter != dict.end();
}
template <typename T>
bool OptionsDict::OwnExists(const OptionId& option_id) const {
  return OwnExists<T>(GetOptionId(option_id));
}

template <typename T>
T OptionsDict::GetOrDefault(const std::string& key,
                            const T& default_val) const {
  for (const auto* alias : aliases_) {
    const auto value = alias->OwnGet<T>(&key);
    if (value) return *value;
  }
  if (parent_) return parent_->GetOrDefault<T>(key, default_val);
  return default_val;
}
template <typename T>
T OptionsDict::GetOrDefault(const OptionId& option_id,
                            const T& default_val) const {
  return GetOrDefault<T>(GetOptionId(option_id), default_val);
}

template <typename T>
void OptionsDict::Set(const std::string& key, const T& value) {
  (*TypeDict<T>::mutable_dict())[key].Set(value);
}
template <typename T>
void OptionsDict::Set(const OptionId& option_id, const T& value) {
  Set<T>(GetOptionId(option_id), value);
}

template <typename T>
T& OptionsDict::GetRef(const std::string& key) {
  return (*TypeDict<T>::mutable_dict())[key].Get();
}
template <typename T>
T& OptionsDict::GetRef(const OptionId& option_id) {
  return GetRef<T>(GetOptionId(option_id));
}

template <typename T>
bool OptionsDict::IsDefault(const std::string& key) const {
  if (!parent_) return true;
  for (const auto* alias : aliases_) {
    if (alias->OwnExists<T>(key)) return false;
  }
  return parent_->IsDefault(key);
}
template <typename T>
bool OptionsDict::IsDefault(const OptionId& option_id) const {
  return IsDefault(GetOptionId(option_id));
}

*/

}  // namespace lczero

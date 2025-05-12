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

namespace lczero {

template <typename K, typename... V>
class KeyValueTree {
 public:
  explicit KeyValueTree(const KeyValueTree* parent = nullptr)
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

  // Get reference to assign value to.
  template <typename T>
  T& GetRef(const K& key);

  // Returns true when the value is not set anywhere maybe except the root
  // dictionary;
  template <typename T>
  bool IsDefault(const K& key) const;

  // Returns subdictionary. Throws exception if doesn't exist.
  const KeyValueTree& GetSubdict(const std::string& name) const;

  // Returns subdictionary. Throws exception if doesn't exist.
  KeyValueTree* GetMutableSubdict(const std::string& name);

  // Creates subdictionary. Throws exception if already exists.
  KeyValueTree* AddSubdict(const std::string& name);

  // Returns list of subdictionaries.
  std::vector<std::string> ListSubdicts() const;

  // Adds alias dictionary.
  void AddAliasDict(const KeyValueTree* dict);

  // Returns true if the subdictionary with the given name exists.
  bool HasSubdict(const std::string& name) const;

 private:
  struct ValueType {
    std::variant<V...> value;
  };

  const KeyValueTree* parent_ = nullptr;
  absl::flat_hash_map<K, ValueType> dict_;
  absl::flat_hash_map<std::string, KeyValueTree> subdicts_;
  std::vector<const KeyValueTree*> aliases_;
};

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
  return parent_->IsDefault<T>(key);
}
template <typename T>
bool OptionsDict::IsDefault(const OptionId& option_id) const {
  return IsDefault<T>(GetOptionId(option_id));
}

*/

}  // namespace lczero

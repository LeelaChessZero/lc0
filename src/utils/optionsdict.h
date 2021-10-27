/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2020 The LCZero Authors

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

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/exception.h"

namespace lczero {

template <typename T>
class TypeDict {
 protected:
  struct V {
    const T& Get() const {
      is_used_ = true;
      return value_;
    }
    T& Get() {
      is_used_ = true;
      return value_;
    }
    void Set(const T& v) {
      is_used_ = false;
      value_ = v;
    }
    bool IsSet() const { return is_used_; }

   private:
    mutable bool is_used_ = false;
    T value_;
  };
  void EnsureNoUnusedOptions(const std::string& type_name,
                             const std::string& prefix) const {
    for (auto const& option : dict_) {
      if (!option.second.IsSet()) {
        throw Exception("Unknown " + type_name + " option: " + prefix +
                        option.first);
      }
    }
  }

  const std::unordered_map<std::string, V>& dict() const { return dict_; }
  std::unordered_map<std::string, V>* mutable_dict() { return &dict_; }

 private:
  std::unordered_map<std::string, V> dict_;
};

class OptionId {
 public:
  OptionId(const char* long_flag, const char* uci_option, const char* help_text,
           const char short_flag = '\0')
      : long_flag_(long_flag),
        uci_option_(uci_option),
        help_text_(help_text),
        short_flag_(short_flag) {}

  OptionId(const OptionId& other) = delete;
  bool operator==(const OptionId& other) const { return this == &other; }

  const char* long_flag() const { return long_flag_; }
  const char* uci_option() const { return uci_option_; }
  const char* help_text() const { return help_text_; }
  char short_flag() const { return short_flag_; }

 private:
  const char* const long_flag_;
  const char* const uci_option_;
  const char* const help_text_;
  const char short_flag_;
};

class OptionsDict : TypeDict<bool>,
                    TypeDict<int>,
                    TypeDict<std::string>,
                    TypeDict<float> {
 public:
  explicit OptionsDict(const OptionsDict* parent = nullptr)
      : parent_(parent), aliases_{this} {}

  // e.g. dict.Get<int>("threads")
  // Returns value of given type. Throws exception if not found.
  template <typename T>
  T Get(const std::string& key) const;
  template <typename T>
  T Get(const OptionId& option_id) const;

  // Returns the own value of given type (doesn't fall back to querying parent).
  // Returns nullopt if doesn't exist.
  template <typename T>
  std::optional<T> OwnGet(const std::string& key) const;
  template <typename T>
  std::optional<T> OwnGet(const OptionId& option_id) const;

  // Checks whether the given key exists for given type.
  template <typename T>
  bool Exists(const std::string& key) const;
  template <typename T>
  bool Exists(const OptionId& option_id) const;

  // Checks whether the given key exists for given type, and throws an exception
  // if not.
  template <typename T>
  void EnsureExists(const OptionId& option_id) const;

  // Checks whether the given key exists for given type. Does not fall back to
  // check parents.
  template <typename T>
  bool OwnExists(const std::string& key) const;
  template <typename T>
  bool OwnExists(const OptionId& option_id) const;

  // Returns value of given type. Returns default if not found.
  template <typename T>
  T GetOrDefault(const std::string& key, const T& default_val) const;
  template <typename T>
  T GetOrDefault(const OptionId& option_id, const T& default_val) const;

  // Sets value for a given type.
  template <typename T>
  void Set(const std::string& key, const T& value);
  template <typename T>
  void Set(const OptionId& option_id, const T& value);

  // Get reference to assign value to.
  template <typename T>
  T& GetRef(const std::string& key);
  template <typename T>
  T& GetRef(const OptionId& option_id);

  // Returns true when the value is not set anywhere maybe except the root
  // dictionary;
  template <typename T>
  bool IsDefault(const std::string& key) const;
  template <typename T>
  bool IsDefault(const OptionId& option_id) const;

  // Returns subdictionary. Throws exception if doesn't exist.
  const OptionsDict& GetSubdict(const std::string& name) const;

  // Returns subdictionary. Throws exception if doesn't exist.
  OptionsDict* GetMutableSubdict(const std::string& name);

  // Creates subdictionary. Throws exception if already exists.
  OptionsDict* AddSubdict(const std::string& name);

  // Returns list of subdictionaries.
  std::vector<std::string> ListSubdicts() const;

  // Adds alias dictionary.
  void AddAliasDict(const OptionsDict* dict);

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
    const auto value = alias->OwnGet<T>(key);
    if (value) return *value;
  }
  if (parent_) return parent_->Get<T>(key);
  throw Exception("Key [" + key + "] was not set in options.");
}
template <typename T>
T OptionsDict::Get(const OptionId& option_id) const {
  return Get<T>(GetOptionId(option_id));
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
  return OwnGet<T>(GetOptionId(option_id));
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
    const auto value = alias->OwnGet<T>(key);
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

}  // namespace lczero

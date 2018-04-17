/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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
*/

#pragma once

#include <string>
#include <unordered_map>
#include "utils/exception.h"

namespace lczero {

template <typename T>
class TypeDict {
 protected:
  std::unordered_map<std::string, T> dict_;
};

class OptionsDict : TypeDict<bool>, TypeDict<int>, TypeDict<std::string> {
 public:
  OptionsDict(OptionsDict* parent = nullptr) : parent_(parent) {}

  // e.g. dict.Get<int>("threads")
  // Returns value of given type. Throws exception if not found.
  template <typename T>
  T Get(const std::string& key) const;

  // Checks whether the given key exists for given type.
  template <typename T>
  bool Exists(const std::string& key) const;

  // Returns value of given type. Returns default if not found.
  template <typename T>
  T GetOrDefault(const std::string& key, const T& default_val) const;

  // Sets value for a given type.
  template <typename T>
  void Set(const std::string& key, const T& value);

  // Get reference to assign value to.
  template <typename T>
  T& GetRef(const std::string& key);

  // Returns true when the value is not set anywhere maybe except the root
  // dictionary;
  template <typename T>
  bool IsDefault(std::string& key) const;

 private:
  OptionsDict* parent_ = nullptr;
};

template <typename T>
T OptionsDict::Get(const std::string& key) const {
  const auto& dict = TypeDict<T>::dict_;
  auto iter = dict.find(key);
  if (iter != dict.end()) return iter->second;
  if (parent_) return parent_->Get<T>(key);
  throw Exception("Key " + key + " was not set in options.");
}

template <typename T>
bool OptionsDict::Exists(const std::string& key) const {
  const auto& dict = TypeDict<T>::dict_;
  auto iter = dict.find(key);
  if (iter != dict.end()) return true;
  if (!parent_) return false;
  return parent_->Exists<T>(key);
}

template <typename T>
T OptionsDict::GetOrDefault(const std::string& key,
                            const T& default_val) const {
  const auto& dict = TypeDict<T>::dict_;
  auto iter = dict.find(key);
  if (iter != dict.end()) return iter->second;
  if (parent_) return parent_->GetOrDefault<T>(key, default_val);
  return default_val;
}

template <typename T>
void OptionsDict::Set(const std::string& key, const T& value) {
  TypeDict<T>::dict_.emplace(key, value);
}

template <typename T>
T& OptionsDict::GetRef(const std::string& key) {
  return TypeDict<T>::dict_[key];
}

template <typename T>
bool OptionsDict::IsDefault(std::string& key) const {
  if (!parent_) return true;
  const auto& dict = TypeDict<T>::dict_;
  if (dict.find(key) != dict.end()) return false;
  return parent_->IsDefault<T>(key);
}

}  // namespace lczero
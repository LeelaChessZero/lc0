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

#include "utils/optionsdict.h"

namespace lczero {

const OptionsDict& OptionsDict::GetSubdict(const std::string& name) const {
  auto iter = subdicts_.find(name);
  if (iter == subdicts_.end())
    throw Exception("Subdictionary not found: " + name);
  return iter->second;
}

// Returns subdictionary. Throws exception if doesn't exist.
OptionsDict* OptionsDict::GetMutableSubdict(const std::string& name) {
  auto iter = subdicts_.find(name);
  if (iter == subdicts_.end())
    throw Exception("Subdictionary not found: " + name);
  return &iter->second;
}

// Creates subdictionary. Throws exception if already exists.
OptionsDict* OptionsDict::AddSubdict(const std::string& name) {
  auto iter = subdicts_.find(name);
  if (iter != subdicts_.end())
    throw Exception("Subdictionary already exists: " + name);
  return &subdicts_.emplace(name, *this).first->second;
}

// Returns list of subdictionaries.
std::vector<std::string> OptionsDict::ListSubdicts() const {
  std::vector<std::string> result;
  for (const auto& subdict : subdicts_) {
    result.emplace_back(subdict.first);
  }
  return result;
}

}  // namespace lczero
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
namespace lczero {

inline uint64_t HashCat(uint64_t hash, uint64_t value) {
  return std::hash<unsigned long long>{}(value) + 0x9e3779b9 + (hash << 6) +
         (hash >> 2);
}

inline uint64_t HashCat(std::initializer_list<uint64_t> args) {
  uint64_t hash = 0;
  for (auto x : args) hash = HashCat(hash, x);
  return hash;
}
}  // namespace lczero
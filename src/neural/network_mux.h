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

#include "neural/network.h"

namespace lczero {

// Returns a computation backend which multiplexes requests from multiple
// calls into one to compute them in batches.
std::unique_ptr<Network> MakeMuxingNetwork(std::unique_ptr<Network> parent,
                                           int threads = 1,
                                           int max_batch = 256);
}  // namespace lczero
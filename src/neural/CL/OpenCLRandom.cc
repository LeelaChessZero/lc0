/*
    This file is part of Leela Zero.
    Copyright (C) 2017 Gian-Carlo Pascutto

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <thread>

#include "OpenCLParams.h"
//#include "Utils.h"
#include "OpenCLRandom.h"

Random::Random(std::uint64_t seed) {
  if (seed == 0) {
    // C++11 doesn't guarantee *anything* about how random this is,
    // and in MinGW it isn't random at all. But we can mix it in, which
    // helps when it *is* high quality (Linux, MSVC).
    std::random_device rd;
    std::ranlux48 gen(rd());
    std::uint64_t seed1 = (gen() << 16) ^ gen();
    // If the above fails, this is one of our best, portable, bets.
    std::uint64_t seed2 = std::chrono::high_resolution_clock::
    now().time_since_epoch().count();

    std::size_t thread_id =
        std::hash<std::thread::id>()(std::this_thread::get_id());
    seed = seed1 ^ seed2 ^ (std::uint64_t)thread_id;
//    Utils::myprintf("RNG seed: 0x%lx (thread: %lu)\n", seed, thread_id);
  }
  rand_engine_.seed(seed);
}

Random& Random::GetRng(void) {
  // the rng is initialized on first GetRng call which is after the cli parsing.
  static thread_local Random rng{cfg_rng_seed};
  return rng;
}

std::uint64_t Random::operator()() {
  return RandInt<std::uint64_t>();
}

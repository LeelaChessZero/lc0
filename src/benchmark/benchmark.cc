/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

#include "benchmark/benchmark.h"
#include "mcts/search.h"

namespace lczero {
namespace {
const int kDefaultThreads = 2;

const OptionId kThreadsOptionId{"threads", "Threads",
                                "Number of (CPU) worker threads to use.", 't'};
const OptionId kNNCacheSizeId{
    "nncache", "NNCacheSize",
    "Number of positions to store in a memory cache. A large cache can speed "
    "up searching, but takes memory."};
const OptionId kNodesId{"nodes", "", "Number of nodes to run as a benchmark."};
const OptionId kMovetimeId{"movetime", "",
                           "Benchmark time allocation, in milliseconds."};
const OptionId kFenId{"fen", "", "Benchmark initial position FEN."};

}  // namespace

void Benchmark::Run() {
  OptionsParser options;
  NetworkFactory::PopulateOptions(&options);
  options.Add<IntOption>(kThreadsOptionId, 1, 128) = kDefaultThreads;
  options.Add<IntOption>(kNNCacheSizeId, 0, 999999999) = 200000;
  SearchParams::Populate(&options);

  options.Add<IntOption>(kNodesId, -1, 999999999) = -1;
  options.Add<IntOption>(kMovetimeId, -1, 999999999) = 10000;
  options.Add<StringOption>(kFenId) = ChessBoard::kStartposFen;

  if (!options.ProcessAllFlags()) return;

  try {
    auto option_dict = options.GetOptionsDict();

    auto network = NetworkFactory::LoadNetwork(option_dict);

    NodeTree tree;
    tree.ResetToPosition(option_dict.Get<std::string>(kFenId.GetId()), {});

    NNCache cache;
    cache.SetCapacity(option_dict.Get<int>(kNNCacheSizeId.GetId()));

    const auto start = std::chrono::steady_clock::now();

    SearchLimits limits;
    int visits = option_dict.Get<int>(kNodesId.GetId());
    const int movetime = option_dict.Get<int>(kMovetimeId.GetId());
    if (movetime > -1) {
      limits.search_deadline = start + std::chrono::milliseconds(movetime);
    }
    if (visits > -1) {
        limits.visits = visits;
    }

    auto search = std::make_unique<Search>(
        tree, network.get(),
        std::bind(&Benchmark::OnBestMove, this, std::placeholders::_1),
        std::bind(&Benchmark::OnInfo, this, std::placeholders::_1), limits,
        option_dict, &cache, nullptr);

    search->StartThreads(option_dict.Get<int>(kThreadsOptionId.GetId()));

    search->Wait();

    const auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> time = end - start;
    std::cout << "Benchmark final time " << time.count() << "s calculating "
              << search->GetTotalPlayouts() / time.count()
              << " nodes per second." << std::endl;
  } catch (Exception& ex) {
    std::cerr << ex.what() << std::endl;
  }
}

void Benchmark::OnBestMove(const BestMoveInfo& move) {
  std::cout << "bestmove " << move.bestmove.as_string() << std::endl;
}

void Benchmark::OnInfo(const std::vector<ThinkingInfo>& infos) {
  std::string line = "Benchmark time " + std::to_string(infos[0].time);
  line += "ms, " + std::to_string(infos[0].nodes) + " nodes, ";
  line += std::to_string(infos[0].nps) + " nps";
  if (!infos[0].pv.empty()) line += ", move " + infos[0].pv[0].as_string();
  std::cout << line << std::endl;
}

}  // namespace lczero

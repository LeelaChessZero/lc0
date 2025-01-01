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

#include "tools/benchmark.h"

#include <numeric>

#include "neural/shared_params.h"
#include "search/classic/search.h"
#include "search/classic/stoppers/factory.h"
#include "search/classic/stoppers/stoppers.h"

namespace lczero {
namespace {
const int kDefaultThreads = 2;

const OptionId kThreadsOptionId{"threads", "Threads",
                                "Number of (CPU) worker threads to use.", 't'};
const OptionId kNodesId{"nodes", "", "Number of nodes to run as a benchmark."};
const OptionId kMovetimeId{"movetime", "",
                           "Benchmark time allocation, in milliseconds."};
const OptionId kFenId{"fen", "", "Benchmark position FEN."};
const OptionId kNumPositionsId{"num-positions", "",
                               "The number of benchmark positions to test."};
}  // namespace

void Benchmark::Run() {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  options.Add<IntOption>(kThreadsOptionId, 1, 128) = kDefaultThreads;
  options.Add<IntOption>(classic::kNNCacheSizeId, 0, 999999999) = 200000;
  classic::SearchParams::Populate(&options);

  options.Add<IntOption>(kNodesId, -1, 999999999) = -1;
  options.Add<IntOption>(kMovetimeId, -1, 999999999) = 10000;
  options.Add<StringOption>(kFenId) = "";
  options.Add<IntOption>(kNumPositionsId, 1, 34) = 34;

  if (!options.ProcessAllFlags()) return;

  try {
    auto option_dict = options.GetOptionsDict();

    auto network = NetworkFactory::LoadNetwork(option_dict);

    const int visits = option_dict.Get<int>(kNodesId);
    const int movetime = option_dict.Get<int>(kMovetimeId);
    const std::string fen = option_dict.Get<std::string>(kFenId);
    int num_positions = option_dict.Get<int>(kNumPositionsId);

    std::vector<std::double_t> times;
    std::vector<std::int64_t> playouts;
    std::uint64_t cnt = 1;

    if (fen.length() > 0) {
      positions = {fen};
      num_positions = 1;
    }
    std::vector<std::string> testing_positions(
        positions.cbegin(), positions.cbegin() + num_positions);

    for (std::string position : testing_positions) {
      std::cout << "\nPosition: " << cnt++ << "/" << testing_positions.size()
                << " " << position << std::endl;

      auto stopper = std::make_unique<classic::ChainedSearchStopper>();
      if (movetime > -1) {
        stopper->AddStopper(
            std::make_unique<classic::TimeLimitStopper>(movetime));
      }
      if (visits > -1) {
        stopper->AddStopper(
            std::make_unique<classic::VisitsStopper>(visits, false));
      }

      NNCache cache;
      cache.SetCapacity(option_dict.Get<int>(classic::kNNCacheSizeId));

      classic::NodeTree tree;
      tree.ResetToPosition(position, {});

      const auto start = std::chrono::steady_clock::now();
      auto search = std::make_unique<classic::Search>(
          tree, network.get(),
          std::make_unique<CallbackUciResponder>(
              std::bind(&Benchmark::OnBestMove, this, std::placeholders::_1),
              std::bind(&Benchmark::OnInfo, this, std::placeholders::_1)),
          MoveList(), start, std::move(stopper), false, false, option_dict,
          &cache, nullptr);
      search->StartThreads(option_dict.Get<int>(kThreadsOptionId));
      search->Wait();
      const auto end = std::chrono::steady_clock::now();

      const auto time =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      times.push_back(time.count());
      playouts.push_back(search->GetTotalPlayouts());
    }

    const auto total_playouts =
        std::accumulate(playouts.begin(), playouts.end(), 0);
    const auto total_time = std::accumulate(times.begin(), times.end(), 0);
    std::cout << "\n==========================="
              << "\nTotal time (ms) : " << total_time
              << "\nNodes searched  : " << total_playouts
              << "\nNodes/second    : "
              << std::lround(1000.0 * total_playouts / (total_time + 1))
              << std::endl;
  } catch (Exception& ex) {
    std::cerr << ex.what() << std::endl;
  }
}

void Benchmark::OnBestMove(const BestMoveInfo& move) {
  std::cout << "bestmove " << move.bestmove.as_string() << std::endl;
}

void Benchmark::OnInfo(const std::vector<ThinkingInfo>& infos) {
  std::string line = "Benchmark time " + std::to_string(infos[0].time);
  line += " ms, " + std::to_string(infos[0].nodes) + " nodes, ";
  line += std::to_string(infos[0].nps) + " nps";
  if (!infos[0].pv.empty()) line += ", move " + infos[0].pv[0].as_string();
  std::cout << line << std::endl;
}

}  // namespace lczero

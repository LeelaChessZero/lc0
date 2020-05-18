/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

#include "benchmark/backendbench.h"

#include "chess/board.h"
#include "mcts/node.h"
#include "neural/factory.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {
const int kDefaultThreads = 1;

const OptionId kThreadsOptionId{"threads", "Threads",
                                "Number of (CPU) worker threads to use.", 't'};
const OptionId kBatchesId{"batches", "",
                          "Number of batches to run as a benchmark."};
const OptionId kMaxBatchSizeId{"max-batch-size", "",
                               "Maximum batch size to benchmark."};
const OptionId kFenId{"fen", "", "Benchmark initial position FEN."};
}  // namespace

void BackendBenchmark::Run() {
  OptionsParser options;
  NetworkFactory::PopulateOptions(&options);
  options.Add<IntOption>(kThreadsOptionId, 1, 128) = kDefaultThreads;

  options.Add<IntOption>(kBatchesId, 1, 999999999) = 100;
  options.Add<IntOption>(kMaxBatchSizeId, 1, 1024) = 256;
  options.Add<StringOption>(kFenId) = ChessBoard::kStartposFen;

  if (!options.ProcessAllFlags()) return;

  try {
    auto option_dict = options.GetOptionsDict();

    auto network = NetworkFactory::LoadNetwork(option_dict);

    NodeTree tree;
    tree.ResetToPosition(option_dict.Get<std::string>(kFenId), {});
    const int batches = option_dict.Get<int>(kBatchesId);

    for (int i = 1; i <= option_dict.Get<int>(kMaxBatchSizeId); i++) {
      const auto start = std::chrono::steady_clock::now();
      // TODO: support threads not equal to 1 to be able to more sensibly test
      // multiplexing backend.
      for (int j = 0; j < batches; j++) {
        // Put i copies of tree root node into computation and compute.
        auto computation = network->NewComputation();
        for (int k = 0; k < i; k++) {
          computation->AddInput(EncodePositionForNN(
              network->GetCapabilities().input_format,
              tree.GetPositionHistory(), 8, FillEmptyHistory::ALWAYS, nullptr));
        }
        computation->ComputeBlocking();
      }

      const auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double> time = end - start;
      std::cout << "Benchmark batch size " << i
                << " with inference average time "
                << time.count() / batches * 1000 << "ms - throughput "
                << i * batches / time.count() << " nps." << std::endl;
    }
  } catch (Exception& ex) {
    std::cerr << ex.what() << std::endl;
  }
}
}  // namespace lczero

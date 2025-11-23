/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020-2021 The LCZero Authors

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

#include "tools/backendbench.h"

#include "chess/board.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "search/classic/node.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {
const int kDefaultThreads = 1;

const OptionId kThreadsOptionId{"threads", "Threads",
                                "Number of (CPU) worker threads to use.", 't'};
const OptionId kBatchesId{"batches", "",
                          "Number of batches to run as a benchmark."};
const OptionId kStartBatchSizeId{"start-batch-size", "",
                                 "Start benchmark from this batch size."};
const OptionId kMaxBatchSizeId{"max-batch-size", "",
                               "Maximum batch size to benchmark."};
const OptionId kBatchStepId{"batch-step", "",
                            "Step of batch size in benchmark."};
const OptionId kHeaderOnlyOnceId{"header-only-once", "",
                                 "Print CSV header only once."};
const OptionId kFenId{"fen", "", "Benchmark initial position FEN."};

const OptionId kClippyId{"clippy", "", "Enable helpful assistant."};

void Clippy(std::string title, std::string msg3, std::string best3,
            std::string msg2, std::string best2, std::string msg,
            std::string best) {
  std::cout << "  __" << std::endl;
  std::cout << " /  \\" << std::endl;
  std::cout << " |  |    " << std::string(title.length() + 2, '_') << std::endl;
  std::cout << " +  +   | " << std::string(title.length() + 1, ' ') << "|"
            << std::endl;
  std::cout << "(@)(@) _| " << title << " |" << std::endl;
  std::cout << " |  |  \\  " << std::string(6, ' ') << msg3
            << std::string(4 - best3.length(), ' ') << best3
            << std::string(title.length() - 33, ' ') << "|" << std::endl;
  std::cout << " || |/  | " << std::string(6, ' ') << msg2
            << std::string(4 - best2.length(), ' ') << best2
            << std::string(title.length() - 33, ' ') << "|" << std::endl;
  std::cout << " || ||  | " << std::string(6, ' ') << msg
            << std::string(4 - best.length(), ' ') << best
            << std::string(title.length() - 33, ' ') << "|" << std::endl;
  std::cout << " |\\_/|  |" << std::string(title.length() + 2, '_') << "|"
            << std::endl;
  std::cout << " \\___/" << std::endl;
}
}  // namespace

void BackendBenchmark::Run() {
  OptionsParser options;
  SharedBackendParams::Populate(&options);
  options.Add<IntOption>(kThreadsOptionId, 1, 128) = kDefaultThreads;

  options.Add<IntOption>(kBatchesId, 1, 999999999) = 100;
  options.Add<IntOption>(kStartBatchSizeId, 1, 1024) = 1;
  options.Add<IntOption>(kMaxBatchSizeId, 1, 1024) = 256;
  options.Add<IntOption>(kBatchStepId, 1, 256) = 1;
  options.Add<BoolOption>(kHeaderOnlyOnceId) = false;
  options.Add<StringOption>(kFenId) = ChessBoard::kStartposFen;
  options.Add<BoolOption>(kClippyId) = false;

  if (!options.ProcessAllFlags()) return;

  try {
    auto option_dict = options.GetOptionsDict();

    auto backend = BackendManager::Get()->CreateFromParams(option_dict);
    const int threads = option_dict.Get<int>(kThreadsOptionId);

    classic::NodeTree tree;
    tree.ResetToPosition(option_dict.Get<std::string>(kFenId), {});
    EvalPosition pos{tree.GetPositionHistory().GetPositions(), {}};
    std::vector<std::thread> handles;

    // Do any backend initialization outside the loop.
    auto warm = [&]() {
      // Give GPU enough work to make it go from idle clocks to max clocks.
      for (int i = 0; i < 2; i++) {
        auto warmup = backend->CreateComputation();
        for (int j = 0; j < option_dict.Get<int>(kMaxBatchSizeId); ++j) {
          warmup->AddInput(pos, {});
        }
        warmup->ComputeBlocking();
      }
    };
    for (int t = 1; t < threads; t++) {
      handles.emplace_back(warm);
    }
    warm();
    for (auto& handle : handles) {
      handle.join();
    }
    handles.clear();

    const int batches = option_dict.Get<int>(kBatchesId);

    int best = 1;
    int best2 = 1;
    int best3 = 1;
    float best_nps = 0.0f;
    float best_nps2 = 0.0f;
    float best_nps3 = 0.0f;
    std::optional<std::chrono::time_point<std::chrono::steady_clock>> pending;
    using tp = std::chrono::time_point<std::chrono::steady_clock>;
    std::vector<std::vector<tp>> ends(threads);
    for (auto& vend : ends) {
      vend.resize(batches + 1);
    }
    std::vector<std::chrono::duration<double>> times(batches);
    std::vector<int> thread_counts(threads);
    for (int i = option_dict.Get<int>(kStartBatchSizeId);
         i <= option_dict.Get<int>(kMaxBatchSizeId);
         i += option_dict.Get<int>(kBatchStepId)) {
      handles.reserve(threads);
      std::atomic<int> j{0};

      auto compute = [&](int tid = 0) {
        int count = 0;
        auto& end = ends[tid];
        // Ignore the first batch to let GPU queue fill for stable measurements.
        while (j++ < batches) {
          // Put i copies of tree root node into computation and compute.
          auto computation = backend->CreateComputation();
          for (int k = 0; k < i; k++) {
            computation->AddInput(pos, {});
          }
          computation->ComputeBlocking();
          end[count++] = std::chrono::steady_clock::now();
        }
        thread_counts[tid] = count;
      };

      for (int t = 1; t < threads; t++) {
        handles.emplace_back(compute, t);
      }

      compute(0);
      for (auto& handle : handles) {
        handle.join();
      }

      handles.clear();

      double stddev = 0;
      double total = 0;
      int batches_done = 0;
      for (int t = 0; t < threads; t++) {
        for (int j = 1; j < thread_counts[t]; j++) {
          times[batches_done] = (ends[t][j] - ends[t][j - 1]) / threads;
          total += times[batches_done].count();
          batches_done++;
        }
      }

      double mean = total / batches_done;

      for (int j = 0; j < batches_done; j++) {
        double diff = times[j].count() - mean;
        stddev += diff * diff;
      }
      stddev = std::sqrt(stddev / (batches_done - 1));
      double cv = stddev / mean;

      std::sort(times.begin(), times.begin() + batches_done);

      mean *= 1000;

      const auto nps = i * batches_done / total;
      const auto median = batches_done % 2 == 0
                              ? 2 * i /
                                    (times[batches_done / 2 - 1].count() +
                                     times[batches_done / 2].count())
                              : i / times[batches_done / 2].count();
      if (option_dict.Get<bool>(kHeaderOnlyOnceId)
              ? i == option_dict.Get<int>(kStartBatchSizeId)
              : ((i - option_dict.Get<int>(kStartBatchSizeId)) /
                     option_dict.Get<int>(kBatchStepId) % 32 ==
                 0)) {
        std::cout << "size,"
                     " mean nps,"
                     " mean ms,"
                     "   sdev,"
                     "     cv,"
                     " max nps,"
                     "  median,"
                     " min nps,"
                  << std::endl;
      }
      // clang-format off
      std::cout << std::setw(4) << i << ","
                << std::fixed << std::setprecision(0)
                << std::setw(9) << nps << ","
                << std::defaultfloat << std::setprecision(4)
                << std::setw(8) << mean  << ","
                << std::fixed << std::setprecision(4)
                << std::setw(7) << stddev * 1000 << ","
                << std::setw(7) << cv << ","
                << std::fixed << std::setprecision(0)
                << std::setw(8) << i / times[0].count() << ","
                << std::setw(8) << median << ","
                << std::setw(8) << i / times[batches_done - 1].count()
                << std::endl;
      // clang-format on

      if (option_dict.Get<bool>(kClippyId)) {
        float nps_ingame = std::pow((nps + best_nps) / 2, 1.085);
        float nps_ingame2 = std::pow((nps + best_nps2) / 2, 1.085);
        float nps_ingame3 = std::pow((nps + best_nps3) / 2, 1.085);
        float threshold = 0.16947 * exp(-4.1695e-6 * nps_ingame * 180) + 0.02;
        float threshold2 = 0.16947 * exp(-4.1695e-6 * nps_ingame2 * 15) + 0.02;
        float threshold3 = 0.16947 * exp(-4.1695e-6 * nps_ingame3 * 1) + 0.02;

        if (nps > best_nps &&
            threshold * (i - best) * best_nps < (nps - best_nps) * best) {
          best_nps = nps;
          best = i;
          if (threshold2 * (i - best2) * best_nps2 <
              (nps - best_nps2) * best2) {
            best_nps2 = nps;
            best2 = i;
            if (threshold3 * (i - best3) * best_nps3 <
                (nps - best_nps3) * best3) {
              best_nps3 = nps;
              best3 = i;
            }
          }
          if (!pending) {
            pending = std::chrono::steady_clock::now();
          }
        }
        if (pending) {
          std::chrono::duration<double> time =
              std::chrono::steady_clock::now() - *pending;
          if (time.count() > 10) {
            Clippy("Recommended minibatch-size for this net (so far):",
                   "1s/move   (Bullet):     ", std::to_string(best3),
                   "15s/move  (Rapid):      ", std::to_string(best2),
                   "3min/move (Tournament): ", std::to_string(best));
            pending.reset();
          }
        }
      }
    }
    if (option_dict.Get<bool>(kClippyId)) {
      Clippy("Recommended minibatch-size for this net:",
             "1s/move   (Bullet):     ", std::to_string(best3),
             "15s/move  (Rapid):      ", std::to_string(best2),
             "3min/move (Tournament): ", std::to_string(best));
    }
  } catch (Exception& ex) {
    std::cerr << ex.what() << std::endl;
  }
}
}  // namespace lczero

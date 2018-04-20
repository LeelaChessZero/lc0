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

#include "selfplay/tournament.h"
#include "mcts/search.h"
#include "optionsparser.h"

namespace lczero {

namespace {
const char* kShareTrees = "Share game trees for two players";
const char* kTotalGames = "Number of games to play";
const char* kParallelGames = "Number of games to play in parallel";
const char* kGpuThreads = "Number of GPU threads";
const char* kMaxGpuBatch = "Maximum GPU batch size";
const char* kThreads = "Number of CPU threads for every game";
const char* kNnCacheSize = "NNCache size";
const char* kNetFile = "Network weights file path";
const char* kAutoDiscover = "<autodiscover>";

}  // namespace

void SelfPlayTournament::PopulateOptions(OptionsParser* options) {
  options->AddContext("player1");
  options->AddContext("player2");

  options->Add<CheckOption>(kShareTrees, "share-trees") = false;
  options->Add<SpinOption>(kTotalGames, -1, 999999, "games") = -1;
  options->Add<SpinOption>(kParallelGames, 1, 256, "parallelism") = 1;
  options->Add<SpinOption>(kGpuThreads, 1, 16, "gpu-threads") = 1;
  options->Add<SpinOption>(kMaxGpuBatch, 1, 1024, "gpu-batch") = 128;
  options->Add<SpinOption>(kThreads, 1, 8, "threads", 't') = 1;
  options->Add<SpinOption>(kNnCacheSize, 0, 999999999, "nncache") = 200000;
  options->Add<StringOption>(kNetFile, "weights", 'w') = kAutoDiscover;

  Search::PopulateUciParams(options);
}

}  // namespace lczero
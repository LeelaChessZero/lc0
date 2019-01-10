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

#pragma once

enum OpenCLTuneAlgo { kTuneAlgoStochastic = 0, kTuneAlgoSystematic };

enum OpenCLTuneEffort {
  kTuneEffortFaster = 0,
  kTuneEffortNormal,
  kTuneEffortSlower,
  kTuneEffortSlowest,
};

struct OpenCLParams {
  int gpuId = -1;
  int tune_batch_size = 1;
  
  bool verbose = false;
  bool tune_only = false;
  bool force_tune = false;
  
  OpenCLTuneAlgo tune_algo = kTuneAlgoStochastic;
  OpenCLTuneEffort tune_effort = kTuneEffortNormal;
};



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

#include <fstream>
#include "analyzer/table.h"
#include "chess/board.h"
#include "chess/callbacks.h"
#include "mcts/node.h"
#include "neural/network.h"
#include "utils/optionsparser.h"

namespace lczero {

class Analyzer {
 public:
  Analyzer();
  void Run();

 private:
  void RunOnePosition(const std::vector<Move>& position);

  void WriteToLog(const std::string& line) const;
  void WriteToTsvLog(const std::vector<std::string>& line) const;

  void InitializeNetwork();
  void OnBestMove(const BestMoveInfo& move) const;
  void OnInfo(const ThinkingInfo& info) const;
  void GatherStats(Table3d* table, const Node* root_node, std::string& col,
                   bool flip);

  std::unique_ptr<Network> network_;
  OptionsParser options_parser_;
  const OptionsDict* play_options_;
  // const OptionsDict* training_options_;
  mutable std::ofstream log_;
  mutable std::ofstream tsvlog_;
};

}  // namespace lczero
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

#include "analyzer/analyzer.h"
#include "analyzer/table.h"
#include "mcts/search.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "utils/optionsparser.h"
#include "utils/string.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace lczero {
namespace {
const char* kTsvReportStr = "Filename of the tab-separated report file";
const char* kTxtReportStr = "Filename of the text report file";
const char* kMovesStr = "Moves in UCI format, space separated";
const char* kMovesToAnalyzeStr = "Number of (last) moves to analyze";
const char* kNodesStr = "(comma separated) How many nodes to calculate";
const char* kTrainExamplesStr =
    "How many examples of training data to generate";
const char* kWeightsStr = "Network weights file path";
const char* kNnBackendStr = "NN backend to use";
const char* kNnBackendOptionsStr = "NN backend parameters";

const char* kAutoDiscover = "<autodiscover>";
}  // namespace

Analyzer::Analyzer() {
  options_parser_.AddContext("play");
  options_parser_.AddContext("training");
  play_options_ = &options_parser_.GetOptionsDict("play");
  play_options_ = &options_parser_.GetOptionsDict("training");

  options_parser_.Add<StringOption>(kTsvReportStr, "tsv-report");
  options_parser_.Add<StringOption>(kTxtReportStr, "txt-report");
  options_parser_.Add<StringOption>(kMovesStr, "moves");
  options_parser_.Add<IntOption>(kMovesToAnalyzeStr, 1, 999, "num-moves") = 4;
  options_parser_.Add<StringOption>(kNodesStr, "nodes-list") =
      "10,50,100,200,400,600,800,1200,1600,5000,10000";
  options_parser_.Add<IntOption>(kTrainExamplesStr, 1, 999,
                                 "training-examples") = 10;
  options_parser_.Add<StringOption>(kWeightsStr, "weights", 'w') =
      kAutoDiscover;

  // TODO(mooskagh) Move all that to network factory.
  const auto backends = NetworkFactory::Get()->GetBackendsList();
  options_parser_.Add<ChoiceOption>(kNnBackendStr, backends, "backend") =
      backends.empty() ? "<none>" : backends[0];
  options_parser_.Add<StringOption>(kNnBackendOptionsStr, "backend-opts");

  Search::PopulateUciParams(&options_parser_);

  // Overriding default options of search.
  auto defaults = options_parser_.GetMutableDefaultsOptions();
  defaults->Set<int>(Search::kMiniBatchSizeStr, 1);      // Minibatch = 1
  defaults->Set<bool>(Search::kSmartPruningStr, false);  // No smart pruning
  defaults->Set<bool>(Search::kVerboseStatsStr, true);   // Verbose stats
}

void Analyzer::GatherStats(Table3d* table, const Node* root_node,
                           std::string& col, bool flip) {
  uint64_t total_n = 0;
  float factor = play_options_->Get<float>(Search::kCpuctStr) *
                 std::sqrt(std::max(root_node->GetN(), 1u));
  const float parent_q = -root_node->GetQ(0, 0) -
                         play_options_->Get<float>(Search::kFpuReductionStr);

  for (Node* node : root_node->Children()) {
    const auto n = node->GetNStarted();
    total_n += n;
    const auto u = factor * node->GetU();
    const auto q = node->GetQ(parent_q, 0);
    const auto move = node->GetMove(flip).as_string();
    table->Add3dVal(col, move, "N", std::to_string(n));
    table->Add3dVal(col, move, "U", std::to_string(u));
    table->Add3dVal(col, move, "Q", std::to_string(q));
    table->Add3dVal(col, move, "U+Q", std::to_string(u + q));
  }

  for (Node* node : root_node->Children()) {
    auto n = node->GetNStarted();
    table->Add3dVal(col, node->GetMove(flip).as_string(), "N%",
                    std::to_string(static_cast<double>(n) / total_n));
  }
}

void Analyzer::RunOnePosition(const std::vector<Move>& moves) {
  NNCache cache(200000);
  NodeTree tree;
  tree.ResetToPosition(ChessBoard::kStartingFen, moves);

  auto nodeses = ParseIntList(play_options_->Get<std::string>(kNodesStr));
  std::sort(nodeses.begin(), nodeses.end());

  Table3d table;
  std::vector<std::string> cols;

  // Run search in increasing number of nodes.
  for (int nodes : nodeses) {
    WriteToLog("Nodes: " + std::to_string(nodes));
    SearchLimits limits;
    limits.visits = nodes;

    Search search(tree, network_.get(),
                  std::bind(&Analyzer::OnBestMove, this, std::placeholders::_1),
                  std::bind(&Analyzer::OnInfo, this, std::placeholders::_1),
                  limits, *play_options_, &cache);

    search.RunBlocking(1);

    auto col = std::to_string(nodes);
    cols.push_back(col);
    GatherStats(&table, tree.GetCurrentHead(), col, tree.IsBlackToMove());
    table.AddColVal(col, "bestmove", search.GetBestMove().first.as_string());
  }

  // Fetch MCTS-agnostic per-move stats P and V.
  std::vector<const Node*> nodes;
  for (Node* child : tree.GetCurrentHead()->Children()) {
    nodes.emplace_back(child);
  }
  std::sort(nodes.begin(), nodes.end(), [](const Node* a, const Node* b) {
    return a->GetNStarted() > b->GetNStarted();
  });
  std::vector<std::string> rows;
  for (const Node* node : nodes) {
    auto move = node->GetMove(tree.IsBlackToMove()).as_string();
    rows.push_back(move);
    table.AddRowVal(move, "P", std::to_string(node->GetP()));
    table.AddRowVal(move, "V", std::to_string(node->GetV()));
  }

  // Dump table to log.
  auto lines = table.RenderTable(cols, rows, {"N", "N%", "U", "Q", "U+Q"},
                                 {"P", "V"}, {"bestmove"});
  for (const auto& line : lines) WriteToTsvLog(line);
}

void Analyzer::Run() {
  if (!options_parser_.ProcessAllFlags()) return;

  // Open log files.
  if (!play_options_->Get<std::string>(kTxtReportStr).empty()) {
    log_.open(play_options_->Get<std::string>(kTxtReportStr).c_str());
  }
  if (!play_options_->Get<std::string>(kTsvReportStr).empty()) {
    tsvlog_.open(play_options_->Get<std::string>(kTsvReportStr).c_str());
  }

  // Load network
  InitializeNetwork();

  // Parse moves.
  std::vector<std::string> moves_str =
      StrSplitAtWhitespace(play_options_->Get<std::string>(kMovesStr));
  std::vector<Move> moves;
  for (const auto& move : moves_str) moves.emplace_back(move);

  for (int i = 0; i < play_options_->Get<int>(kMovesToAnalyzeStr); ++i) {
    // Write position to logs.
    std::string initial_pos;
    if (moves_str.empty()) {
      WriteToLog("Position: startpos");
      WriteToTsvLog({"Startpos."});
    } else {
      WriteToLog("Position: moves " + StrJoin(moves_str));
      WriteToTsvLog({"Moves " + StrJoin(moves_str)});
    }

    // Run Mcts at different depths.
    RunOnePosition(moves);

    // Run training several times.

    if (moves.empty()) break;
    moves.pop_back();
    moves_str.pop_back();
    WriteToTsvLog({});
  }

  // DumpFlags();
}

void Analyzer::InitializeNetwork() {
  std::string network_path = play_options_->Get<std::string>(kWeightsStr);
  std::string backend = play_options_->Get<std::string>(kNnBackendStr);
  std::string backend_options =
      play_options_->Get<std::string>(kNnBackendOptionsStr);

  std::string net_path = network_path;
  if (net_path == kAutoDiscover) {
    net_path = DiscoveryWeightsFile();
  }
  Weights weights = LoadWeightsFromFile(net_path);

  OptionsDict network_options = OptionsDict::FromString(
      backend_options, &options_parser_.GetOptionsDict());

  network_ = NetworkFactory::Get()->Create(backend, weights, network_options);
}

void Analyzer::WriteToLog(const std::string& line) const {
  std::cout << line << std::endl;
  if (log_) log_ << line << std::endl;
}

void Analyzer::WriteToTsvLog(const std::vector<std::string>& line) const {
  if (!tsvlog_) return;
  bool first = true;
  for (const auto& chunk : line) {
    if (first) {
      first = false;
    } else {
      tsvlog_ << "\t";
    }
    tsvlog_ << chunk;
  }
  tsvlog_ << std::endl;
}

void Analyzer::OnBestMove(const BestMoveInfo& move) const {
  WriteToLog("BestMove: " + move.bestmove.as_string());
}

void Analyzer::OnInfo(const ThinkingInfo& info) const {
  std::string res = "Info";
  if (info.depth >= 0) res += " depth " + std::to_string(info.depth);
  if (info.seldepth >= 0) res += " seldepth " + std::to_string(info.seldepth);
  if (info.time >= 0) res += " time " + std::to_string(info.time);
  if (info.nodes >= 0) res += " nodes " + std::to_string(info.nodes);
  if (info.score) res += " score cp " + std::to_string(*info.score);
  if (info.nps >= 0) res += " nps " + std::to_string(info.nps);

  if (!info.pv.empty()) {
    res += " pv";
    for (const auto& move : info.pv) res += " " + move.as_string();
  }
  if (!info.comment.empty()) res += " string " + info.comment;
  WriteToLog(res);
}

}  // namespace lczero

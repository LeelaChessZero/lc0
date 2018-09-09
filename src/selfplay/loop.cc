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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "selfplay/loop.h"
#include "neural/encoder.h"
#include "neural/writer.h"
#include "selfplay/tournament.h"
#include "utils/configfile.h"
#include "utils/filesystem.h"

namespace lczero {

namespace {
const char* kInteractive = "Run in interactive mode with uci-like interface";
const char* kSyzygyTablebaseStr = "List of Syzygy tablebase directories";
const char* kInputDirStr = "Directory with gzipped files in need of rescoring.";
const char* kOutputDirStr = "Directory to write rescored files.";
const char* kThreadsStr = "Number of concurrent threads to rescore with.";

void ProcessFile(const std::string& file, SyzygyTablebase* tablebase,
                 std::string outputDir) {
  // Scope to ensure reader and writer are closed before deleting source file.
  {
    TrainingDataReader reader(file);
    std::string fileName = file.substr(file.find_last_of("/\\") + 1);
    TrainingDataWriter writer(outputDir + "/" + fileName);
    std::vector<V3TrainingData> fileContents;
    V3TrainingData data;
    while (reader.ReadChunk(&data)) {
      fileContents.push_back(data);
    }
    MoveList moves;
    for (int i = 1; i < fileContents.size(); i++) {
      moves.push_back(
          DecodeMoveFromInput(PlanesFromTrainingData(fileContents[i])));
      // All moves decoded are from the point of view of the side after the move
      // so need to mirror them all to be applicable to apply to the position before.
      moves.back().Mirror();
    }
    PositionHistory history;
    int rule50ply;
    int gameply;
    ChessBoard board;
    board.SetFromFen(ChessBoard::kStartingFen, &rule50ply, &gameply);
    history.Reset(board, rule50ply, gameply);
    int last_rescore = -1;
    for (int i = 0; i < moves.size(); i++) {
      history.Append(moves[i]);
      const auto& board = history.Last().GetBoard();
      if (board.castlings().no_legal_castle() &&
          history.Last().GetNoCaptureNoPawnPly() == 0 &&
          (board.ours() + board.theirs()).count() <=
              tablebase->max_cardinality()) {
        ProbeState state;
        WDLScore wdl = tablebase->probe_wdl(history.Last(), &state);
        // Only fail state means the WDL is wrong, probe_wdl may produce correct
        // result with a stat other than OK.
        if (state != FAIL) {
          int8_t score_to_apply = 0;
          if (wdl == WDL_WIN) {
            score_to_apply = 1;
          } else if (wdl == WDL_LOSS) {
            score_to_apply = -1;
          }
          for (int j = i + 1; j > last_rescore; j--) {
            /*
            if (fileContents[j].result != score_to_apply) {
            std::cerr << "Rescoring: " << (int)fileContents[j].result << " -> "
                      << (int)score_to_apply 
                      << std::endl;
            }
            */
            fileContents[j].result = score_to_apply;
            score_to_apply = -score_to_apply;
          }
          last_rescore = i + 1;
        }
      }
    }
    for (auto chunk : fileContents) {
      writer.WriteChunk(chunk);
    }
  }
  remove(file.c_str());
}

void ProcessFiles(const std::vector<std::string>& files,
                  SyzygyTablebase* tablebase, std::string outputDir, int offset,
                  int mod) {
  for (int i = offset; i < files.size(); i += mod) {
    ProcessFile(files[i], tablebase, outputDir);
  }
}
}  // namespace

RescoreLoop::RescoreLoop() {}

RescoreLoop::~RescoreLoop() {}

void RescoreLoop::RunLoop() {
  options_.Add<StringOption>(kSyzygyTablebaseStr, "syzygy-paths", 's');
  options_.Add<StringOption>(kInputDirStr, "input", 'i');
  options_.Add<StringOption>(kOutputDirStr, "output", 'o');
  options_.Add<IntOption>(kThreadsStr, 1, 20, "threads", 't') = 1;
  SelfPlayTournament::PopulateOptions(&options_);

  if (!options_.ProcessAllFlags()) return;
  SyzygyTablebase tablebase;
  if (!tablebase.init(
          options_.GetOptionsDict().Get<std::string>(kSyzygyTablebaseStr)) ||
      tablebase.max_cardinality() < 3) {
    std::cerr << "FAILED TO LOAD SYZYGY" << std::endl;
    return;
  }
  auto inputDir = options_.GetOptionsDict().Get<std::string>(kInputDirStr);
  auto files =
      GetFileList(inputDir);
  if (files.size() == 0) {
    std::cerr << "No files to process" << std::endl;
    return;
  }
  for (int i = 0; i < files.size(); i++) {
    files[i] = inputDir + "/" + files[i];
  }
  // TODO: support threads option.
  ProcessFiles(files, &tablebase,
               options_.GetOptionsDict().Get<std::string>(kOutputDirStr), 0, 1);
}

SelfPlayLoop::SelfPlayLoop() {}

SelfPlayLoop::~SelfPlayLoop() {
  if (tournament_) tournament_->Abort();
  if (thread_) thread_->join();
}

void SelfPlayLoop::RunLoop() {
  options_.Add<BoolOption>(kInteractive, "interactive") = false;
  SelfPlayTournament::PopulateOptions(&options_);

  if (!options_.ProcessAllFlags()) return;
  if (options_.GetOptionsDict().Get<bool>(kInteractive)) {
    UciLoop::RunLoop();
  } else {
    // Send id before starting tournament to allow wrapping client to know
    // who we are.
    SendId();
    SelfPlayTournament tournament(
        options_.GetOptionsDict(),
        std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
        std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
        std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
    tournament.RunBlocking();
  }
}

void SelfPlayLoop::CmdUci() {
  SendId();
  for (const auto& option : options_.ListOptionsUci()) {
    SendResponse(option);
  }
  SendResponse("uciok");
}

void SelfPlayLoop::CmdStart() {
  if (tournament_) return;
  options_.SendAllOptions();
  tournament_ = std::make_unique<SelfPlayTournament>(
      options_.GetOptionsDict(),
      std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
      std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
  thread_ =
      std::make_unique<std::thread>([this]() { tournament_->RunBlocking(); });
}

void SelfPlayLoop::SendGameInfo(const GameInfo& info) {
  std::vector<std::string> responses;
  // Send separate resign report before gameready as client gameready parsing
  // will easily get confused by adding new parameters as both training file
  // and move list potentially contain spaces.
  if (info.min_false_positive_threshold) {
    std::string resign_res = "resign_report";
    resign_res +=
        " fp_threshold " + std::to_string(*info.min_false_positive_threshold);
    responses.push_back(resign_res);
  }
  std::string res = "gameready";
  if (!info.training_filename.empty())
    res += " trainingfile " + info.training_filename;
  if (info.game_id != -1) res += " gameid " + std::to_string(info.game_id);
  if (info.is_black)
    res += " player1 " + std::string(*info.is_black ? "black" : "white");
  if (info.game_result != GameResult::UNDECIDED) {
    res += std::string(" result ") +
           ((info.game_result == GameResult::DRAW)
                ? "draw"
                : (info.game_result == GameResult::WHITE_WON) ? "whitewon"
                                                              : "blackwon");
  }
  if (!info.moves.empty()) {
    res += " moves";
    for (const auto& move : info.moves) res += " " + move.as_string();
  }
  responses.push_back(res);
  SendResponses(responses);
}

void SelfPlayLoop::CmdSetOption(const std::string& name,
                                const std::string& value,
                                const std::string& context) {
  options_.SetOption(name, value, context);
}

void SelfPlayLoop::SendTournament(const TournamentInfo& info) {
  std::string res = "tournamentstatus";
  if (info.finished) res += " final";
  res += " win " + std::to_string(info.results[0][0]) + " " +
         std::to_string(info.results[0][1]);
  res += " lose " + std::to_string(info.results[2][0]) + " " +
         std::to_string(info.results[2][1]);
  res += " draw " + std::to_string(info.results[1][0]) + " " +
         std::to_string(info.results[1][1]);
  SendResponse(res);
}

}  // namespace lczero

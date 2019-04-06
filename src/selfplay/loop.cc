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
const OptionId kInteractiveId{
    "interactive", "", "Run in interactive mode with UCI-like interface."};
const OptionId kSyzygyTablebaseId{"syzygy-paths", "",
                                  "List of Syzygy tablebase directories"};
const OptionId kInputDirId{
    "input", "", "Directory with gzipped files in need of rescoring."};
const OptionId kOutputDirId{"output", "", "Directory to write rescored files."};
const OptionId kThreadsId{"threads", "",
                          "Number of concurrent threads to rescore with."};
const OptionId kTempId{"temperature", "",
                       "Additional temperature to apply to policy target."};
const OptionId kDistributionOffsetId{
    "dist_offset", "",
    "Additional offset to apply to policy target before temperature."};
const OptionId kMinDTZBoostId{
    "dtz_policy_boost", "",
    "Additional offset to apply to policy target before temperature for moves "
    "that are best dtz option."};

std::atomic<int> games(0);
std::atomic<int> positions(0);
std::atomic<int> rescored(0);
std::atomic<int> delta(0);
std::atomic<int> rescored2(0);
std::atomic<int> rescored3(0);
std::atomic<int> orig_counts[3];
std::atomic<int> fixed_counts[3];
std::atomic<int> policy_bump(0);
std::atomic<int> policy_nobump_total_hist[11];
std::atomic<int> policy_bump_total_hist[11];

void ProcessFile(const std::string& file, SyzygyTablebase* tablebase,
                 std::string outputDir, float distTemp, float distOffset,
                 float dtzBoost) {
  // Scope to ensure reader and writer are closed before deleting source file.
  {
    TrainingDataReader reader(file);
    std::string fileName = file.substr(file.find_last_of("/\\") + 1);
    TrainingDataWriter writer(outputDir + "/" + fileName);
    std::vector<V4TrainingData> fileContents;
    V4TrainingData data;
    while (reader.ReadChunk(&data)) {
      fileContents.push_back(data);
    }
    MoveList moves;
    for (int i = 1; i < fileContents.size(); i++) {
      moves.push_back(
          DecodeMoveFromInput(PlanesFromTrainingData(fileContents[i])));
      // All moves decoded are from the point of view of the side after the move
      // so need to mirror them all to be applicable to apply to the position
      // before.
      moves.back().Mirror();
    }
    games += 1;
    positions += fileContents.size();
    PositionHistory history;
    int rule50ply;
    int gameply;
    ChessBoard board;
    board.SetFromFen(ChessBoard::kStartposFen, &rule50ply, &gameply);
    history.Reset(board, rule50ply, gameply);
    int last_rescore = -1;
    orig_counts[fileContents[0].result + 1]++;
    fixed_counts[fileContents[0].result + 1]++;
    for (int i = 0; i < moves.size(); i++) {
      history.Append(moves[i]);
      const auto& board = history.Last().GetBoard();
      if (board.castlings().no_legal_castle() &&
          history.Last().GetNoCaptureNoPawnPly() == 0 &&
          (board.ours() | board.theirs()).count() <=
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
            if (fileContents[j].result != score_to_apply) {
              if (j == i + 1 && last_rescore == -1) {
                fixed_counts[fileContents[0].result + 1]--;
                bool flip = (i % 2) == 0;
                fixed_counts[(flip ? -score_to_apply : score_to_apply) + 1]++;
                /*
                std::cerr << "Rescoring: " << file << " "  <<
                (int)fileContents[j].result << " -> "
                          << (int)score_to_apply
                          << std::endl;
                          */
              }
              rescored += 1;
              delta += abs(fileContents[j].result - score_to_apply);
              /*
            std::cerr << "Rescoring: " << (int)fileContents[j].result << " -> "
                      << (int)score_to_apply
                      << std::endl;
                      */
            }

            fileContents[j].result = score_to_apply;
            score_to_apply = -score_to_apply;
          }
          last_rescore = i + 1;
        }
      }
    }
    board.SetFromFen(ChessBoard::kStartposFen, &rule50ply, &gameply);
    history.Reset(board, rule50ply, gameply);
    for (int i = 0; i < moves.size(); i++) {
      history.Append(moves[i]);
      const auto& board = history.Last().GetBoard();
      if (board.castlings().no_legal_castle() &&
          history.Last().GetNoCaptureNoPawnPly() != 0 &&
          (board.ours() | board.theirs()).count() <=
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
          // If the WDL result disagrees with the game outcome, make it a draw.
          // WDL draw is always draw regardless of prior moves since zero, so
          // that clearly works. Otherwise, the WDL result could be correct or
          // draw, so best we can do is change scores that don't agree, to be a
          // draw. If score was a draw this is a no-op, if it was opposite it
          // becomes a draw.
          int8_t new_score = fileContents[i + 1].result != score_to_apply
                                 ? 0
                                 : fileContents[i + 1].result;
          bool dtz_rescored = false;
          // if score is not already right, and the score to apply isn't 0, dtz
          // can let us know its definitely correct.
          if (fileContents[i + 1].result != score_to_apply &&
              score_to_apply != 0) {
            // Any repetitions in the history since last 50 ply makes it risky
            // to assume dtz is still correct.
            int steps = history.Last().GetNoCaptureNoPawnPly();
            bool no_reps = true;
            for (int i = 0; i < steps; i++) {
              if (history.GetPositionAt(history.GetLength() - i - 1)
                      .GetRepetitions() != 0) {
                no_reps = false;
                break;
              }
            }
            if (no_reps) {
              int depth = tablebase->probe_dtz(history.Last(), &state);
              if (state != FAIL) {
                // This should be able to be <= 99 safely, but I've not
                // convinced myself thats true.
                if (steps + std::abs(depth) < 99) {
                  rescored3++;
                  new_score = score_to_apply;
                  dtz_rescored = true;
                }
              }
            }
          }

          // If score is not already a draw, and its not obviously a draw, check
          // if 50 move rule has advanced so far its obviously a draw. Obviously
          // not needed if we've already proven with dtz that its a win/loss.
          if (fileContents[i + 1].result != 0 && score_to_apply != 0 &&
              !dtz_rescored) {
            int depth = tablebase->probe_dtz(history.Last(), &state);
            if (state != FAIL) {
              int steps = history.Last().GetNoCaptureNoPawnPly();
              // This should be able to be >= 101 safely, but I've not convinced
              // myself thats true.
              if (steps + std::abs(depth) > 101) {
                rescored3++;
                new_score = 0;
                dtz_rescored = true;
              }
            }
          }
          if (new_score != fileContents[i + 1].result) {
            rescored2 += 1;
            /*
          std::cerr << "Rescoring: " << (int)fileContents[j].result << " -> "
                    << (int)score_to_apply
                    << std::endl;
                    */
          }

          fileContents[i + 1].result = new_score;
        }
      }
    }
    if (distTemp != 1.0f || distOffset != 0.0f || dtzBoost != 0.0f) {
      board.SetFromFen(ChessBoard::kStartposFen, &rule50ply, &gameply);
      history.Reset(board, rule50ply, gameply);
      int move_index = 0;
      for (auto& chunk : fileContents) {
        const auto& board = history.Last().GetBoard();
        std::vector<bool> boost_probs(1858, false);
        int boost_count = 0;

        if (dtzBoost != 0.0f && board.castlings().no_legal_castle() &&
            (board.ours() | board.theirs()).count() <=
                tablebase->max_cardinality()) {
          MoveList to_boost;
          tablebase->root_probe(history.Last(), true, true, &to_boost);
          for (auto& move : to_boost) {
            boost_probs[move.as_nn_index()] = true;
          }
          boost_count = to_boost.size();
        }
        float sum = 0.0;
        int prob_index = 0;
        float preboost_sum = 0.0f;
        for (auto& prob : chunk.probabilities) {
          float offset =
              distOffset + (boost_probs[prob_index] ? (dtzBoost / boost_count): 0.0f);
          if (dtzBoost != 0.0f && boost_probs[prob_index]) {
            preboost_sum += prob;
            if (prob < 0 || std::isnan(prob))
              std::cerr << "Bump for move that is illegal????" << std::endl;
            policy_bump++;
          }
          prob_index++;
          if (prob < 0 || std::isnan(prob)) continue;
          prob = std::max(0.0f, prob + offset);
          prob = std::pow(prob, 1.0f / distTemp);
          sum += prob;
        }
        prob_index = 0;
        float boost_sum = 0.0f;
        for (auto& prob : chunk.probabilities) {
          if (dtzBoost != 0.0f && boost_probs[prob_index]) {
            boost_sum += prob / sum;
          }
          prob_index++;
          if (prob < 0 || std::isnan(prob)) continue;
          prob /= sum;
        }
        if (boost_count > 0) {
          policy_nobump_total_hist[(int)(preboost_sum * 10)]++;
          policy_bump_total_hist[(int)(boost_sum * 10)]++;
        }
        history.Append(moves[move_index]);
        move_index++;
      }
    }

    int offset = 0;
    for (auto& chunk : fileContents) {
      chunk.move_count = std::min(255, (int)(fileContents.size() - offset) / 2);
      offset++;
    }
    // Correct move_count using DTM. Since we don't actually have DTM, use DTZ
    // for 3 piece no-pawn positions only.
    board.SetFromFen(ChessBoard::kStartposFen, &rule50ply, &gameply);
    history.Reset(board, rule50ply, gameply);
    for (int i = 0; i < moves.size(); i++) {
      history.Append(moves[i]);
      const auto& board = history.Last().GetBoard();
      if (board.castlings().no_legal_castle() &&
          (board.ours() | board.theirs()).count() <= 3 &&
          board.pawns().empty()) {
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
          // No point updating for draws.
          if (score_to_apply == 0) continue;
          // Any repetitions in the history since last 50 ply makes it risky
          // to assume dtz is still correct.
          int steps = history.Last().GetNoCaptureNoPawnPly();
          bool no_reps = true;
          for (int i = 0; i < steps; i++) {
            if (history.GetPositionAt(history.GetLength() - i - 1)
                    .GetRepetitions() != 0) {
              no_reps = false;
              break;
            }
          }
          if (no_reps) {
            int depth = tablebase->probe_dtz(history.Last(), &state);
            if (state != FAIL) {
              // if depth == -1 this is wrong, since that is mate and the answer
              // should be 0, but the move before depth is -2. Since data never
              // contains mate position, ignore that discrepency.
              int converted_ply_remaining = std::abs(depth);
              // This should be able to be <= 99 safely, but I've not
              // convinced myself thats true.
              if (steps + std::abs(depth) < 99) {
                fileContents[i + 1].move_count =
                    std::min(255, converted_ply_remaining / 2);
              }
              if (steps == 0) {
                for (int j = i; j >= 0; j--) {
                  fileContents[j].move_count = std::min(
                      255, (converted_ply_remaining + (i + 1 - j)) / 2);
                }
              }
            }
          }
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
                  SyzygyTablebase* tablebase, std::string outputDir,
                  float distTemp, float distOffset, float dtzBoost, int offset,
                  int mod) {
  std::cerr << "Thread: " << offset << " starting" << std::endl;
  for (int i = offset; i < files.size(); i += mod) {
    ProcessFile(files[i], tablebase, outputDir, distTemp, distOffset, dtzBoost);
  }
}
}  // namespace

RescoreLoop::RescoreLoop() {}

RescoreLoop::~RescoreLoop() {}

void RescoreLoop::RunLoop() {
  orig_counts[0] = 0;
  orig_counts[1] = 0;
  orig_counts[2] = 0;
  fixed_counts[0] = 0;
  fixed_counts[1] = 0;
  fixed_counts[2] = 0;
  for (int i = 0; i < 11; i++) policy_bump_total_hist[i] = 0;
  for (int i = 0; i < 11; i++) policy_nobump_total_hist[i] = 0;
  options_.Add<StringOption>(kSyzygyTablebaseId);
  options_.Add<StringOption>(kInputDirId);
  options_.Add<StringOption>(kOutputDirId);
  options_.Add<IntOption>(kThreadsId, 1, 20) = 1;
  options_.Add<FloatOption>(kTempId, 0.001, 100) = 1;
  // Positive dist offset requires knowing the legal move set, so not supported
  // for now.
  options_.Add<FloatOption>(kDistributionOffsetId, -0.999, 0) = 0;
  options_.Add<FloatOption>(kMinDTZBoostId, 0, 1) = 0;
  SelfPlayTournament::PopulateOptions(&options_);

  if (!options_.ProcessAllFlags()) return;
  SyzygyTablebase tablebase;
  if (!tablebase.init(options_.GetOptionsDict().Get<std::string>(
          kSyzygyTablebaseId.GetId())) ||
      tablebase.max_cardinality() < 3) {
    std::cerr << "FAILED TO LOAD SYZYGY" << std::endl;
    return;
  }
  auto inputDir =
      options_.GetOptionsDict().Get<std::string>(kInputDirId.GetId());
  auto files = GetFileList(inputDir);
  if (files.size() == 0) {
    std::cerr << "No files to process" << std::endl;
    return;
  }
  for (int i = 0; i < files.size(); i++) {
    files[i] = inputDir + "/" + files[i];
  }
  float dtz_boost =
      options_.GetOptionsDict().Get<float>(kMinDTZBoostId.GetId());
  int threads = options_.GetOptionsDict().Get<int>(kThreadsId.GetId());
  if (threads > 1) {
    std::vector<std::thread> threads_;
    int offset = 0;
    while (threads_.size() < threads) {
      int offset_val = offset;
      offset++;
      threads_.emplace_back([this, offset_val, files, &tablebase, threads,
                             dtz_boost]() {
        ProcessFiles(
            files, &tablebase,
            options_.GetOptionsDict().Get<std::string>(kOutputDirId.GetId()),
            options_.GetOptionsDict().Get<float>(kTempId.GetId()),
            options_.GetOptionsDict().Get<float>(kDistributionOffsetId.GetId()),
            dtz_boost, offset_val, threads);
      });
    }
    for (int i = 0; i < threads_.size(); i++) {
      threads_[i].join();
    }

  } else {
    ProcessFiles(
        files, &tablebase,
        options_.GetOptionsDict().Get<std::string>(kOutputDirId.GetId()),
        options_.GetOptionsDict().Get<float>(kTempId.GetId()),
        options_.GetOptionsDict().Get<float>(kDistributionOffsetId.GetId()),
        dtz_boost, 0, 1);
  }
  std::cout << "Games processed: " << games << std::endl;
  std::cout << "Positions processed: " << positions << std::endl;
  std::cout << "Rescores performed: " << rescored << std::endl;
  std::cout << "Cumulative outcome change: " << delta << std::endl;
  std::cout << "Secondary rescores performed: " << rescored2 << std::endl;
  std::cout << "Secondary rescores performed used dtz: " << rescored3
            << std::endl;
  std::cout << "Number of policy values boosted by dtz " << policy_bump
            << std::endl;
  std::cout << "Orig policy_sum dist of boost candidate:";
  std::cout << std::endl;
  int event_sum = 0;
  for (int i = 0; i < 11; i++) event_sum += policy_bump_total_hist[i];
  for (int i = 0; i < 11; i++) {
    std::cout << " " << std::setprecision(4) << ((float)policy_nobump_total_hist[i] / (float)event_sum);
  }
  std::cout << std::endl;
  std::cout << "Boosted policy_sum dist of boost candidate:";
  std::cout << std::endl;
  for (int i = 0; i < 11; i++) {
    std::cout << " " << std::setprecision(4)
              << ((float)policy_bump_total_hist[i] / (float)event_sum);
  }
  std::cout << std::endl;
  std::cout << "Original L: " << orig_counts[0] << " D: " << orig_counts[1]
            << " W: " << orig_counts[2] << std::endl;
  std::cout << "After L: " << fixed_counts[0] << " D: " << fixed_counts[1]
            << " W: " << fixed_counts[2] << std::endl;
}

SelfPlayLoop::SelfPlayLoop() {}

SelfPlayLoop::~SelfPlayLoop() {
  if (tournament_) tournament_->Abort();
  if (thread_) thread_->join();
}

void SelfPlayLoop::RunLoop() {
  SelfPlayTournament::PopulateOptions(&options_);

  options_.Add<BoolOption>(kInteractiveId) = false;

  if (!options_.ProcessAllFlags()) return;
  if (options_.GetOptionsDict().Get<bool>(kInteractiveId.GetId())) {
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
  tournament_ = std::make_unique<SelfPlayTournament>(
      options_.GetOptionsDict(),
      std::bind(&UciLoop::SendBestMove, this, std::placeholders::_1),
      std::bind(&UciLoop::SendInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendGameInfo, this, std::placeholders::_1),
      std::bind(&SelfPlayLoop::SendTournament, this, std::placeholders::_1));
  thread_ =
      std::make_unique<std::thread>([this]() { tournament_->RunBlocking(); });
}

void SelfPlayLoop::CmdStop() {
  tournament_->Stop();
  tournament_->Wait();
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
  options_.SetUciOption(name, value, context);
}

void SelfPlayLoop::SendTournament(const TournamentInfo& info) {
  const int winp1 = info.results[0][0] + info.results[0][1];
  const int losep1 = info.results[2][0] + info.results[2][1];
  const int draws = info.results[1][0] + info.results[1][1];

  // Initialize variables.
  float percentage = -1;
  optional<float> elo;
  optional<float> los;

  // Only caculate percentage if any games at all (avoid divide by 0).
  if ((winp1 + losep1 + draws) > 0) {
    percentage =
        (static_cast<float>(draws) / 2 + winp1) / (winp1 + losep1 + draws);
  }
  // Calculate elo and los if percentage strictly between 0 and 1 (avoids divide
  // by 0 or overflow).
  if ((percentage < 1) && (percentage > 0))
    elo = -400 * log(1 / percentage - 1) / log(10);
  if ((winp1 + losep1) > 0) {
    los = .5f +
          .5f * std::erf((winp1 - losep1) / std::sqrt(2.0 * (winp1 + losep1)));
  }
  std::ostringstream oss;
  oss << "tournamentstatus";
  if (info.finished) oss << " final";
  oss << " P1: +" << winp1 << " -" << losep1 << " =" << draws;

  if (percentage > 0) {
    oss << " Win: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (percentage * 100.0f) << "%";
  }
  if (elo) {
    oss << " Elo: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (elo.value_or(0.0f));
  }
  if (los) {
    oss << " LOS: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (los.value_or(0.0f) * 100.0f) << "%";
  }

  oss << " P1-W: +" << info.results[0][0] << " -" << info.results[2][0] << " ="
      << info.results[1][0];
  oss << " P1-B: +" << info.results[0][1] << " -" << info.results[2][1] << " ="
      << info.results[1][1];
  SendResponse(oss.str());
}

}  // namespace lczero

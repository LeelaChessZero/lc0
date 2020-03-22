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

#include <optional>

#include "neural/encoder.h"
#include "neural/writer.h"
#include "selfplay/tournament.h"
#include "utils/configfile.h"
#include "utils/filesystem.h"

#include "gaviotatb/gtb-probe.h"

namespace lczero {

namespace {
const OptionId kInteractiveId{
    "interactive", "", "Run in interactive mode with UCI-like interface."};
const OptionId kSyzygyTablebaseId{"syzygy-paths", "",
                                  "List of Syzygy tablebase directories"};
const OptionId kGaviotaTablebaseId{"gaviotatb-paths", "",
                                   "List of Gaviota tablebase directories"};
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
std::atomic<int> policy_dtm_bump(0);
std::atomic<int> gaviota_dtm_rescores(0);
bool gaviotaEnabled = false;

void DataAssert(bool check_result) {
  if (!check_result) throw Exception("Range Violation");
}

void Validate(const std::vector<V5TrainingData>& fileContents) {
  if (fileContents.empty()) throw Exception("Empty File");

  for (int i = 0; i < fileContents.size(); i++) {
    auto& data = fileContents[i];
    DataAssert(data.input_format ==
                   pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE ||
               data.input_format ==
                   pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE);
    DataAssert(data.best_d >= 0.0f && data.best_d <= 1.0f);
    DataAssert(data.root_d >= 0.0f && data.root_d <= 1.0f);
    DataAssert(data.best_q >= -1.0f && data.best_q <= 1.0f);
    DataAssert(data.root_q >= -1.0f && data.root_q <= 1.0f);
    DataAssert(data.root_m >= 0.0f);
    DataAssert(data.best_m >= 0.0f);
    DataAssert(data.plies_left >= 0.0f);
    DataAssert(data.castling_them_oo >= 0 && data.castling_them_oo <= 1);
    DataAssert(data.castling_them_ooo >= 0 && data.castling_them_ooo <= 1);
    DataAssert(data.castling_us_oo >= 0 && data.castling_us_oo <= 1);
    DataAssert(data.castling_us_ooo >= 0 && data.castling_us_ooo <= 1);
    DataAssert(data.side_to_move >= 0 && data.side_to_move <= 1);
    DataAssert(data.result >= -1 && data.result <= 1);
    DataAssert(data.rule50_count >= 0 && data.rule50_count <= 100);
    float sum = 0.0f;
    for (int j = 0; j < sizeof(data.probabilities) / sizeof(float); j++) {
      float prob = data.probabilities[j];
      DataAssert(prob >= 0.0f && prob <= 1.0f || prob == -1.0f ||
                 std::isnan(prob));
      if (prob >= 0.0f) {
        sum += prob;
      }
    }
    if (sum < 0.99f || sum > 1.01f) {
      throw Exception("Probability sum error is huge!");
    }
  }
}

void Validate(const std::vector<V5TrainingData>& fileContents,
              const MoveList& moves) {
  PositionHistory history;
  int rule50ply;
  int gameply;
  ChessBoard board;
  auto input_format = static_cast<pblczero::NetworkFormat::InputFormat>(
      fileContents[0].input_format);
  PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]), &board,
                &rule50ply, &gameply);
  history.Reset(board, rule50ply, gameply);
  for (int i = 0; i < moves.size(); i++) {
    if (!(fileContents[i].probabilities[moves[i].as_nn_index()] >= 0.0f)) {
      std::cerr << "Illegal move: " << moves[i].as_string() << std::endl;
      throw Exception("Move performed is marked illegal in probabilities.");
    }
    auto legal = history.Last().GetBoard().GenerateLegalMoves();
    if (std::find(legal.begin(), legal.end(), moves[i]) == legal.end()) {
      std::cerr << "Illegal move: " << moves[i].as_string() << std::endl;
      throw Exception("Move performed is an illegal move.");
    }
    history.Append(moves[i]);
  }
}

void gaviota_tb_probe_hard(const Position& pos, unsigned int& info,
                           unsigned int& dtm) {
  unsigned int wsq[17];
  unsigned int bsq[17];
  unsigned char wpc[17];
  unsigned char bpc[17];

  auto stm = pos.IsBlackToMove() ? tb_BLACK_TO_MOVE : tb_WHITE_TO_MOVE;
  auto& board = pos.IsBlackToMove() ? pos.GetThemBoard() : pos.GetBoard();
  auto epsq = tb_NOSQUARE;
  for (auto sq : board.en_passant()) {
    // Our internal representation stores en_passant 2 rows away
    // from the actual sq.
    if (sq.row() == 0) {
      epsq = (TB_squares)(sq.as_int() + 16);
    } else {
      epsq = (TB_squares)(sq.as_int() - 16);
    }
  }
  int idx = 0;
  for (auto sq : (board.ours() & board.kings())) {
    wsq[idx] = (TB_squares)sq.as_int();
    wpc[idx] = tb_KING;
    idx++;
  }
  for (auto sq : (board.ours() & board.knights())) {
    wsq[idx] = (TB_squares)sq.as_int();
    wpc[idx] = tb_KNIGHT;
    idx++;
  }
  for (auto sq : (board.ours() & board.queens())) {
    wsq[idx] = (TB_squares)sq.as_int();
    wpc[idx] = tb_QUEEN;
    idx++;
  }
  for (auto sq : (board.ours() & board.rooks())) {
    wsq[idx] = (TB_squares)sq.as_int();
    wpc[idx] = tb_ROOK;
    idx++;
  }
  for (auto sq : (board.ours() & board.bishops())) {
    wsq[idx] = (TB_squares)sq.as_int();
    wpc[idx] = tb_BISHOP;
    idx++;
  }
  for (auto sq : (board.ours() & board.pawns())) {
    wsq[idx] = (TB_squares)sq.as_int();
    wpc[idx] = tb_PAWN;
    idx++;
  }
  wsq[idx] = tb_NOSQUARE;
  wpc[idx] = tb_NOPIECE;

  idx = 0;
  for (auto sq : (board.theirs() & board.kings())) {
    bsq[idx] = (TB_squares)sq.as_int();
    bpc[idx] = tb_KING;
    idx++;
  }
  for (auto sq : (board.theirs() & board.knights())) {
    bsq[idx] = (TB_squares)sq.as_int();
    bpc[idx] = tb_KNIGHT;
    idx++;
  }
  for (auto sq : (board.theirs() & board.queens())) {
    bsq[idx] = (TB_squares)sq.as_int();
    bpc[idx] = tb_QUEEN;
    idx++;
  }
  for (auto sq : (board.theirs() & board.rooks())) {
    bsq[idx] = (TB_squares)sq.as_int();
    bpc[idx] = tb_ROOK;
    idx++;
  }
  for (auto sq : (board.theirs() & board.bishops())) {
    bsq[idx] = (TB_squares)sq.as_int();
    bpc[idx] = tb_BISHOP;
    idx++;
  }
  for (auto sq : (board.theirs() & board.pawns())) {
    bsq[idx] = (TB_squares)sq.as_int();
    bpc[idx] = tb_PAWN;
    idx++;
  }
  bsq[idx] = tb_NOSQUARE;
  bpc[idx] = tb_NOPIECE;

  tb_probe_hard(stm, epsq, tb_NOCASTLE, wsq, bsq, wpc, bpc, &info, &dtm);
}

void ProcessFile(const std::string& file, SyzygyTablebase* tablebase,
                 std::string outputDir, float distTemp, float distOffset,
                 float dtzBoost) {
  // Scope to ensure reader and writer are closed before deleting source file.
  {
    try {
      TrainingDataReader reader(file);
      std::vector<V5TrainingData> fileContents;
      V5TrainingData data;
      while (reader.ReadChunk(&data)) {
        fileContents.push_back(data);
      }
      Validate(fileContents);
      MoveList moves;
      for (int i = 1; i < fileContents.size(); i++) {
        moves.push_back(
            DecodeMoveFromInput(PlanesFromTrainingData(fileContents[i])));
        // All moves decoded are from the point of view of the side after the
        // move so need to mirror them all to be applicable to apply to the
        // position before.
        moves.back().Mirror();
      }
      Validate(fileContents, moves);
      games += 1;
      positions += fileContents.size();
      PositionHistory history;
      int rule50ply;
      int gameply;
      ChessBoard board;
      auto input_format = static_cast<pblczero::NetworkFormat::InputFormat>(
          fileContents[0].input_format);
      PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                    &board, &rule50ply, &gameply);
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
          // Only fail state means the WDL is wrong, probe_wdl may produce
          // correct result with a stat other than OK.
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
              std::cerr << "Rescoring: " << (int)fileContents[j].result << " ->
              "
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
      PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                    &board, &rule50ply, &gameply);
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
          // Only fail state means the WDL is wrong, probe_wdl may produce
          // correct result with a stat other than OK.
          if (state != FAIL) {
            int8_t score_to_apply = 0;
            if (wdl == WDL_WIN) {
              score_to_apply = 1;
            } else if (wdl == WDL_LOSS) {
              score_to_apply = -1;
            }
            // If the WDL result disagrees with the game outcome, make it a
            // draw. WDL draw is always draw regardless of prior moves since
            // zero, so that clearly works. Otherwise, the WDL result could be
            // correct or draw, so best we can do is change scores that don't
            // agree, to be a draw. If score was a draw this is a no-op, if it
            // was opposite it becomes a draw.
            int8_t new_score = fileContents[i + 1].result != score_to_apply
                                   ? 0
                                   : fileContents[i + 1].result;
            bool dtz_rescored = false;
            // if score is not already right, and the score to apply isn't 0,
            // dtz can let us know its definitely correct.
            if (fileContents[i + 1].result != score_to_apply &&
                score_to_apply != 0) {
              // Any repetitions in the history since last 50 ply makes it risky
              // to assume dtz is still correct.
              int steps = history.Last().GetNoCaptureNoPawnPly();
              bool no_reps = true;
              for (int i = 0; i < steps; i++) {
                // If game started from non-zero 50 move rule, this could
                // underflow. Only safe option is to assume there were
                // repetitions before this point.
                if (history.GetLength() - i - 1 < 0) {
                  no_reps = false;
                  break;
                }
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

            // If score is not already a draw, and its not obviously a draw,
            // check if 50 move rule has advanced so far its obviously a draw.
            // Obviously not needed if we've already proven with dtz that its a
            // win/loss.
            if (fileContents[i + 1].result != 0 && score_to_apply != 0 &&
                !dtz_rescored) {
              int depth = tablebase->probe_dtz(history.Last(), &state);
              if (state != FAIL) {
                int steps = history.Last().GetNoCaptureNoPawnPly();
                // This should be able to be >= 101 safely, but I've not
                // convinced myself thats true.
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
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
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
            MoveList maybe_boost;
            tablebase->root_probe(history.Last(),
                                  history.DidRepeatSinceLastZeroingMove(), true,
                                  &to_boost, &maybe_boost);
            // If there is only one move, dtm fixup is not helpful.
            // This code assumes all gaviota 3-4-5 tbs are present, as checked
            // at startup.
            if (gaviotaEnabled && maybe_boost.size() > 1 &&
                (board.ours() | board.theirs()).count() <= 5) {
              std::vector<int> dtms;
              dtms.resize(maybe_boost.size());
              int mininum_dtm = 1000;
              // Only safe moves being considered, boost the smallest dtm
              // amongst them.
              for (auto& move : maybe_boost) {
                Position next_pos = Position(history.Last(), move);
                unsigned int info;
                unsigned int dtm;
                gaviota_tb_probe_hard(next_pos, info, dtm);
                dtms.push_back(dtm);
                if (dtm < mininum_dtm) mininum_dtm = dtm;
              }
              if (mininum_dtm < 1000) {
                to_boost.clear();
                int dtm_idx = 0;
                for (auto& move : maybe_boost) {
                  if (dtms[dtm_idx] == mininum_dtm) {
                    to_boost.push_back(move);
                  }
                  dtm_idx++;
                }
                policy_dtm_bump++;
              }
            }
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
                distOffset +
                (boost_probs[prob_index] ? (dtzBoost / boost_count) : 0.0f);
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

      // Make move_count field plies_left for moves left head.
      int offset = 0;
      bool all_draws = true;
      for (auto& chunk : fileContents) {
        // plies_left can't be 0 for real v5 data, so if it is 0 it must be a v4
        // conversion, and we should populate it ourselves with a better
        // starting estimate.
        if (chunk.plies_left == 0.0f) {
          chunk.plies_left = (int)(fileContents.size() - offset);
        }
        offset++;
        all_draws = all_draws && (chunk.result == 0);
      }

      // Correct plies_left using Gaviota TBs for 5 piece and less positions.
      if (gaviotaEnabled && !all_draws) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        int last_rescore = 0;
        for (int i = 0; i < moves.size(); i++) {
          history.Append(moves[i]);
          const auto& board = history.Last().GetBoard();

          // Gaviota TBs don't have 50 move rule.
          // Only consider positions that are not draw after rescoring.
          if ((fileContents[i + 1].result != 0) &&
              board.castlings().no_legal_castle() &&
              (board.ours() | board.theirs()).count() <= 5) {
            std::vector<int> dtms;
            unsigned int info;
            unsigned int dtm;
            gaviota_tb_probe_hard(history.Last(), info, dtm);
            if (info != tb_WMATE && info != tb_BMATE) {
              // Not a win for either player.
              continue;
            }
            int steps = history.Last().GetNoCaptureNoPawnPly();
            if ((dtm + steps > 99) && (dtm <= fileContents[i + 1].plies_left)) {
              // Following DTM could trigger 50 move rule and the current
              // move_count is more than DTM.
              // If DTM is more than the current move_count then we can rescore
              // using it since DTM50 is not shorter than DTM.
              continue;
            }
            bool no_reps = true;
            for (int i = 0; i < steps; i++) {
              // If game started from non-zero 50 move rule, this could
              // underflow. Only safe option is to assume there were repetitions
              // before this point.
              if (history.GetLength() - i - 1 < 0) {
                no_reps = false;
                break;
              }
              if (history.GetPositionAt(history.GetLength() - i - 1)
                      .GetRepetitions() != 0) {
                no_reps = false;
                break;
              }
            }
            if (!no_reps) {
              // There were repetitions. Do nothing since DTM path
              // could trigger draw by repetition.
              continue;
            }
            gaviota_dtm_rescores++;
            int j;
            for (j = i; j >= -1; j--) {
              if (j <= last_rescore) {
                break;
              }
              // std::cerr << j << " " << int(fileContents[j + 1].move_count) <<
              // " -> " << int(dtm + (i - j)) << std::endl;
              fileContents[j + 1].plies_left = int(dtm + (i - j));
            }
            last_rescore = i;
          }
        }
      }

      // Correct move_count using DTZ for 3 piece no-pawn positions only.
      // If Gaviota TBs are enabled no need to use syzygy.
      if (!gaviotaEnabled && !all_draws) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        for (int i = 0; i < moves.size(); i++) {
          history.Append(moves[i]);
          const auto& board = history.Last().GetBoard();
          if (board.castlings().no_legal_castle() &&
              (board.ours() | board.theirs()).count() <= 3 &&
              board.pawns().empty()) {
            ProbeState state;
            WDLScore wdl = tablebase->probe_wdl(history.Last(), &state);
            // Only fail state means the WDL is wrong, probe_wdl may produce
            // correct result with a stat other than OK.
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
                // If game started from non-zero 50 move rule, this could
                // underflow. Only safe option is to assume there were
                // repetitions before this point.
                if (history.GetLength() - i - 1 < 0) {
                  no_reps = false;
                  break;
                }
                if (history.GetPositionAt(history.GetLength() - i - 1)
                        .GetRepetitions() != 0) {
                  no_reps = false;
                  break;
                }
              }
              if (no_reps) {
                int depth = tablebase->probe_dtz(history.Last(), &state);
                if (state != FAIL) {
                  // if depth == -1 this is wrong, since that is mate and the
                  // answer should be 0, but the move before depth is -2. Since
                  // data never contains mate position, ignore that discrepency.
                  int converted_ply_remaining = std::abs(depth);
                  // This should be able to be <= 99 safely, but I've not
                  // convinced myself thats true.
                  if (steps + std::abs(depth) < 99) {
                    fileContents[i + 1].plies_left = converted_ply_remaining;
                  }
                  if (steps == 0) {
                    for (int j = i; j >= 0; j--) {
                      fileContents[j].plies_left =
                          converted_ply_remaining + (i + 1 - j);
                    }
                  }
                }
              }
            }
          }
        }
      }

      std::string fileName = file.substr(file.find_last_of("/\\") + 1);
      TrainingDataWriter writer(outputDir + "/" + fileName);
      for (auto chunk : fileContents) {
        writer.WriteChunk(chunk);
      }
    } catch (Exception& ex) {
      std::cerr << "While processing: " << file
                << " - Exception thrown: " << ex.what() << std::endl;
      std::cerr << "It will be deleted." << std::endl;
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
    if (files[i].rfind(".gz") != files[i].size() - 3) {
      std::cerr << "Skipping: " << files[i] << std::endl;
      continue;
    }
    ProcessFile(files[i], tablebase, outputDir, distTemp, distOffset, dtzBoost);
  }
}
}  // namespace

RescoreLoop::RescoreLoop() {}

RescoreLoop::~RescoreLoop() {}

#ifdef _WIN32
#define SEP_CHAR ';'
#else
#define SEP_CHAR ':'
#endif

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
  options_.Add<StringOption>(kGaviotaTablebaseId);
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
  auto dtmPaths =
      options_.GetOptionsDict().Get<std::string>(kGaviotaTablebaseId.GetId());
  if (dtmPaths.size() != 0) {
    std::stringstream path_string_stream(dtmPaths);
    std::string path;
    auto paths = tbpaths_init();
    while (std::getline(path_string_stream, path, SEP_CHAR)) {
      paths = tbpaths_add(paths, path.c_str());
    }
    tb_init(0, tb_CP4, paths);
    tbcache_init(64 * 1024 * 1024, 64);
    if (tb_availability() != 63) {
      std::cerr << "UNEXPECTED gaviota availability" << std::endl;
      return;
    } else {
      std::cerr << "Found Gaviota TBs" << std::endl;
    }
    gaviotaEnabled = true;
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
  std::cout << "Number of policy values boosted by dtz or dtm " << policy_bump
            << std::endl;
  std::cout << "Number of policy values boosted by dtm " << policy_dtm_bump
            << std::endl;
  std::cout << "Orig policy_sum dist of boost candidate:";
  std::cout << std::endl;
  int event_sum = 0;
  for (int i = 0; i < 11; i++) event_sum += policy_bump_total_hist[i];
  for (int i = 0; i < 11; i++) {
    std::cout << " " << std::setprecision(4)
              << ((float)policy_nobump_total_hist[i] / (float)event_sum);
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
  std::cout << "Gaviota DTM move_count rescores: " << gaviota_dtm_rescores
            << std::endl;
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
  res += " play_start_ply " + std::to_string(info.play_start_ply);
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
  std::optional<float> elo;
  std::optional<float> los;

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
        << (*elo);
  }
  if (los) {
    oss << " LOS: " << std::fixed << std::setw(5) << std::setprecision(2)
        << (*los * 100.0f) << "%";
  }

  oss << " P1-W: +" << info.results[0][0] << " -" << info.results[2][0] << " ="
      << info.results[1][0];
  oss << " P1-B: +" << info.results[0][1] << " -" << info.results[2][1] << " ="
      << info.results[1][1];
  oss << " npm " + std::to_string(static_cast<double>(info.nodes_total_) /
                                  info.move_count_);
  oss << " nodes " + std::to_string(info.nodes_total_);
  oss << " moves " + std::to_string(info.move_count_);
  SendResponse(oss.str());
}

}  // namespace lczero

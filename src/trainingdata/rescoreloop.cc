/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2024 The LCZero Authors

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

#include "trainingdata/rescoreloop.h"

#include <optional>
#include <sstream>

#include "gtb-probe.h"
#include "neural/decoder.h"
#include "syzygy/syzygy.h"
#include "trainingdata/reader.h"
#include "utils/filesystem.h"
#include "utils/optionsparser.h"

namespace lczero {

namespace {
const OptionId kSyzygyTablebaseId{"syzygy-paths", "",
                                  "List of Syzygy tablebase directories"};
const OptionId kGaviotaTablebaseId{"gaviotatb-paths", "",
                                   "List of Gaviota tablebase directories"};
const OptionId kInputDirId{
    "input", "", "Directory with gzipped files in need of rescoring."};
const OptionId kPolicySubsDirId{"policy-substitutions", "",
                                "Directory with gzipped files are to use to "
                                "replace policy for some of the data."};
const OptionId kOutputDirId{"output", "", "Directory to write rescored files."};
const OptionId kThreadsId{"threads", "",
                          "Number of concurrent threads to rescore with.", 't'};
const OptionId kTempId{"temperature", "",
                       "Additional temperature to apply to policy target."};
const OptionId kDistributionOffsetId{
    "dist_offset", "",
    "Additional offset to apply to policy target before temperature."};
const OptionId kMinDTZBoostId{
    "dtz_policy_boost", "",
    "Additional offset to apply to policy target before temperature for moves "
    "that are best dtz option."};
const OptionId kNewInputFormatId{
    "new-input-format", "",
    "Input format to convert training data to during rescoring."};
const OptionId kDeblunder{
    "deblunder", "",
    "If true, whether to use move Q information to infer a different Z value "
    "if the the selected move appears to be a blunder."};
const OptionId kDeblunderQBlunderThreshold{
    "deblunder-q-blunder-threshold", "",
    "The amount Q of played move needs to be worse than best move in order to "
    "assume the played move is a blunder."};
const OptionId kDeblunderQBlunderWidth{
    "deblunder-q-blunder-width", "",
    "Width of the transition between accepted temp moves and blunders."};
const OptionId kNnuePlainFileId{"nnue-plain-file", "",
                                "Append SF plain format training data to this "
                                "file. Will be generated if not there."};
const OptionId kNnueBestScoreId{"nnue-best-score", "",
                                "For the SF training data use the score of the "
                                "best move instead of the played one."};
const OptionId kNnueBestMoveId{
    "nnue-best-move", "",
    "For the SF training data record the best move instead of the played one. "
    "If set to true the generated files do not compress well."};
const OptionId kDeleteFilesId{"delete-files", "",
                              "Delete the input files after processing."};

class PolicySubNode {
 public:
  PolicySubNode() {
    for (int i = 0; i < 1858; i++) children[i] = nullptr;
  }
  bool active = false;
  float policy[1858];
  PolicySubNode* children[1858];
};

std::atomic<int> games(0);
std::atomic<int> positions(0);
std::atomic<int> rescored(0);
std::atomic<int> delta(0);
std::atomic<int> rescored2(0);
std::atomic<int> rescored3(0);
std::atomic<int> blunders(0);
std::atomic<int> orig_counts[3];
std::atomic<int> fixed_counts[3];
std::atomic<int> policy_bump(0);
std::atomic<int> policy_nobump_total_hist[11];
std::atomic<int> policy_bump_total_hist[11];
std::atomic<int> policy_dtm_bump(0);
std::atomic<int> gaviota_dtm_rescores(0);
std::map<uint64_t, PolicySubNode> policy_subs;
bool gaviotaEnabled = false;
bool deblunderEnabled = false;
float deblunderQBlunderThreshold = 2.0f;
float deblunderQBlunderWidth = 0.0f;

void DataAssert(bool check_result) {
  if (!check_result) throw Exception("Range Violation");
}

void Validate(const std::vector<V6TrainingData>& fileContents) {
  if (fileContents.empty()) throw Exception("Empty File");

  for (size_t i = 0; i < fileContents.size(); i++) {
    auto& data = fileContents[i];
    DataAssert(
        data.input_format ==
            pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE ||
        data.input_format ==
            pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE ||
        data.input_format ==
            pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION ||
        data.input_format == pblczero::NetworkFormat::
                                 INPUT_112_WITH_CANONICALIZATION_HECTOPLIES ||
        data.input_format ==
            pblczero::NetworkFormat::
                INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
        data.input_format ==
            pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2 ||
        data.input_format == pblczero::NetworkFormat::
                                 INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON);
    DataAssert(data.best_d >= 0.0f && data.best_d <= 1.0f);
    DataAssert(data.root_d >= 0.0f && data.root_d <= 1.0f);
    DataAssert(data.best_q >= -1.0f && data.best_q <= 1.0f);
    DataAssert(data.root_q >= -1.0f && data.root_q <= 1.0f);
    DataAssert(data.root_m >= 0.0f);
    DataAssert(data.best_m >= 0.0f);
    DataAssert(data.plies_left >= 0.0f);
    switch (data.input_format) {
      case pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE:
        DataAssert(data.castling_them_oo <= 1);
        DataAssert(data.castling_them_ooo <= 1);
        DataAssert(data.castling_us_oo <= 1);
        DataAssert(data.castling_us_ooo <= 1);
        break;
      default:
        // Verifiy at most one bit set.
        DataAssert((data.castling_them_oo & (data.castling_them_oo - 1)) == 0);
        DataAssert((data.castling_them_ooo & (data.castling_them_ooo - 1)) ==
                   0);
        DataAssert((data.castling_us_oo & (data.castling_us_oo - 1)) == 0);
        DataAssert((data.castling_us_ooo & (data.castling_us_ooo - 1)) == 0);
    }
    if (IsCanonicalFormat(static_cast<pblczero::NetworkFormat::InputFormat>(
            data.input_format))) {
      // At most one en-passant bit.
      DataAssert((data.side_to_move_or_enpassant &
                  (data.side_to_move_or_enpassant - 1)) == 0);
    } else {
      DataAssert(data.side_to_move_or_enpassant <= 1);
    }
    DataAssert(data.result_q >= -1 && data.result_q <= 1);
    DataAssert(data.result_d >= 0 && data.result_q <= 1);
    DataAssert(data.rule50_count <= 100);
    float sum = 0.0f;
    for (size_t j = 0; j < sizeof(data.probabilities) / sizeof(float); j++) {
      float prob = data.probabilities[j];
      DataAssert((prob >= 0.0f && prob <= 1.0f) || prob == -1.0f ||
                 std::isnan(prob));
      if (prob >= 0.0f) {
        sum += prob;
      }
      // Only check best_idx/played_idx for real v6 data.
      if (data.visits > 0) {
        // Best_idx and played_idx must be marked legal in probabilities.
        if (j == data.best_idx || j == data.played_idx) {
          DataAssert(prob >= 0.0f);
        }
      }
    }
    if (sum < 0.99f || sum > 1.01f) {
      throw Exception("Probability sum error is huge!");
    }
    DataAssert(data.best_idx <= 1858);
    DataAssert(data.played_idx <= 1858);
    DataAssert(data.played_q >= -1.0f && data.played_q <= 1.0f);
    DataAssert(data.played_d >= 0.0f && data.played_d <= 1.0f);
    DataAssert(data.played_m >= 0.0f);
    DataAssert(std::isnan(data.orig_q) ||
               (data.orig_q >= -1.0f && data.orig_q <= 1.0f));
    DataAssert(std::isnan(data.orig_d) ||
               (data.orig_d >= 0.0f && data.orig_d <= 1.0f));
    DataAssert(std::isnan(data.orig_m) || data.orig_m >= 0.0f);
    // TODO: if visits > 0 - assert best_idx/played_idx are valid in
    // probabilities.
  }
}

void Validate(const std::vector<V6TrainingData>& fileContents,
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
  for (size_t i = 0; i < moves.size(); i++) {
    int transform = TransformForPosition(input_format, history);
    // If real v6 data, can confirm that played_idx matches the inferred move.
    if (fileContents[i].visits > 0) {
      if (fileContents[i].played_idx != MoveToNNIndex(moves[i], transform)) {
        throw Exception("Move performed is not listed as played.");
      }
    }
    // Move shouldn't be marked illegal unless there is 0 visits, which should
    // only happen if invariance_info is marked with the placeholder bit.
    if (!(fileContents[i].probabilities[MoveToNNIndex(moves[i], transform)] >=
          0.0f) &&
        (fileContents[i].invariance_info & 64) == 0) {
      std::cerr << "Illegal move: " << moves[i].ToString(true) << std::endl;
      throw Exception("Move performed is marked illegal in probabilities.");
    }
    auto legal = history.Last().GetBoard().GenerateLegalMoves();
    if (std::find(legal.begin(), legal.end(), moves[i]) == legal.end()) {
      std::cerr << "Illegal move: " << moves[i].ToString(true) << std::endl;
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
  ChessBoard board = pos.GetBoard();
  if (pos.IsBlackToMove()) board.Mirror();
  auto epsq = tb_NOSQUARE;
  for (auto sq : board.en_passant()) {
    // Our internal representation stores en_passant 2 rows away
    // from the actual sq.
    if (sq.rank().idx == 0) {
      epsq = (TB_squares)(sq.as_idx() + 16);
    } else {
      epsq = (TB_squares)(sq.as_idx() - 16);
    }
  }
  int idx = 0;
  for (auto sq : (board.ours() & board.kings())) {
    wsq[idx] = (TB_squares)sq.as_idx();
    wpc[idx] = tb_KING;
    idx++;
  }
  for (auto sq : (board.ours() & board.knights())) {
    wsq[idx] = (TB_squares)sq.as_idx();
    wpc[idx] = tb_KNIGHT;
    idx++;
  }
  for (auto sq : (board.ours() & board.queens())) {
    wsq[idx] = (TB_squares)sq.as_idx();
    wpc[idx] = tb_QUEEN;
    idx++;
  }
  for (auto sq : (board.ours() & board.rooks())) {
    wsq[idx] = (TB_squares)sq.as_idx();
    wpc[idx] = tb_ROOK;
    idx++;
  }
  for (auto sq : (board.ours() & board.bishops())) {
    wsq[idx] = (TB_squares)sq.as_idx();
    wpc[idx] = tb_BISHOP;
    idx++;
  }
  for (auto sq : (board.ours() & board.pawns())) {
    wsq[idx] = (TB_squares)sq.as_idx();
    wpc[idx] = tb_PAWN;
    idx++;
  }
  wsq[idx] = tb_NOSQUARE;
  wpc[idx] = tb_NOPIECE;

  idx = 0;
  for (auto sq : (board.theirs() & board.kings())) {
    bsq[idx] = (TB_squares)sq.as_idx();
    bpc[idx] = tb_KING;
    idx++;
  }
  for (auto sq : (board.theirs() & board.knights())) {
    bsq[idx] = (TB_squares)sq.as_idx();
    bpc[idx] = tb_KNIGHT;
    idx++;
  }
  for (auto sq : (board.theirs() & board.queens())) {
    bsq[idx] = (TB_squares)sq.as_idx();
    bpc[idx] = tb_QUEEN;
    idx++;
  }
  for (auto sq : (board.theirs() & board.rooks())) {
    bsq[idx] = (TB_squares)sq.as_idx();
    bpc[idx] = tb_ROOK;
    idx++;
  }
  for (auto sq : (board.theirs() & board.bishops())) {
    bsq[idx] = (TB_squares)sq.as_idx();
    bpc[idx] = tb_BISHOP;
    idx++;
  }
  for (auto sq : (board.theirs() & board.pawns())) {
    bsq[idx] = (TB_squares)sq.as_idx();
    bpc[idx] = tb_PAWN;
    idx++;
  }
  bsq[idx] = tb_NOSQUARE;
  bpc[idx] = tb_NOPIECE;

  tb_probe_hard(stm, epsq, tb_NOCASTLE, wsq, bsq, wpc, bpc, &info, &dtm);
}

void ChangeInputFormat(int newInputFormat, V6TrainingData* data,
                       const PositionHistory& history) {
  data->input_format = newInputFormat;
  auto input_format =
      static_cast<pblczero::NetworkFormat::InputFormat>(newInputFormat);

  // Populate planes.
  int transform;
  InputPlanes planes = EncodePositionForNN(input_format, history, 8,
                                           FillEmptyHistory::NO, &transform);
  int plane_idx = 0;
  for (auto& plane : data->planes) {
    plane = ReverseBitsInBytes(planes[plane_idx++].mask);
  }

  if ((data->invariance_info & 7) != transform) {
    // Probabilities need reshuffling.
    float newProbs[1858];
    std::fill(std::begin(newProbs), std::end(newProbs), -1);
    bool played_fixed = false;
    bool best_fixed = false;
    for (auto move : history.Last().GetBoard().GenerateLegalMoves()) {
      int i = MoveToNNIndex(move, transform);
      int j = MoveToNNIndex(move, data->invariance_info & 7);
      newProbs[i] = data->probabilities[j];
      // For V6 data only, the played/best idx need updating.
      if (data->visits > 0) {
        if (data->played_idx == j && !played_fixed) {
          data->played_idx = i;
          played_fixed = true;
        }
        if (data->best_idx == j && !best_fixed) {
          data->best_idx = i;
          best_fixed = true;
        }
      }
    }
    for (int i = 0; i < 1858; i++) {
      data->probabilities[i] = newProbs[i];
    }
  }

  const auto& position = history.Last();
  const auto& castlings = position.GetBoard().castlings();
  // Populate castlings.
  // For non-frc trained nets, just send 1 like we used to.
  uint8_t our_queen_side = 1;
  uint8_t our_king_side = 1;
  uint8_t their_queen_side = 1;
  uint8_t their_king_side = 1;
  // If frc trained, send the bit mask representing rook position.
  if (Is960CastlingFormat(input_format)) {
    our_queen_side <<= castlings.our_queenside_rook.idx;
    our_king_side <<= castlings.our_kingside_rook.idx;
    their_queen_side <<= castlings.their_queenside_rook.idx;
    their_king_side <<= castlings.their_kingside_rook.idx;
  }

  data->castling_us_ooo = castlings.we_can_000() ? our_queen_side : 0;
  data->castling_us_oo = castlings.we_can_00() ? our_king_side : 0;
  data->castling_them_ooo = castlings.they_can_000() ? their_queen_side : 0;
  data->castling_them_oo = castlings.they_can_00() ? their_king_side : 0;

  // Save the bits that aren't connected to the input_format.
  uint8_t invariance_mask = data->invariance_info & 0x78;
  // Other params.
  if (IsCanonicalFormat(input_format)) {
    data->side_to_move_or_enpassant =
        position.GetBoard().en_passant().as_int() >> 56;
    if ((transform & FlipTransform) != 0) {
      data->side_to_move_or_enpassant =
          ReverseBitsInBytes(data->side_to_move_or_enpassant);
    }
    // Send transform in deprecated move count so rescorer can reverse it to
    // calculate the actual move list from the input data.
    data->invariance_info =
        transform | (position.IsBlackToMove() ? (1u << 7) : 0u);
  } else {
    data->side_to_move_or_enpassant = position.IsBlackToMove() ? 1 : 0;
    data->invariance_info = 0;
  }
  // Put the mask back.
  data->invariance_info |= invariance_mask;
}

int ResultForData(const V6TrainingData& data) {
  // Ensure we aren't reprocessing some data that has had custom adjustments to
  // result training target applied.
  DataAssert(data.result_q == -1.0f || data.result_q == 1.0f ||
             data.result_q == 0.0f);
  // Paranoia - ensure int cast never breaks the value.
  DataAssert(data.result_q ==
             static_cast<float>(static_cast<int>(data.result_q)));
  return static_cast<int>(data.result_q);
}

std::string AsNnueString(const Position& p, Move m, float q, int result) {
  std::ostringstream out;
  out << "fen " << GetFen(p) << std::endl;
  if (p.IsBlackToMove()) m.Flip();
  out << "move " << m.ToString(false) << std::endl;
  // Formula from PR1477 adjusted for SF PawnValueEg.
  out << "score " << round(660.6 * q / (1 - 0.9751875 * std::pow(q, 10)))
      << std::endl;
  out << "ply " << p.GetGamePly() << std::endl;
  out << "result " << result << std::endl;
  out << "e" << std::endl;
  return out.str();
}

struct ProcessFileFlags {
  bool delete_files : 1;
  bool nnue_best_score : 1;
  bool nnue_best_move : 1;
};

void ProcessFile(const std::string& file, SyzygyTablebase* tablebase,
                 std::string outputDir, float distTemp, float distOffset,
                 float dtzBoost, int newInputFormat,
                 std::string nnue_plain_file, ProcessFileFlags flags) {
  // Scope to ensure reader and writer are closed before deleting source file.
  {
    try {
      TrainingDataReader reader(file);
      std::vector<V6TrainingData> fileContents;
      V6TrainingData data;
      while (reader.ReadChunk(&data)) {
        fileContents.push_back(data);
      }
      Validate(fileContents);
      MoveList moves;
      for (size_t i = 1; i < fileContents.size(); i++) {
        moves.push_back(
            DecodeMoveFromInput(PlanesFromTrainingData(fileContents[i]),
                                PlanesFromTrainingData(fileContents[i - 1])));
        // All moves decoded are from the point of view of the side after the
        // move so need to mirror them all to be applicable to apply to the
        // position before.
        moves.back().Flip();
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
      uint64_t rootHash = HashCat(board.Hash(), rule50ply);
      if (policy_subs.find(rootHash) != policy_subs.end()) {
        PolicySubNode* rootNode = &policy_subs[rootHash];
        for (size_t i = 0; i < fileContents.size(); i++) {
          if (rootNode->active) {
            /* Some logic for choosing a softmax to apply to better align the
            new policy with the old policy...
            double bestkld =
              std::numeric_limits<double>::max(); float besttemp = 1.0f;
            // Minima is usually in this range for 'better' data.
            for (float temp = 1.0f; temp < 3.0f; temp += 0.1f) {
              float soft[1858];
              float sum = 0.0f;
              for (int j = 0; j < 1858; j++) {
                if (rootNode->policy[j] >= 0.0) {
                  soft[j] = std::pow(rootNode->policy[j], 1.0f / temp);
                  sum += soft[j];
                } else {
                  soft[j] = -1.0f;
                }
              }
              double kld = 0.0;
              for (int j = 0; j < 1858; j++) {
                if (soft[j] >= 0.0) soft[j] /= sum;
                if (rootNode->policy[j] > 0.0 &&
                    fileContents[i].probabilities[j] > 0) {
                  kld += -1.0f * soft[j] *
                    std::log(fileContents[i].probabilities[j] / soft[j]);
                }
              }
              if (kld < bestkld) {
                bestkld = kld;
                besttemp = temp;
              }
            }
            std::cerr << i << " " << besttemp << " " << bestkld << std::endl;
            */
            for (int j = 0; j < 1858; j++) {
              /*
              if (rootNode->policy[j] >= 0.0) {
                std::cerr << i << " " << j << " " << rootNode->policy[j] << " "
                          << fileContents[i].probabilities[j] << std::endl;
              }
              */
              fileContents[i].probabilities[j] = rootNode->policy[j];
            }
          }
          if (i + 1 < fileContents.size()) {
            int transform = TransformForPosition(input_format, history);
            int idx = MoveToNNIndex(moves[i], transform);
            if (rootNode->children[idx] == nullptr) {
              break;
            }
            rootNode = rootNode->children[idx];
            history.Append(moves[i]);
          }
        }
      }

      PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                    &board, &rule50ply, &gameply);
      history.Reset(board, rule50ply, gameply);
      int last_rescore = -1;
      orig_counts[ResultForData(fileContents[0]) + 1]++;
      fixed_counts[ResultForData(fileContents[0]) + 1]++;
      for (int i = 0; i < static_cast<int>(moves.size()); i++) {
        history.Append(moves[i]);
        const auto& board = history.Last().GetBoard();
        if (board.castlings().no_legal_castle() &&
            history.Last().GetRule50Ply() == 0 &&
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
              if (ResultForData(fileContents[j]) != score_to_apply) {
                if (j == i + 1 && last_rescore == -1) {
                  fixed_counts[ResultForData(fileContents[0]) + 1]--;
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
                delta += abs(ResultForData(fileContents[j]) - score_to_apply);
                /*
              std::cerr << "Rescoring: " << (int)fileContents[j].result << " ->
              "
                        << (int)score_to_apply
                        << std::endl;
                        */
              }

              if (score_to_apply == 0) {
                fileContents[j].result_d = 1.0f;
              } else {
                fileContents[j].result_d = 0.0f;
              }
              fileContents[j].result_q = static_cast<float>(score_to_apply);
              score_to_apply = -score_to_apply;
            }
            last_rescore = i + 1;
          }
        }
      }
      PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                    &board, &rule50ply, &gameply);
      history.Reset(board, rule50ply, gameply);
      for (size_t i = 0; i < moves.size(); i++) {
        history.Append(moves[i]);
        const auto& board = history.Last().GetBoard();
        if (board.castlings().no_legal_castle() &&
            history.Last().GetRule50Ply() != 0 &&
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
            int8_t new_score =
                ResultForData(fileContents[i + 1]) != score_to_apply
                    ? 0
                    : ResultForData(fileContents[i + 1]);
            bool dtz_rescored = false;
            // if score is not already right, and the score to apply isn't 0,
            // dtz can let us know its definitely correct.
            if (ResultForData(fileContents[i + 1]) != score_to_apply &&
                score_to_apply != 0) {
              // Any repetitions in the history since last 50 ply makes it risky
              // to assume dtz is still correct.
              int steps = history.Last().GetRule50Ply();
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
            if (ResultForData(fileContents[i + 1]) != 0 &&
                score_to_apply != 0 && !dtz_rescored) {
              int depth = tablebase->probe_dtz(history.Last(), &state);
              if (state != FAIL) {
                int steps = history.Last().GetRule50Ply();
                // This should be able to be >= 101 safely, but I've not
                // convinced myself thats true.
                if (steps + std::abs(depth) > 101) {
                  rescored3++;
                  new_score = 0;
                  dtz_rescored = true;
                }
              }
            }
            if (new_score != ResultForData(fileContents[i + 1])) {
              rescored2 += 1;
              /*
            std::cerr << "Rescoring: " << (int)fileContents[j].result << " -> "
                      << (int)score_to_apply
                      << std::endl;
                      */
            }

            if (new_score == 0) {
              fileContents[i + 1].result_d = 1.0f;
            } else {
              fileContents[i + 1].result_d = 0.0f;
            }
            fileContents[i + 1].result_q = static_cast<float>(new_score);
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
            tablebase->root_probe(history.Last(), true, true, &to_boost);
            if (history.DidRepeatSinceLastZeroingMove()) {
              maybe_boost = to_boost;
            } else {
              tablebase->root_probe(history.Last(), false, true, &maybe_boost);
            }
            // If there is only one move, dtm fixup is not helpful.
            // This code assumes all gaviota 3-4-5 tbs are present, as checked
            // at startup.
            if (gaviotaEnabled && maybe_boost.size() > 1 &&
                (board.ours() | board.theirs()).count() <= 5) {
              std::vector<unsigned int> dtms;
              dtms.resize(maybe_boost.size());
              unsigned int mininum_dtm = 1000;
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
            int transform = TransformForPosition(input_format, history);
            for (auto& move : to_boost) {
              boost_probs[MoveToNNIndex(move, transform)] = true;
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
        all_draws = all_draws && (ResultForData(chunk) == 0);
      }

      // Correct plies_left using Gaviota TBs for 5 piece and less positions.
      if (gaviotaEnabled && !all_draws) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        int last_rescore = 0;
        for (size_t i = 0; i < moves.size(); i++) {
          history.Append(moves[i]);
          const auto& board = history.Last().GetBoard();

          // Gaviota TBs don't have 50 move rule.
          // Only consider positions that are not draw after rescoring.
          if ((ResultForData(fileContents[i + 1]) != 0) &&
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
            int steps = history.Last().GetRule50Ply();
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
        for (size_t i = 0; i < moves.size(); i++) {
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
              int steps = history.Last().GetRule50Ply();
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
      // Deblunder only works from v6 data onwards. We therefore check
      // the visits field which is 0 if we're dealing with upgraded data.
      if (deblunderEnabled && fileContents.back().visits > 0) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        for (size_t i = 0; i < moves.size(); i++) {
          history.Append(moves[i]);
          const auto& board = history.Last().GetBoard();
          if (board.castlings().no_legal_castle() &&
              (board.ours() | board.theirs()).count() <=
                  tablebase->max_cardinality()) {
            history.Pop();
            break;
          }
        }
        float activeZ[3] = {fileContents.back().result_q,
                            fileContents.back().result_d,
                            fileContents.back().plies_left};
        bool deblunderingStarted = false;
        while (true) {
          auto& cur = fileContents[history.GetLength() - 1];
          // A blunder is defined by the played move being worse than the
          // best move by a defined threshold, missing a forced win, or
          // playing into a proven loss without being forced.
          bool deblunderTriggerThreshold =
              (cur.best_q - cur.played_q >
               deblunderQBlunderThreshold - deblunderQBlunderWidth / 2.0);
          bool deblunderTriggerTerminal =
              (cur.best_q > -1 && cur.played_q < 1 &&
               ((cur.best_q == 1 && ((cur.invariance_info & 8) != 0)) ||
                cur.played_q == -1));
          if (deblunderTriggerThreshold || deblunderTriggerTerminal) {
            float newZRatio = 1.0f;
            // If width > 0 and the deblunder didn't involve a terminal
            // position, we apply a soft threshold by averaging old and new Z.
            if (deblunderQBlunderWidth > 0 && !deblunderTriggerTerminal) {
              newZRatio = std::min(1.0f, (cur.best_q - cur.played_q -
                                          deblunderQBlunderThreshold) /
                                                 deblunderQBlunderWidth +
                                             0.5f);
            }
            // Instead of averaging, a randomization can be applied here with
            // newZRatio = newZRatio > rand( [0, 1) ) ? 1.0f : 0.0f;
            activeZ[0] = (1 - newZRatio) * activeZ[0] + newZRatio * cur.best_q;
            activeZ[1] = (1 - newZRatio) * activeZ[1] + newZRatio * cur.best_d;
            activeZ[2] = (1 - newZRatio) * activeZ[2] + newZRatio * cur.best_m;
            deblunderingStarted = true;
            blunders += 1;
            /* std::cout << "Blunder detected. Best move q=" << cur.best_q <<
             " played move q=" << cur.played_q; */
          }
          if (deblunderingStarted) {
            /*
            std::cerr << "Deblundering: "
                      << fileContents[history.GetLength() - 1].best_q << " "
                      << fileContents[history.GetLength() - 1].best_d << " "
                      << (int)fileContents[history.GetLength() - 1].result << "
            "
                      << (int)activeZ << std::endl;
                      */
            fileContents[history.GetLength() - 1].result_q = activeZ[0];
            fileContents[history.GetLength() - 1].result_d = activeZ[1];
            fileContents[history.GetLength() - 1].plies_left = activeZ[2];
          }
          if (history.GetLength() == 1) break;
          // Q values are always from the player to move.
          activeZ[0] = -activeZ[0];
          // Estimated remaining plies left has to be increased.
          activeZ[2] += 1.0f;
          history.Pop();
        }
      }
      if (newInputFormat != -1) {
        PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]),
                      &board, &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        ChangeInputFormat(newInputFormat, &fileContents[0], history);
        for (size_t i = 0; i < moves.size(); i++) {
          history.Append(moves[i]);
          ChangeInputFormat(newInputFormat, &fileContents[i + 1], history);
        }
      }

      if (!outputDir.empty()) {
        std::string fileName = file.substr(file.find_last_of("/\\") + 1);
        TrainingDataWriter writer(outputDir + "/" + fileName);
        for (auto chunk : fileContents) {
          // Don't save chunks that just provide move history.
          if ((chunk.invariance_info & 64) == 0) {
            writer.WriteChunk(chunk);
          }
        }
      }

      // Output data in Stockfish plain format.
      if (!nnue_plain_file.empty()) {
        static Mutex mutex;
        std::ostringstream out;
        pblczero::NetworkFormat::InputFormat format;
        if (newInputFormat != -1) {
          format =
              static_cast<pblczero::NetworkFormat::InputFormat>(newInputFormat);
        } else {
          format = input_format;
        }
        PopulateBoard(format, PlanesFromTrainingData(fileContents[0]), &board,
                      &rule50ply, &gameply);
        history.Reset(board, rule50ply, gameply);
        for (size_t i = 0; i < fileContents.size(); i++) {
          auto chunk = fileContents[i];
          Position p = history.Last();
          if (chunk.visits > 0) {
            // Format is v6 and position is evaluated.
            Move m = MoveFromNNIndex(
                flags.nnue_best_move ? chunk.best_idx : chunk.played_idx,
                TransformForPosition(format, history));
            float q = flags.nnue_best_score ? chunk.best_q : chunk.played_q;
            out << AsNnueString(p, m, q, round(chunk.result_q));
          } else if (i < moves.size()) {
            out << AsNnueString(p, moves[i], chunk.best_q,
                                round(chunk.result_q));
          }
          if (i < moves.size()) {
            history.Append(moves[i]);
          }
        }
        std::ofstream file;
        Mutex::Lock lock(mutex);
        file.open(nnue_plain_file, std::ios_base::app);
        if (file.is_open()) {
          file << out.str();
          file.close();
        }
      }
    } catch (Exception& ex) {
      std::cerr << "While processing: " << file
                << " - Exception thrown: " << ex.what() << std::endl;
      if (flags.delete_files) {
        std::cerr << "It will be deleted." << std::endl;
      }
    }
  }
  if (flags.delete_files) {
    remove(file.c_str());
  }
}

void ProcessFiles(const std::vector<std::string>& files,
                  SyzygyTablebase* tablebase, std::string outputDir,
                  float distTemp, float distOffset, float dtzBoost,
                  int newInputFormat, int offset, int mod,
                  std::string nnue_plain_file, ProcessFileFlags flags) {
  std::cerr << "Thread: " << offset << " starting" << std::endl;
  for (size_t i = offset; i < files.size(); i += mod) {
    if (files[i].rfind(".gz") != files[i].size() - 3) {
      std::cerr << "Skipping: " << files[i] << std::endl;
      continue;
    }
    ProcessFile(files[i], tablebase, outputDir, distTemp, distOffset, dtzBoost,
                newInputFormat, nnue_plain_file, flags);
  }
}

void BuildSubs(const std::vector<std::string>& files) {
  for (auto& file : files) {
    TrainingDataReader reader(file);
    std::vector<V6TrainingData> fileContents;
    V6TrainingData data;
    while (reader.ReadChunk(&data)) {
      fileContents.push_back(data);
    }
    Validate(fileContents);
    MoveList moves;
    for (size_t i = 1; i < fileContents.size(); i++) {
      moves.push_back(
          DecodeMoveFromInput(PlanesFromTrainingData(fileContents[i]),
                              PlanesFromTrainingData(fileContents[i - 1])));
      // All moves decoded are from the point of view of the side after the
      // move so need to mirror them all to be applicable to apply to the
      // position before.
      moves.back().Flip();
    }
    Validate(fileContents, moves);

    // Subs are 'valid'.
    PositionHistory history;
    int rule50ply;
    int gameply;
    ChessBoard board;
    auto input_format = static_cast<pblczero::NetworkFormat::InputFormat>(
        fileContents[0].input_format);
    PopulateBoard(input_format, PlanesFromTrainingData(fileContents[0]), &board,
                  &rule50ply, &gameply);
    history.Reset(board, rule50ply, gameply);
    uint64_t rootHash = HashCat(board.Hash(), rule50ply);
    PolicySubNode* rootNode = &policy_subs[rootHash];
    for (size_t i = 0; i < fileContents.size(); i++) {
      if ((fileContents[i].invariance_info & 64) == 0) {
        rootNode->active = true;
        for (int j = 0; j < 1858; j++) {
          rootNode->policy[j] = fileContents[i].probabilities[j];
        }
      }
      if (i < fileContents.size() - 1) {
        int transform = TransformForPosition(input_format, history);
        int idx = MoveToNNIndex(moves[i], transform);
        if (rootNode->children[idx] == nullptr) {
          rootNode->children[idx] = new PolicySubNode();
        }
        rootNode = rootNode->children[idx];
        history.Append(moves[i]);
      }
    }
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
  options_.Add<StringOption>(kPolicySubsDirId);
  options_.Add<IntOption>(kThreadsId, 1, 20) = 1;
  options_.Add<FloatOption>(kTempId, 0.001, 100) = 1;
  // Positive dist offset requires knowing the legal move set, so not supported
  // for now.
  options_.Add<FloatOption>(kDistributionOffsetId, -0.999, 0) = 0;
  options_.Add<FloatOption>(kMinDTZBoostId, 0, 1) = 0;
  options_.Add<IntOption>(kNewInputFormatId, -1, 256) = -1;
  options_.Add<BoolOption>(kDeblunder) = false;
  options_.Add<FloatOption>(kDeblunderQBlunderThreshold, 0.0f, 2.0f) = 2.0f;
  options_.Add<FloatOption>(kDeblunderQBlunderWidth, 0.0f, 2.0f) = 0.0f;
  options_.Add<StringOption>(kNnuePlainFileId);
  options_.Add<BoolOption>(kNnueBestScoreId) = true;
  options_.Add<BoolOption>(kNnueBestMoveId) = false;
  options_.Add<BoolOption>(kDeleteFilesId) = true;

  if (!options_.ProcessAllFlags()) return;

  if (options_.GetOptionsDict().IsDefault<std::string>(kOutputDirId) &&
      options_.GetOptionsDict().IsDefault<std::string>(kNnuePlainFileId)) {
    std::cerr << "Must provide an output dir or NNUE plain file." << std::endl;
    return;
  }

  deblunderEnabled = options_.GetOptionsDict().Get<bool>(kDeblunder);
  deblunderQBlunderThreshold =
      options_.GetOptionsDict().Get<float>(kDeblunderQBlunderThreshold);
  deblunderQBlunderWidth =
      options_.GetOptionsDict().Get<float>(kDeblunderQBlunderWidth);

  SyzygyTablebase tablebase;
  if (!tablebase.init(
          options_.GetOptionsDict().Get<std::string>(kSyzygyTablebaseId)) ||
      tablebase.max_cardinality() < 3) {
    std::cerr << "FAILED TO LOAD SYZYGY" << std::endl;
    return;
  }
  auto dtmPaths =
      options_.GetOptionsDict().Get<std::string>(kGaviotaTablebaseId);
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
  auto policySubsDir =
      options_.GetOptionsDict().Get<std::string>(kPolicySubsDirId);
  if (policySubsDir.size() != 0) {
    auto policySubFiles = GetFileList(policySubsDir);
    for (size_t i = 0; i < policySubFiles.size(); i++) {
      policySubFiles[i] = policySubsDir + "/" + policySubFiles[i];
    }
    BuildSubs(policySubFiles);
  }

  auto inputDir = options_.GetOptionsDict().Get<std::string>(kInputDirId);
  if (inputDir.size() == 0) {
    std::cerr << "Must provide an input dir." << std::endl;
    return;
  }
  auto files = GetFileList(inputDir);
  if (files.size() == 0) {
    std::cerr << "No files to process" << std::endl;
    return;
  }
  for (size_t i = 0; i < files.size(); i++) {
    files[i] = inputDir + "/" + files[i];
  }
  float dtz_boost = options_.GetOptionsDict().Get<float>(kMinDTZBoostId);
  unsigned int threads = options_.GetOptionsDict().Get<int>(kThreadsId);
  ProcessFileFlags flags;
  flags.delete_files = options_.GetOptionsDict().Get<bool>(kDeleteFilesId);
  flags.nnue_best_score = options_.GetOptionsDict().Get<bool>(kNnueBestScoreId);
  flags.nnue_best_move = options_.GetOptionsDict().Get<bool>(kNnueBestMoveId);
  if (threads > 1) {
    std::vector<std::thread> threads_;
    int offset = 0;
    while (threads_.size() < threads) {
      int offset_val = offset;
      offset++;
      threads_.emplace_back([this, offset_val, files, &tablebase, threads,
                             dtz_boost, flags]() {
        ProcessFiles(
            files, &tablebase,
            options_.GetOptionsDict().Get<std::string>(kOutputDirId),
            options_.GetOptionsDict().Get<float>(kTempId),
            options_.GetOptionsDict().Get<float>(kDistributionOffsetId),
            dtz_boost, options_.GetOptionsDict().Get<int>(kNewInputFormatId),
            offset_val, threads,
            options_.GetOptionsDict().Get<std::string>(kNnuePlainFileId),
            flags);
      });
    }
    for (size_t i = 0; i < threads_.size(); i++) {
      threads_[i].join();
    }

  } else {
    ProcessFiles(files, &tablebase,
                 options_.GetOptionsDict().Get<std::string>(kOutputDirId),
                 options_.GetOptionsDict().Get<float>(kTempId),
                 options_.GetOptionsDict().Get<float>(kDistributionOffsetId),
                 dtz_boost,
                 options_.GetOptionsDict().Get<int>(kNewInputFormatId), 0, 1,
                 options_.GetOptionsDict().Get<std::string>(kNnuePlainFileId),
                 flags);
  }
  std::cout << "Games processed: " << games << std::endl;
  std::cout << "Positions processed: " << positions << std::endl;
  std::cout << "Rescores performed: " << rescored << std::endl;
  std::cout << "Cumulative outcome change: " << delta << std::endl;
  std::cout << "Secondary rescores performed: " << rescored2 << std::endl;
  std::cout << "Secondary rescores performed used dtz: " << rescored3
            << std::endl;
  std::cout << "Blunders picked up by deblunder threshold: " << blunders
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

}  // namespace lczero

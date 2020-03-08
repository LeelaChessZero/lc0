/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

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

#include "neural/encoder.h"

#include <algorithm>

namespace lczero {

namespace {
const int kMoveHistory = 8;
const int kPlanesPerBoard = 13;
const int kAuxPlaneBase = kPlanesPerBoard * kMoveHistory;
}  // namespace

namespace {
BoardSquare SingleSquare(BitBoard input) {
  for (auto sq : input) {
    return sq;
  }
  assert(false);
  return BoardSquare();
}
}  // namespace

void PopulateBoard(pblczero::NetworkFormat::InputFormat input_format,
                   InputPlanes planes, ChessBoard* board, int* rule50,
                   int* gameply) {
  auto pawnsOurs = BitBoard(planes[0].mask);
  auto knightsOurs = BitBoard(planes[1].mask);
  auto bishopOurs = BitBoard(planes[2].mask);
  auto rookOurs = BitBoard(planes[3].mask);
  auto queenOurs = BitBoard(planes[4].mask);
  auto kingOurs = BitBoard(planes[5].mask);
  auto pawnsTheirs = BitBoard(planes[6].mask);
  auto knightsTheirs = BitBoard(planes[7].mask);
  auto bishopTheirs = BitBoard(planes[8].mask);
  auto rookTheirs = BitBoard(planes[9].mask);
  auto queenTheirs = BitBoard(planes[10].mask);
  auto kingTheirs = BitBoard(planes[11].mask);
  ChessBoard::Castlings castlings;
  if (input_format == pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE) {
    if (planes[kAuxPlaneBase + 0].mask != 0) {
      castlings.set_we_can_000();
    }
    if (planes[kAuxPlaneBase + 1].mask != 0) {
      castlings.set_we_can_00();
    }
    if (planes[kAuxPlaneBase + 2].mask != 0) {
      castlings.set_they_can_000();
    }
    if (planes[kAuxPlaneBase + 3].mask != 0) {
      castlings.set_they_can_00();
    }
  } else {
    // TODO: support.
    throw Exception("New castling format not supported yet.");
  }
  std::string fen;
  if (planes[kAuxPlaneBase + 4].mask != 0) {
    // Flip to white perspective rather than side to move perspective.
    std::swap(pawnsOurs, pawnsTheirs);
    std::swap(knightsOurs, knightsTheirs);
    std::swap(bishopOurs, bishopTheirs);
    std::swap(rookOurs, rookTheirs);
    std::swap(queenOurs, queenTheirs);
    std::swap(kingOurs, kingTheirs);
    pawnsOurs.Mirror();
    pawnsTheirs.Mirror();
    knightsOurs.Mirror();
    knightsTheirs.Mirror();
    bishopOurs.Mirror();
    bishopTheirs.Mirror();
    rookOurs.Mirror();
    rookTheirs.Mirror();
    queenOurs.Mirror();
    queenTheirs.Mirror();
    kingOurs.Mirror();
    kingTheirs.Mirror();
    castlings.Mirror();
  }
  for (int row = 7; row >= 0; --row) {
    int emptycounter = 0;
    for (int col = 0; col < 8; ++col) {
      char piece = '\0';
      if (pawnsOurs.get(row, col)) {
        piece = 'P';
      } else if (pawnsTheirs.get(row, col)) {
        piece = 'p';
      } else if (knightsOurs.get(row, col)) {
        piece = 'N';
      } else if (knightsTheirs.get(row, col)) {
        piece = 'n';
      } else if (bishopOurs.get(row, col)) {
        piece = 'B';
      } else if (bishopTheirs.get(row, col)) {
        piece = 'b';
      } else if (rookOurs.get(row, col)) {
        piece = 'R';
      } else if (rookTheirs.get(row, col)) {
        piece = 'r';
      } else if (queenOurs.get(row, col)) {
        piece = 'Q';
      } else if (queenTheirs.get(row, col)) {
        piece = 'q';
      } else if (kingOurs.get(row, col)) {
        piece = 'K';
      } else if (kingTheirs.get(row, col)) {
        piece = 'k';
      }
      if (emptycounter > 0 && piece) {
        fen += std::to_string(emptycounter);
        emptycounter = 0;
      }
      if (piece) {
        fen += piece;
      } else {
        emptycounter++;
      }
    }
    if (emptycounter > 0) fen += std::to_string(emptycounter);
    if (row > 0) fen += "/";
  }
  fen += " ";
  fen += (planes[kAuxPlaneBase + 4].mask != 0) ? "b" : "w";
  fen += " ";
  fen += castlings.as_string();
  fen += " ";
  auto pawndiff = BitBoard(planes[6].mask ^ planes[kPlanesPerBoard + 6].mask);
  // If no pawns then 2 pawns, history isn't filled properly and we shouldn't
  // try and infer enpassant.
  if (pawndiff.count() == 2 && planes[kPlanesPerBoard + 6].mask != 0) {
    auto from =
        SingleSquare(planes[kPlanesPerBoard + 6].mask & pawndiff.as_int());
    auto to = SingleSquare(planes[6].mask & pawndiff.as_int());
    if (from.col() != to.col() || std::abs(from.row() - to.row()) != 2) {
      fen += "-";
    } else {
      // TODO: Ensure enpassant is legal rather than setting it blindly?
      // Doesn't matter for rescoring use case as only legal moves will be
      // performed afterwards.
      fen +=
          BoardSquare((planes[kAuxPlaneBase + 4].mask != 0) ? 2 : 5, to.col())
              .as_string();
    }
  } else {
    fen += "-";
  }
  fen += " ";
  fen += std::to_string((int)planes[kAuxPlaneBase + 5].value);
  // Reuse the 50 move rule as gameply since we don't know better.
  fen += " ";
  fen += std::to_string((int)planes[kAuxPlaneBase + 5].value);
  board->SetFromFen(fen, rule50, gameply);
}

InputPlanes EncodePositionForNN(
    pblczero::NetworkFormat::InputFormat input_format,
    const PositionHistory& history, int history_planes,
    FillEmptyHistory fill_empty_history) {
  InputPlanes result(kAuxPlaneBase + 8);

  {
    const ChessBoard& board = history.Last().GetBoard();
    const bool we_are_black = board.flipped();
    switch (input_format) {
      case pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE: {
        // "Legacy" input planes with:
        // - Plane 104 (0-based) filled with 1 if white can castle queenside.
        // - Plane 105 filled with ones if white can castle kingside.
        // - Plane 106 filled with ones if black can castle queenside.
        // - Plane 107 filled with ones if white can castle kingside.
        if (board.castlings().we_can_000()) result[kAuxPlaneBase + 0].SetAll();
        if (board.castlings().we_can_00()) result[kAuxPlaneBase + 1].SetAll();
        if (board.castlings().they_can_000()) {
          result[kAuxPlaneBase + 2].SetAll();
        }
        if (board.castlings().they_can_00()) result[kAuxPlaneBase + 3].SetAll();
        break;
      }

      case pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE: {
        // - Plane 104 for positions of rooks (both white and black) which have
        // a-side (queenside) castling right.
        // - Plane 105 for positions of rooks (both white and black) which have
        // h-side (kingside) castling right.
        const auto& cast = board.castlings();
        result[kAuxPlaneBase + 0].mask =
            ((cast.we_can_000() ? ChessBoard::A1 : 0) |
             (cast.they_can_000() ? ChessBoard::A8 : 0))
            << cast.queenside_rook();
        result[kAuxPlaneBase + 1].mask =
            ((cast.we_can_00() ? ChessBoard::A1 : 0) |
             (cast.they_can_00() ? ChessBoard::A8 : 0))
            << cast.kingside_rook();
        break;
      }

      default:
        throw Exception("Unsupported input plane encoding " +
                        std::to_string(input_format));
    };
    if (we_are_black) result[kAuxPlaneBase + 4].SetAll();
    result[kAuxPlaneBase + 5].Fill(history.Last().GetNoCaptureNoPawnPly());
    // Plane kAuxPlaneBase + 6 used to be movecount plane, now it's all zeros.
    // Plane kAuxPlaneBase + 7 is all ones to help NN find board edges.
    result[kAuxPlaneBase + 7].SetAll();
  }

  bool flip = false;
  int history_idx = history.GetLength() - 1;
  for (int i = 0; i < std::min(history_planes, kMoveHistory);
       ++i, --history_idx) {
    const Position& position =
        history.GetPositionAt(history_idx < 0 ? 0 : history_idx);
    const ChessBoard& board =
        flip ? position.GetThemBoard() : position.GetBoard();
    if (history_idx < 0 && fill_empty_history == FillEmptyHistory::NO) break;
    // Board may be flipped so compare with position.GetBoard().
    if (history_idx < 0 && fill_empty_history == FillEmptyHistory::FEN_ONLY &&
        position.GetBoard() == ChessBoard::kStartposBoard) {
      break;
    }

    const int base = i * kPlanesPerBoard;
    result[base + 0].mask = (board.ours() & board.pawns()).as_int();
    result[base + 1].mask = (board.our_knights()).as_int();
    result[base + 2].mask = (board.ours() & board.bishops()).as_int();
    result[base + 3].mask = (board.ours() & board.rooks()).as_int();
    result[base + 4].mask = (board.ours() & board.queens()).as_int();
    result[base + 5].mask = (board.our_king()).as_int();

    result[base + 6].mask = (board.theirs() & board.pawns()).as_int();
    result[base + 7].mask = (board.their_knights()).as_int();
    result[base + 8].mask = (board.theirs() & board.bishops()).as_int();
    result[base + 9].mask = (board.theirs() & board.rooks()).as_int();
    result[base + 10].mask = (board.theirs() & board.queens()).as_int();
    result[base + 11].mask = (board.their_king()).as_int();

    const int repetitions = position.GetRepetitions();
    if (repetitions >= 1) result[base + 12].SetAll();

    // If en passant flag is set, undo last pawn move by removing the pawn from
    // the new square and putting into pre-move square.
    if (history_idx < 0 && !board.en_passant().empty()) {
      const auto idx = GetLowestBit(board.en_passant().as_int());
      if (idx < 8) {  // "Us" board
        result[base + 0].mask +=
            ((0x0000000000000100ULL - 0x0000000001000000ULL) << idx);
      } else {
        result[base + 6].mask +=
            ((0x0001000000000000ULL - 0x0000000100000000ULL) << (idx - 56));
      }
    }
    if (history_idx > 0) flip = !flip;
  }

  return result;
}

Move DecodeMoveFromInput(const InputPlanes& planes) {
  auto pawndiff = BitBoard(planes[6].mask ^ planes[kPlanesPerBoard + 6].mask);
  auto knightdiff = BitBoard(planes[7].mask ^ planes[kPlanesPerBoard + 7].mask);
  auto bishopdiff = BitBoard(planes[8].mask ^ planes[kPlanesPerBoard + 8].mask);
  auto rookdiff = BitBoard(planes[9].mask ^ planes[kPlanesPerBoard + 9].mask);
  auto queendiff =
      BitBoard(planes[10].mask ^ planes[kPlanesPerBoard + 10].mask);
  // Handle Promotion.
  if (pawndiff.count() == 1) {
    auto from = SingleSquare(pawndiff);
    if (knightdiff.count() == 1) {
      auto to = SingleSquare(knightdiff);
      return Move(from, to, Move::Promotion::Knight);
    }
    if (bishopdiff.count() == 1) {
      auto to = SingleSquare(bishopdiff);
      return Move(from, to, Move::Promotion::Bishop);
    }
    if (rookdiff.count() == 1) {
      auto to = SingleSquare(rookdiff);
      return Move(from, to, Move::Promotion::Rook);
    }
    if (queendiff.count() == 1) {
      auto to = SingleSquare(queendiff);
      return Move(from, to, Move::Promotion::Queen);
    }
    assert(false);
    return Move();
  }
  // check king first as castling moves both king and rook.
  auto kingdiff = BitBoard(planes[11].mask ^ planes[kPlanesPerBoard + 11].mask);
  if (kingdiff.count() == 2) {
    if (rookdiff.count() == 2) {
      // TODO: Fix this properly for full 960 support.
      auto from =
          SingleSquare(planes[kPlanesPerBoard + 11].mask & kingdiff.as_int());
      auto to =
          SingleSquare(planes[kPlanesPerBoard + 9].mask & rookdiff.as_int());
      return Move(from, to);
    }
    auto from =
        SingleSquare(planes[kPlanesPerBoard + 11].mask & kingdiff.as_int());
    auto to = SingleSquare(planes[11].mask & kingdiff.as_int());
    return Move(from, to);
  }
  if (queendiff.count() == 2) {
    auto from =
        SingleSquare(planes[kPlanesPerBoard + 10].mask & queendiff.as_int());
    auto to = SingleSquare(planes[10].mask & queendiff.as_int());
    return Move(from, to);
  }
  if (rookdiff.count() == 2) {
    auto from =
        SingleSquare(planes[kPlanesPerBoard + 9].mask & rookdiff.as_int());
    auto to = SingleSquare(planes[9].mask & rookdiff.as_int());
    return Move(from, to);
  }
  if (bishopdiff.count() == 2) {
    auto from =
        SingleSquare(planes[kPlanesPerBoard + 8].mask & bishopdiff.as_int());
    auto to = SingleSquare(planes[8].mask & bishopdiff.as_int());
    return Move(from, to);
  }
  if (knightdiff.count() == 2) {
    auto from =
        SingleSquare(planes[kPlanesPerBoard + 7].mask & knightdiff.as_int());
    auto to = SingleSquare(planes[7].mask & knightdiff.as_int());
    return Move(from, to);
  }
  if (pawndiff.count() == 2) {
    auto from =
        SingleSquare(planes[kPlanesPerBoard + 6].mask & pawndiff.as_int());
    auto to = SingleSquare(planes[6].mask & pawndiff.as_int());
    return Move(from, to);
  }
  assert(false);
  return Move();
}

}  // namespace lczero

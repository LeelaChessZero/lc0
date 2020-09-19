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

int CompareTransposing(BitBoard board, int initial_transform) {
  uint64_t value = board.as_int();
  if ((initial_transform & FlipTransform) != 0) {
    value = ReverseBitsInBytes(value);
  }
  if ((initial_transform & MirrorTransform) != 0) {
    value = ReverseBytesInBytes(value);
  }
  auto alternative = TransposeBitsInBytes(value);
  if (value < alternative) return -1;
  if (value > alternative) return 1;
  return 0;
}

int ChooseTransform(const ChessBoard& board) {
  // If there are any castling options no transform is valid.
  // Even using FRC rules, king and queen side castle moves are not symmetrical.
  if (!board.castlings().no_legal_castle()) {
    return 0;
  }
  auto our_king = (board.kings() & board.ours()).as_int();
  int transform = NoTransform;
  if ((our_king & 0x0F0F0F0F0F0F0F0FULL) != 0) {
    transform |= FlipTransform;
    our_king = ReverseBitsInBytes(our_king);
  }
  // If there are any pawns only horizontal flip is valid.
  if (board.pawns().as_int() != 0) {
    return transform;
  }
  if ((our_king & 0xFFFFFFFF00000000ULL) != 0) {
    transform |= MirrorTransform;
    our_king = ReverseBytesInBytes(our_king);
  }
  // Our king is now always in bottom right quadrant.
  // Transpose for king in top right triangle, or if on diagonal whichever has
  // the smaller integer value for each test scenario.
  if ((our_king & 0xE0C08000ULL) != 0) {
    transform |= TransposeTransform;
  } else if ((our_king & 0x10204080ULL) != 0) {
    auto outcome = CompareTransposing(board.ours() | board.theirs(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.ours(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.kings(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.queens(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.rooks(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.knights(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    outcome = CompareTransposing(board.bishops(), transform);
    if (outcome == -1) return transform;
    if (outcome == 1) return transform | TransposeTransform;
    // If all piece types are symmetrical and ours is symmetrical and
    // ours+theirs is symmetrical, everything is symmetrical, so transpose is a
    // no-op.
  }
  return transform;
}

BoardSquare SingleSquare(BitBoard input) {
  for (auto sq : input) {
    return sq;
  }
  assert(false);
  return BoardSquare();
}

BitBoard MaskDiffWithMirror(const InputPlane& cur, const InputPlane& prev) {
  auto to_mirror = BitBoard(prev.mask);
  to_mirror.Mirror();
  return BitBoard(cur.mask ^ to_mirror.as_int());
}

BoardSquare OldPosition(const InputPlane& prev, BitBoard mask_diff) {
  auto to_mirror = BitBoard(prev.mask);
  to_mirror.Mirror();
  return SingleSquare(to_mirror & mask_diff);
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
  switch (input_format) {
    case pblczero::NetworkFormat::InputFormat::INPUT_CLASSICAL_112_PLANE: {
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
      break;
    }
    case pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
    case pblczero::NetworkFormat::
        INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON:
    case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2:
    case pblczero::NetworkFormat::
        INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON: {
      auto queenside = 0;
      auto kingside = 7;
      if (planes[kAuxPlaneBase + 0].mask != 0) {
        auto mask = planes[kAuxPlaneBase + 0].mask;
        queenside = GetLowestBit((mask >> 56) | mask);
        if ((mask & 0xFFLL) != 0) {
          castlings.set_we_can_000();
        }
        if (mask >> 56 != 0) {
          castlings.set_they_can_000();
        }
      }
      if (planes[kAuxPlaneBase + 1].mask != 0) {
        auto mask = planes[kAuxPlaneBase + 1].mask;
        kingside = GetLowestBit((mask >> 56) | mask);
        if ((mask & 0xFFLL) != 0) {
          castlings.set_we_can_00();
        }
        if (mask >> 56 != 0) {
          castlings.set_they_can_00();
        }
      }
      castlings.SetRookPositions(queenside, kingside);
      break;
    }

    default:
      throw Exception("Unsupported input plane encoding " +
                      std::to_string(input_format));
  }
  std::string fen;
  // Canonical input has no sense of side to move, so we should simply assume
  // the starting position is always white.
  bool black_to_move =
      !IsCanonicalFormat(input_format) && planes[kAuxPlaneBase + 4].mask != 0;
  if (black_to_move) {
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
  fen += black_to_move ? "b" : "w";
  fen += " ";
  fen += castlings.as_string();
  fen += " ";
  if (IsCanonicalFormat(input_format)) {
    // Canonical format helpfully has the en passant details ready for us.
    if (planes[kAuxPlaneBase + 4].mask == 0) {
      fen += "-";
    } else {
      int col = GetLowestBit(planes[kAuxPlaneBase + 4].mask >> 56);
      fen += BoardSquare(5, col).as_string();
    }
  } else {
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
  }
  fen += " ";
  int rule50plane = (int)planes[kAuxPlaneBase + 5].value;
  if (IsHectopliesFormat(input_format)) {
    rule50plane = (int)(100.0f * planes[kAuxPlaneBase + 5].value);
  }
  fen += std::to_string(rule50plane);
  // Reuse the 50 move rule as gameply since we don't know better.
  fen += " ";
  fen += std::to_string(rule50plane);
  board->SetFromFen(fen, rule50, gameply);
}

bool IsCanonicalFormat(pblczero::NetworkFormat::InputFormat input_format) {
  return input_format >=
         pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION;
}
bool IsCanonicalArmageddonFormat(
    pblczero::NetworkFormat::InputFormat input_format) {
  return input_format ==
             pblczero::NetworkFormat::
                 INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON ||
         input_format == pblczero::NetworkFormat::
                             INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
}
bool IsHectopliesFormat(pblczero::NetworkFormat::InputFormat input_format) {
  return input_format >=
         pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES;
}
bool Is960CastlingFormat(pblczero::NetworkFormat::InputFormat input_format) {
  return input_format >= pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE;
}

int TransformForPosition(pblczero::NetworkFormat::InputFormat input_format,
                         const PositionHistory& history) {
  if (!IsCanonicalFormat(input_format)) {
    return 0;
  }
  const ChessBoard& board = history.Last().GetBoard();
  return ChooseTransform(board);
}

InputPlanes EncodePositionForNN(
    pblczero::NetworkFormat::InputFormat input_format,
    const PositionHistory& history, int history_planes,
    FillEmptyHistory fill_empty_history, int* transform_out) {
  InputPlanes result(kAuxPlaneBase + 8);

  int transform = 0;
  // Canonicalization format needs to stop early to avoid applying transform in
  // history across incompatible transitions.  It is also more canonical since
  // history before these points is not relevant to the final result.
  bool stop_early = IsCanonicalFormat(input_format);
  // When stopping early, we want to know if castlings has changed, so capture
  // it for the first board.
  ChessBoard::Castlings castlings;
  {
    const ChessBoard& board = history.Last().GetBoard();
    const bool we_are_black = board.flipped();
    if (IsCanonicalFormat(input_format)) {
      transform = ChooseTransform(board);
    }
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

      case pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE:
      case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION:
      case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES:
      case pblczero::NetworkFormat::
          INPUT_112_WITH_CANONICALIZATION_HECTOPLIES_ARMAGEDDON:
      case pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2:
      case pblczero::NetworkFormat::
          INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON: {
        // - Plane 104 for positions of rooks (both white and black) which
        // have
        // a-side (queenside) castling right.
        // - Plane 105 for positions of rooks (both white and black) which have
        // h-side (kingside) castling right.
        const auto& cast = board.castlings();
        result[kAuxPlaneBase + 0].mask =
            ((cast.we_can_000() ? BoardSquare(ChessBoard::A1).as_board() : 0) |
             (cast.they_can_000() ? BoardSquare(ChessBoard::A8).as_board() : 0))
            << cast.queenside_rook();
        result[kAuxPlaneBase + 1].mask =
            ((cast.we_can_00() ? BoardSquare(ChessBoard::A1).as_board() : 0) |
             (cast.they_can_00() ? BoardSquare(ChessBoard::A8).as_board() : 0))
            << cast.kingside_rook();
        break;
      }
      default:
        throw Exception("Unsupported input plane encoding " +
                        std::to_string(input_format));
    };
    if (IsCanonicalFormat(input_format)) {
      result[kAuxPlaneBase + 4].mask = board.en_passant().as_int();
    } else {
      if (we_are_black) result[kAuxPlaneBase + 4].SetAll();
    }
    if (IsHectopliesFormat(input_format)) {
      result[kAuxPlaneBase + 5].Fill(history.Last().GetRule50Ply() / 100.0f);
    } else {
      result[kAuxPlaneBase + 5].Fill(history.Last().GetRule50Ply());
    }
    // Plane kAuxPlaneBase + 6 used to be movecount plane, now it's all zeros
    // unless we need it for canonical armageddon side to move.
    if (IsCanonicalArmageddonFormat(input_format)) {
      if (we_are_black) result[kAuxPlaneBase + 6].SetAll();
    }
    // Plane kAuxPlaneBase + 7 is all ones to help NN find board edges.
    result[kAuxPlaneBase + 7].SetAll();
    if (stop_early) {
      castlings = board.castlings();
    }
  }
  bool skip_non_repeats =
      input_format ==
          pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2 ||
      input_format == pblczero::NetworkFormat::
                          INPUT_112_WITH_CANONICALIZATION_V2_ARMAGEDDON;
  bool flip = false;
  int history_idx = history.GetLength() - 1;
  for (int i = 0; i < std::min(history_planes, kMoveHistory);
       ++i, --history_idx) {
    const Position& position =
        history.GetPositionAt(history_idx < 0 ? 0 : history_idx);
    const ChessBoard& board =
        flip ? position.GetThemBoard() : position.GetBoard();
    // Castling changes can't be repeated, so we can stop early.
    if (stop_early && board.castlings().as_int() != castlings.as_int()) break;
    // Enpassants can't be repeated, but we do need to always send the current
    // position.
    if (stop_early && history_idx != history.GetLength() - 1 &&
        !board.en_passant().empty()) {
      break;
    }
    if (history_idx < 0 && fill_empty_history == FillEmptyHistory::NO) break;
    // Board may be flipped so compare with position.GetBoard().
    if (history_idx < 0 && fill_empty_history == FillEmptyHistory::FEN_ONLY &&
        position.GetBoard() == ChessBoard::kStartposBoard) {
      break;
    }
    const int repetitions = position.GetRepetitions();
    // Canonical v2 only writes an item if it is a repeat, unless its the most
    // recent position.
    if (skip_non_repeats && repetitions == 0 && i > 0) {
      if (history_idx > 0) flip = !flip;
      // If no capture no pawn is 0, the previous was start of game, capture or
      // pawn push, so there can't be any more repeats that are worth
      // considering.
      if (position.GetRule50Ply() == 0) break;
      // Decrement i so it remains the same as the history_idx decrements.
      --i;
      continue;
    }

    const int base = i * kPlanesPerBoard;
    result[base + 0].mask = (board.ours() & board.pawns()).as_int();
    result[base + 1].mask = (board.ours() & board.knights()).as_int();
    result[base + 2].mask = (board.ours() & board.bishops()).as_int();
    result[base + 3].mask = (board.ours() & board.rooks()).as_int();
    result[base + 4].mask = (board.ours() & board.queens()).as_int();
    result[base + 5].mask = (board.ours() & board.kings()).as_int();

    result[base + 6].mask = (board.theirs() & board.pawns()).as_int();
    result[base + 7].mask = (board.theirs() & board.knights()).as_int();
    result[base + 8].mask = (board.theirs() & board.bishops()).as_int();
    result[base + 9].mask = (board.theirs() & board.rooks()).as_int();
    result[base + 10].mask = (board.theirs() & board.queens()).as_int();
    result[base + 11].mask = (board.theirs() & board.kings()).as_int();

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
    // If no capture no pawn is 0, the previous was start of game, capture or
    // pawn push, so no need to go back further if stopping early.
    if (stop_early && position.GetRule50Ply() == 0) break;
  }
  if (transform != NoTransform) {
    // Transform all masks.
    for (int i = 0; i <= kAuxPlaneBase + 4; i++) {
      auto v = result[i].mask;
      if (v == 0 || v == ~0ULL) continue;
      if ((transform & FlipTransform) != 0) {
        v = ReverseBitsInBytes(v);
      }
      if ((transform & MirrorTransform) != 0) {
        v = ReverseBytesInBytes(v);
      }
      if ((transform & TransposeTransform) != 0) {
        v = TransposeBitsInBytes(v);
      }
      result[i].mask = v;
    }
  }
  if (transform_out) *transform_out = transform;
  return result;
}

Move DecodeMoveFromInput(const InputPlanes& planes, const InputPlanes& prior) {
  auto pawndiff = MaskDiffWithMirror(planes[6], prior[0]);
  auto knightdiff = MaskDiffWithMirror(planes[7], prior[1]);
  auto bishopdiff = MaskDiffWithMirror(planes[8], prior[2]);
  auto rookdiff = MaskDiffWithMirror(planes[9], prior[3]);
  auto queendiff = MaskDiffWithMirror(planes[10], prior[4]);
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
  auto kingdiff = MaskDiffWithMirror(planes[11], prior[5]);
  if (kingdiff.count() == 2) {
    if (rookdiff.count() == 2) {
      auto from = OldPosition(prior[5], kingdiff);
      auto to = OldPosition(prior[3], rookdiff);
      return Move(from, to);
    }
    auto from = OldPosition(prior[5], kingdiff);
    auto to = SingleSquare(planes[11].mask & kingdiff.as_int());
    if (std::abs(from.col() - to.col()) > 1) {
      // Chess 960 castling can leave the rook in place, but the king has moved
      // from one side of the rook to the other - thus has gone at least 2
      // squares, which is impossible for a normal king move. Can't work out the
      // rook location from rookdiff since its empty, but it is known given the
      // direction of the king movement and the knowledge that the rook hasn't
      // moved.
      if (from.col() > to.col()) {
        to = BoardSquare(from.row(), to.col() + 1);
      } else {
        to = BoardSquare(from.row(), to.col() - 1);
      }
    }
    return Move(from, to);
  }
  if (queendiff.count() == 2) {
    auto from = OldPosition(prior[4], queendiff);
    auto to = SingleSquare(planes[10].mask & queendiff.as_int());
    return Move(from, to);
  }
  if (rookdiff.count() == 2) {
    auto from = OldPosition(prior[3], rookdiff);
    auto to = SingleSquare(planes[9].mask & rookdiff.as_int());
    // Only one king, so we can simply grab its current location directly.
    auto kingpos = SingleSquare(planes[11].mask);
    if (from.row() == kingpos.row() && to.row() == kingpos.row() &&
        (from.col() < kingpos.col() && to.col() > kingpos.col() ||
         from.col() > kingpos.col() && to.col() < kingpos.col())) {
      // If the king hasn't moved, this could still be a chess 960 castling move
      // if the rook has passed through the king.
      // Destination of the castling move is where the rook started.
      to = from;
      // And since the king didn't move it forms the start position.
      from = kingpos;
    }
    return Move(from, to);
  }
  if (bishopdiff.count() == 2) {
    auto from = OldPosition(prior[2], bishopdiff);
    auto to = SingleSquare(planes[8].mask & bishopdiff.as_int());
    return Move(from, to);
  }
  if (knightdiff.count() == 2) {
    auto from = OldPosition(prior[1], knightdiff);
    auto to = SingleSquare(planes[7].mask & knightdiff.as_int());
    return Move(from, to);
  }
  if (pawndiff.count() == 2) {
    auto from = OldPosition(prior[0], pawndiff);
    auto to = SingleSquare(planes[6].mask & pawndiff.as_int());
    return Move(from, to);
  }
  assert(false);
  return Move();
}

}  // namespace lczero

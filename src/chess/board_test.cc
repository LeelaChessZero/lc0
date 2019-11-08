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
*/

#include "chess/board.h"

#include <gtest/gtest.h>

#include <iostream>

#include "chess/bitboard.h"

namespace lczero {

TEST(BoardSquare, BoardSquare) {
  {
    auto x = BoardSquare(ChessBoard::C2);
    EXPECT_EQ(x.row(), 1);
    EXPECT_EQ(x.col(), 2);
  }

  {
    auto x = BoardSquare("c2");
    EXPECT_EQ(x.row(), 1);
    EXPECT_EQ(x.col(), 2);
  }

  {
    auto x = BoardSquare(1, 2);
    EXPECT_EQ(x.row(), 1);
    EXPECT_EQ(x.col(), 2);
  }

  {
    auto x = BoardSquare(1, 2);
    x.Mirror();
    EXPECT_EQ(x.row(), 6);
    EXPECT_EQ(x.col(), 2);
  }
}

TEST(ChessBoard, PseudolegalMovesStartingPos) {
  ChessBoard board;
  board.SetFromFen(ChessBoard::kStartposFen);
  board.Mirror();
  auto moves = board.GeneratePseudolegalMoves();

  EXPECT_EQ(moves.size(), 20);
}

namespace {
int Perft(const ChessBoard& board, int max_depth, bool dump = false,
          int depth = 0) {
  if (depth == max_depth) return 1;
  int total_count = 0;
  auto moves = board.GeneratePseudolegalMoves();

  auto legal_moves = board.GenerateLegalMoves();
  auto iter = legal_moves.begin();

  for (const auto& move : moves) {
    auto new_board = board;
    new_board.ApplyMove(move);
    if (new_board.IsUnderCheck()) {
      if (iter != legal_moves.end()) {
        EXPECT_NE(iter->as_packed_int(), move.as_packed_int())
            << board.DebugString() << "legal:[" << iter->as_string()
            << "]==pseudo:(" << move.as_string() << ") Under check:\n"
            << new_board.DebugString();
      }
      continue;
    }

    EXPECT_EQ(iter->as_packed_int(), move.as_packed_int())
        << board.DebugString() << "legal:[" << iter->as_string() << "]pseudo:("
        << move.as_string() << ") after:\n"
        << new_board.DebugString();

    new_board.Mirror();
    ++iter;
    int count = Perft(new_board, max_depth, dump, depth + 1);
    if (dump && depth == 0) {
      Move m = move;
      if (board.flipped()) m.Mirror();
      std::cerr << m.as_string() << ": " << count << '\n';
    }
    total_count += count;
  }

  EXPECT_EQ(iter, legal_moves.end());
  return total_count;
}
}  // namespace

/* TEST(ChessBoard, MoveGenStartingPos) {
  ChessBoard board;
  board.SetFromFen(ChessBoard::kStartposFen);

  EXPECT_EQ(Perft(board, 0), 1);
  EXPECT_EQ(Perft(board, 1), 20);
  EXPECT_EQ(Perft(board, 2), 400);
  EXPECT_EQ(Perft(board, 3), 8902);
  EXPECT_EQ(Perft(board, 4), 197281);
  EXPECT_EQ(Perft(board, 5), 4865609);
  EXPECT_EQ(Perft(board, 6), 119060324);
} */

TEST(ChessBoard, MoveGenKiwipete) {
  ChessBoard board;
  board.SetFromFen(
      "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 1 1");

  EXPECT_EQ(Perft(board, 1), 48);
  EXPECT_EQ(Perft(board, 2), 2039);
  EXPECT_EQ(Perft(board, 3), 97862);
  EXPECT_EQ(Perft(board, 4), 4085603);
  //  EXPECT_EQ(Perft(board, 5), 193690690);
}

TEST(ChessBoard, MoveGenPosition3) {
  ChessBoard board;
  board.SetFromFen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 1 1");

  EXPECT_EQ(Perft(board, 1), 14);
  EXPECT_EQ(Perft(board, 2), 191);
  EXPECT_EQ(Perft(board, 3), 2812);
  EXPECT_EQ(Perft(board, 4), 43238);
  EXPECT_EQ(Perft(board, 5), 674624);
  EXPECT_EQ(Perft(board, 6), 11030083);
}

TEST(ChessBoard, MoveGenPosition4) {
  ChessBoard board;
  board.SetFromFen(
      "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1");

  EXPECT_EQ(Perft(board, 1), 6);
  EXPECT_EQ(Perft(board, 2), 264);
  EXPECT_EQ(Perft(board, 3), 9467);
  EXPECT_EQ(Perft(board, 4), 422333);
  EXPECT_EQ(Perft(board, 5), 15833292);
}

TEST(ChessBoard, MoveGenPosition5) {
  ChessBoard board;
  board.SetFromFen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8");

  EXPECT_EQ(Perft(board, 1), 44);
  EXPECT_EQ(Perft(board, 2), 1486);
  EXPECT_EQ(Perft(board, 3), 62379);
  EXPECT_EQ(Perft(board, 4), 2103487);
  //  EXPECT_EQ(Perft(board, 5), 89941194);
}

TEST(ChessBoard, MoveGenPosition6) {
  ChessBoard board;
  board.SetFromFen(
      "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 "
      "10");

  EXPECT_EQ(Perft(board, 1), 46);
  EXPECT_EQ(Perft(board, 2), 2079);
  EXPECT_EQ(Perft(board, 3), 89890);
  EXPECT_EQ(Perft(board, 4), 3894594);
}

namespace {
const struct {
  const char* const fen;
  const uint32_t perft[6];
} kChess960Positions[] = {
    {"bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9",
     {21, 528, 12189, 326672, 8146062, 227689589}},  // 1
    {"2nnrbkr/p1qppppp/8/1ppb4/6PP/3PP3/PPP2P2/BQNNRBKR w HEhe - 1 9",
     {21, 807, 18002, 667366, 16253601, 590751109}},  // 2
    {"b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9",
     {20, 479, 10471, 273318, 6417013, 177654692}},  // 3
    {"qbbnnrkr/2pp2pp/p7/1p2pp2/8/P3PP2/1PPP1KPP/QBBNNR1R w hf - 0 9",
     {22, 593, 13440, 382958, 9183776, 274103539}},  // 4
    {"1nbbnrkr/p1p1ppp1/3p4/1p3P1p/3Pq2P/8/PPP1P1P1/QNBBNRKR w HFhf - 0 9",
     {28, 1120, 31058, 1171749, 34030312, 1250970898}},  // 5
    {"qnbnr1kr/ppp1b1pp/4p3/3p1p2/8/2NPP3/PPP1BPPP/QNB1R1KR w HEhe - 1 9",
     {29, 899, 26578, 824055, 24851983, 775718317}},  // 6
    {"q1bnrkr1/ppppp2p/2n2p2/4b1p1/2NP4/8/PPP1PPPP/QNB1RRKB w ge - 1 9",
     {30, 860, 24566, 732757, 21093346, 649209803}},  // 7
    {"qbn1brkr/ppp1p1p1/2n4p/3p1p2/P7/6PP/QPPPPP2/1BNNBRKR w HFhf - 0 9",
     {25, 635, 17054, 465806, 13203304, 377184252}},  // 8
    {"qnnbbrkr/1p2ppp1/2pp3p/p7/1P5P/2NP4/P1P1PPP1/Q1NBBRKR w HFhf - 0 9",
     {24, 572, 15243, 384260, 11110203, 293989890}},  // 9
    {"qn1rbbkr/ppp2p1p/1n1pp1p1/8/3P4/P6P/1PP1PPPK/QNNRBB1R w hd - 2 9",
     {28, 811, 23175, 679699, 19836606, 594527992}},  // 10
    {"qnr1bkrb/pppp2pp/3np3/5p2/8/P2P2P1/NPP1PP1P/QN1RBKRB w GDg - 3 9",
     {33, 823, 26895, 713420, 23114629, 646390782}},  // 11
    {"qb1nrkbr/1pppp1p1/1n3p2/p1B4p/8/3P1P1P/PPP1P1P1/QBNNRK1R w HEhe - 0 9",
     {31, 855, 25620, 735703, 21796206, 651054626}},  // 12
    {"qnnbrk1r/1p1ppbpp/2p5/p4p2/2NP3P/8/PPP1PPP1/Q1NBRKBR w HEhe - 0 9",
     {26, 790, 21238, 642367, 17819770, 544866674}},  // 13
    {"1qnrkbbr/1pppppp1/p1n4p/8/P7/1P1N1P2/2PPP1PP/QN1RKBBR w HDhd - 0 9",
     {37, 883, 32187, 815535, 29370838, 783201510}},  // 14
    {"qn1rkrbb/pp1p1ppp/2p1p3/3n4/4P2P/2NP4/PPP2PP1/Q1NRKRBB w FDfd - 1 9",
     {24, 585, 14769, 356950, 9482310, 233468620}},  // 15
    {"bb1qnrkr/pp1p1pp1/1np1p3/4N2p/8/1P4P1/P1PPPP1P/BBNQ1RKR w HFhf - 0 9",
     {29, 864, 25747, 799727, 24219627, 776836316}},  // 16
    {"bnqbnr1r/p1p1ppkp/3p4/1p4p1/P7/3NP2P/1PPP1PP1/BNQB1RKR w HF - 0 9",
     {26, 889, 24353, 832956, 23701014, 809194268}},  // 17
    {"bnqnrbkr/1pp2pp1/p7/3pP2p/4P1P1/8/PPPP3P/BNQNRBKR w HEhe d6 0 9",
     {31, 984, 28677, 962591, 29032175, 1008880643}},  // 18
    {"b1qnrrkb/ppp1pp1p/n2p1Pp1/8/8/P7/1PPPP1PP/BNQNRKRB w GE - 0 9",
     {20, 484, 10532, 281606, 6718715, 193594729}},  // 19
    {"n1bqnrkr/pp1ppp1p/2p5/6p1/2P2b2/PN6/1PNPPPPP/1BBQ1RKR w HFhf - 2 9",
     {23, 732, 17746, 558191, 14481581, 457140569}},  // 20
    {"n1bb1rkr/qpnppppp/2p5/p7/P1P5/5P2/1P1PPRPP/NQBBN1KR w Hhf - 1 9",
     {27, 697, 18724, 505089, 14226907, 400942568}},  // 21
    {"nqb1rbkr/pppppp1p/4n3/6p1/4P3/1NP4P/PP1P1PP1/1QBNRBKR w HEhe - 1 9",
     {28, 641, 18811, 456916, 13780398, 354122358}},  // 22
    {"n1bnrrkb/pp1pp2p/2p2p2/6p1/5B2/3P4/PPP1PPPP/NQ1NRKRB w GE - 2 9",
     {28, 606, 16883, 381646, 10815324, 254026570}},  // 23
    {"nbqnbrkr/2ppp1p1/pp3p1p/8/4N2P/1N6/PPPPPPP1/1BQ1BRKR w HFhf - 0 9",
     {26, 626, 17268, 437525, 12719546, 339132046}},  // 24
    {"nq1bbrkr/pp2nppp/2pp4/4p3/1PP1P3/1B6/P2P1PPP/NQN1BRKR w HFhf - 2 9",
     {21, 504, 11812, 302230, 7697880, 207028745}},  // 25
    {"nqnrb1kr/2pp1ppp/1p1bp3/p1B5/5P2/3N4/PPPPP1PP/NQ1R1BKR w HDhd - 0 9",
     {30, 672, 19307, 465317, 13454573, 345445468}},  // 26
    {"nqn2krb/p1prpppp/1pbp4/7P/5P2/8/PPPPPKP1/NQNRB1RB w g - 3 9",
     {21, 461, 10608, 248069, 6194124, 152861936}},  // 27
    {"nb1n1kbr/ppp1rppp/3pq3/P3p3/8/4P3/1PPPRPPP/NBQN1KBR w Hh - 1 9",
     {19, 566, 11786, 358337, 8047916, 249171636}},  // 28
    {"nqnbrkbr/1ppppp1p/p7/6p1/6P1/P6P/1PPPPP2/NQNBRKBR w HEhe - 1 9",
     {20, 382, 8694, 187263, 4708975, 112278808}},  // 29
    {"nq1rkb1r/pp1pp1pp/1n2bp1B/2p5/8/5P1P/PPPPP1P1/NQNRKB1R w HDhd - 2 9",
     {24, 809, 20090, 673811, 17647882, 593457788}},  // 30
    {"nqnrkrb1/pppppp2/7p/4b1p1/8/PN1NP3/1PPP1PPP/1Q1RKRBB w FDfd - 1 9",
     {26, 683, 18102, 473911, 13055173, 352398011}},  // 31
    {"bb1nqrkr/1pp1ppp1/pn5p/3p4/8/P2NNP2/1PPPP1PP/BB2QRKR w HFhf - 0 9",
     {29, 695, 21193, 552634, 17454857, 483785639}},  // 32
    {"bnn1qrkr/pp1ppp1p/2p5/b3Q1p1/8/5P1P/PPPPP1P1/BNNB1RKR w HFhf - 2 9",
     {44, 920, 35830, 795317, 29742670, 702867204}},  // 33
    {"bnnqrbkr/pp1p2p1/2p1p2p/5p2/1P5P/1R6/P1PPPPP1/BNNQRBK1 w Ehe - 0 9",
     {33, 1022, 32724, 1024721, 32898113, 1047360456}},  // 34
    {"b1nqrkrb/2pppppp/p7/1P6/1n6/P4P2/1P1PP1PP/BNNQRKRB w GEge - 0 9",
     {23, 638, 15744, 446539, 11735969, 344211589}},  // 35
    {"n1bnqrkr/3ppppp/1p6/pNp1b3/2P3P1/8/PP1PPP1P/NBB1QRKR w HFhf - 1 9",
     {29, 728, 20768, 532084, 15621236, 415766465}},  // 36
    {"n2bqrkr/p1p1pppp/1pn5/3p1b2/P6P/1NP5/1P1PPPP1/1NBBQRKR w HFhf - 3 9",
     {20, 533, 12152, 325059, 8088751, 223068417}},  // 37
    {"nnbqrbkr/1pp1p1p1/p2p4/5p1p/2P1P3/N7/PPQP1PPP/N1B1RBKR w HEhe - 0 9",
     {27, 619, 18098, 444421, 13755384, 357222394}},  // 38
    {"nnbqrkr1/pp1pp2p/2p2b2/5pp1/1P5P/4P1P1/P1PP1P2/NNBQRKRB w GEge - 1 9",
     {32, 1046, 33721, 1111186, 36218182, 1202830851}},  // 39
    {"nb1qbrkr/p1pppp2/1p1n2pp/8/1P6/2PN3P/P2PPPP1/NB1QBRKR w HFhf - 0 9",
     {25, 521, 14021, 306427, 8697700, 201455191}},  // 40
    {"nnq1brkr/pp1pppp1/8/2p4P/8/5K2/PPPbPP1P/NNQBBR1R w hf - 0 9",
     {23, 724, 18263, 571072, 15338230, 484638597}},  // 41
    {"nnqrbb1r/pppppk2/5pp1/7p/1P6/3P2PP/P1P1PP2/NNQRBBKR w HD - 0 9",
     {30, 717, 21945, 547145, 17166700, 450069742}},  // 42
    {"nnqr1krb/p1p1pppp/2bp4/8/1p1P4/4P3/PPP2PPP/NNQRBKRB w GDgd - 0 9",
     {25, 873, 20796, 728628, 18162741, 641708630}},  // 43
    {"nbnqrkbr/p2ppp2/1p4p1/2p4p/3P3P/3N4/PPP1PPPR/NB1QRKB1 w Ehe - 0 9",
     {24, 589, 15190, 382317, 10630667, 279474189}},  // 44
    {"n1qbrkbr/p1ppp2p/2n2pp1/1p6/1P6/2P3P1/P2PPP1P/NNQBRKBR w HEhe - 0 9",
     {22, 592, 14269, 401976, 10356818, 301583306}},  // 45
    {"2qrkbbr/ppn1pppp/n1p5/3p4/5P2/P1PP4/1P2P1PP/NNQRKBBR w HDhd - 1 9",
     {27, 750, 20584, 605458, 16819085, 516796736}},  // 46
    {"1nqr1rbb/pppkp1pp/1n3p2/3p4/1P6/5P1P/P1PPPKP1/NNQR1RBB w - - 1 9",
     {24, 623, 15921, 429446, 11594634, 322745925}},  // 47
    {"bbn1rqkr/pp1pp2p/4npp1/2p5/1P6/2BPP3/P1P2PPP/1BNNRQKR w HEhe - 0 9",
     {23, 730, 17743, 565340, 14496370, 468608864}},  // 48
    {"bn1brqkr/pppp2p1/3npp2/7p/PPP5/8/3PPPPP/BNNBRQKR w HEhe - 0 9",
     {25, 673, 17835, 513696, 14284338, 434008567}},  // 49
    {"bn1rqbkr/ppp1ppp1/1n6/2p4p/7P/3P4/PPP1PPP1/BN1RQBKR w HDhd - 0 9",
     {25, 776, 20562, 660217, 18486027, 616653869}},  // 50
    {"bnnr1krb/ppp2ppp/3p4/3Bp3/q1P3PP/8/PP1PPP2/BNNRQKR1 w GDgd - 0 9",
     {29, 1040, 30772, 1053113, 31801525, 1075147725}},  // 51
    {"1bbnrqkr/pp1ppppp/8/2p5/n7/3PNPP1/PPP1P2P/NBB1RQKR w HEhe - 1 9",
     {24, 598, 15673, 409766, 11394778, 310589129}},  // 52
    {"nnbbrqkr/p2ppp1p/1pp5/8/6p1/N1P5/PPBPPPPP/N1B1RQKR w HEhe - 0 9",
     {26, 530, 14031, 326312, 8846766, 229270702}},  // 53
    {"nnbrqbkr/2p1p1pp/p4p2/1p1p4/8/NP6/P1PPPPPP/N1BRQBKR w HDhd - 0 9",
     {17, 496, 10220, 303310, 7103549, 217108001}},  // 54
    {"nnbrqk1b/pp2pprp/2pp2p1/8/3PP1P1/8/PPP2P1P/NNBRQRKB w d - 1 9",
     {33, 820, 27856, 706784, 24714401, 645835197}},  // 55
    {"1bnrbqkr/ppnpp1p1/2p2p1p/8/1P6/4PPP1/P1PP3P/NBNRBQKR w HDhd - 0 9",
     {27, 705, 19760, 548680, 15964771, 464662032}},  // 56
    {"n1rbbqkr/pp1pppp1/7p/P1p5/1n6/2PP4/1P2PPPP/NNRBBQKR w HChc - 0 9",
     {22, 631, 14978, 431801, 10911545, 320838556}},  // 57
    {"n1rqb1kr/p1pppp1p/1pn4b/3P2p1/P7/1P6/2P1PPPP/NNRQBBKR w HChc - 0 9",
     {24, 477, 12506, 263189, 7419372, 165945904}},  // 58
    {"nnrqbkrb/pppp1pp1/7p/4p3/6P1/2N2B2/PPPPPP1P/NR1QBKR1 w Ggc - 2 9",
     {29, 658, 19364, 476620, 14233587, 373744834}},  // 59
    {"n1nrqkbr/ppb2ppp/3pp3/2p5/2P3P1/5P2/PP1PPB1P/NBNRQK1R w HDhd - 1 9",
     {32, 801, 25861, 681428, 22318948, 619857455}},  // 60
    {"2rbqkbr/p1pppppp/1nn5/1p6/7P/P4P2/1PPPP1PB/NNRBQK1R w HChc - 2 9",
     {27, 647, 18030, 458057, 13189156, 354689323}},  // 61
    {"nn1qkbbr/pp2ppp1/2rp4/2p4p/P2P4/1N5P/1PP1PPP1/1NRQKBBR w HCh - 1 9",
     {24, 738, 18916, 586009, 16420659, 519075930}},  // 62
    {"nnrqk1bb/p1ppp2p/5rp1/1p3p2/1P4P1/5P1P/P1PPP3/NNRQKRBB w FCc - 1 9",
     {25, 795, 20510, 648945, 17342527, 556144017}},  // 63
    {"bb1nrkqr/ppppn2p/4ppp1/8/1P4P1/4P3/P1PPKP1P/BBNNR1QR w he - 0 9",
     {29, 664, 20024, 498376, 15373803, 406016364}},  // 64
    {"bnnbrkqr/1p1ppp2/8/p1p3pp/1P6/N4P2/PBPPP1PP/2NBRKQR w HEhe - 0 9",
     {31, 770, 24850, 677212, 22562080, 662029574}},  // 65
    {"1nnrkbqr/p1pp1ppp/4p3/1p6/1Pb1P3/6PB/P1PP1P1P/BNNRK1QR w HDhd - 0 9",
     {27, 776, 22133, 641002, 19153245, 562738257}},  // 66
    {"bnr1kqrb/pppp1pp1/1n5p/4p3/P3P3/3P2P1/1PP2P1P/BNNRKQRB w GDg - 0 9",
     {26, 624, 16411, 435426, 11906515, 338092952}},  // 67
    {"nbbnrkqr/p1ppp1pp/1p3p2/8/2P5/4P3/PP1P1PPP/NBBNRKQR w HEhe - 1 9",
     {25, 624, 15561, 419635, 10817378, 311138112}},  // 68
    {"nn1brkqr/pp1bpppp/8/2pp4/P4P2/1PN5/2PPP1PP/N1BBRKQR w HEhe - 1 9",
     {23, 659, 16958, 476567, 13242252, 373557073}},  // 69
    {"n1brkbqr/ppp1pp1p/6pB/3p4/2Pn4/8/PP2PPPP/NN1RKBQR w HDhd - 0 9",
     {32, 1026, 30360, 978278, 29436320, 957904151}},  // 70
    {"nnbrkqrb/p2ppp2/Q5pp/1pp5/4PP2/2N5/PPPP2PP/N1BRK1RB w GDgd - 0 9",
     {36, 843, 29017, 715537, 24321197, 630396940}},  // 71
    {"nbnrbk1r/pppppppq/8/7p/8/1N2QPP1/PPPPP2P/NB1RBK1R w HDhd - 2 9",
     {36, 973, 35403, 1018054, 37143354, 1124883780}},  // 72
    {"nnrbbkqr/2pppp1p/p7/6p1/1p2P3/4QPP1/PPPP3P/NNRBBK1R w HChc - 0 9",
     {36, 649, 22524, 489526, 16836636, 416139320}},  // 73
    {"nnrkbbqr/1p2pppp/p2p4/2p5/8/1N2P1P1/PPPP1P1P/1NKRBBQR w hc - 0 9",
     {26, 672, 18136, 477801, 13342771, 363074681}},  // 74
    {"n1rkbqrb/pp1ppp2/2n3p1/2p4p/P5PP/1P6/2PPPP2/NNRKBQRB w GCgc - 0 9",
     {24, 804, 20712, 684001, 18761475, 617932151}},  // 75
    {"nbkr1qbr/1pp1pppp/pn1p4/8/3P2P1/5R2/PPP1PP1P/NBN1KQBR w H - 2 9",
     {30, 627, 18669, 423329, 12815016, 312798696}},  // 76
    {"nnr1kqbr/pp1pp1p1/2p5/b4p1p/P7/1PNP4/2P1PPPP/N1RBKQBR w HChc - 1 9",
     {12, 421, 6530, 227044, 4266410, 149176979}},  // 77
    {"n1rkqbbr/p1pp1pp1/np2p2p/8/8/N4PP1/PPPPP1BP/N1RKQ1BR w HChc - 0 9",
     {27, 670, 19119, 494690, 14708490, 397268628}},  // 78
    {"nnr1qrbb/p2kpppp/1p1p4/2p5/6P1/PP1P4/2P1PP1P/NNRKQRBB w FC - 0 9",
     {27, 604, 17043, 409665, 11993332, 308518181}},  // 79
    {"bbnnrkrq/ppp1pp2/6p1/3p4/7p/7P/PPPPPPP1/BBNNRRKQ w ge - 0 9",
     {20, 559, 12242, 355326, 8427161, 252274233}},  // 80
    {"bnnbrkr1/ppp2p1p/5q2/3pp1p1/4P3/1N4P1/PPPPRP1P/BN1B1KRQ w Gge - 0 9",
     {26, 1036, 27228, 1028084, 28286576, 1042120495}},  // 81
    {"bn1rkbrq/1pppppp1/p6p/1n6/3P4/6PP/PPPRPP2/BNN1KBRQ w Ggd - 2 9",
     {29, 633, 19278, 455476, 14333034, 361900466}},  // 82
    {"b1nrkrqb/1p1npppp/p2p4/2p5/5P2/4P2P/PPPP1RP1/BNNRK1QB w Dfd - 1 9",
     {25, 475, 12603, 270909, 7545536, 179579818}},  // 83
    {"1bbnrkrq/ppppppp1/8/7p/1n4P1/1PN5/P1PPPP1P/NBBR1KRQ w Gge - 0 9",
     {30, 803, 25473, 709716, 23443854, 686365049}},  // 84
    {"nnbbrkrq/2pp1pp1/1p5p/pP2p3/7P/N7/P1PPPPP1/N1BBRKRQ w GEge - 0 9",
     {18, 432, 9638, 242350, 6131124, 160393505}},  // 85
    {"nnbrkbrq/1pppp1p1/p7/7p/1P2Pp2/BN6/P1PP1PPP/1N1RKBRQ w GDgd - 0 9",
     {27, 482, 13441, 282259, 8084701, 193484216}},  // 86
    {"n1brkrqb/pppp3p/n3pp2/6p1/3P1P2/N1P5/PP2P1PP/N1BRKRQB w FDfd - 0 9",
     {28, 642, 19005, 471729, 14529434, 384837696}},  // 87
    {"nbnrbk2/p1pppp1p/1p3qr1/6p1/1B1P4/1N6/PPP1PPPP/1BNR1RKQ w d - 2 9",
     {30, 796, 22780, 687302, 20120565, 641832725}},  // 88
    {"nnrbbrkq/1pp2ppp/3p4/p3p3/3P1P2/1P2P3/P1P3PP/NNRBBKRQ w GC - 1 9",
     {31, 827, 24538, 663082, 19979594, 549437308}},  // 89
    {"nnrkbbrq/1pp2p1p/p2pp1p1/2P5/8/8/PP1PPPPP/NNRKBBRQ w Ggc - 0 9",
     {24, 762, 19283, 624598, 16838099, 555230555}},  // 90
    {"nnr1brqb/1ppkp1pp/8/p2p1p2/1P1P4/N1P5/P3PPPP/N1RKBRQB w FC - 1 9",
     {23, 640, 15471, 444905, 11343507, 334123513}},  // 91
    {"nbnrkrbq/2ppp2p/p4p2/1P4p1/4PP2/8/1PPP2PP/NBNRKRBQ w FDfd - 0 9",
     {31, 826, 26137, 732175, 23555139, 686250413}},  // 92
    {"1nrbkr1q/1pppp1pp/1n6/p4p2/N1b4P/8/PPPPPPPB/N1RBKR1Q w FCfc - 2 9",
     {27, 862, 24141, 755171, 22027695, 696353497}},  // 93
    {"nnrkrbbq/pppp2pp/8/4pp2/4P3/P7/1PPPBPPP/NNKRR1BQ w c - 0 9",
     {25, 792, 19883, 636041, 16473376, 532214177}},  // 94
    {"n1rk1qbb/pppprpp1/2n4p/4p3/2PP3P/8/PP2PPP1/NNRKRQBB w ECc - 1 9",
     {25, 622, 16031, 425247, 11420973, 321855685}},  // 95
    {"bbq1rnkr/pnp1pp1p/1p1p4/6p1/2P5/2Q1P2P/PP1P1PP1/BB1NRNKR w HEhe - 2 9",
     {36, 870, 30516, 811047, 28127620, 799738334}},  // 96
    {"bq1brnkr/1p1ppp1p/1np5/p5p1/8/1N5P/PPPPPPP1/BQ1BRNKR w HEhe - 0 9",
     {22, 588, 13524, 380068, 9359618, 273795898}},  // 97
    {"bq1rn1kr/1pppppbp/Nn4p1/8/8/P7/1PPPPPPP/BQ1RNBKR w HDhd - 1 9",
     {24, 711, 18197, 542570, 14692779, 445827351}},  // 98
    {"bqnr1kr1/pppppp1p/6p1/5n2/4B3/3N2PP/PbPPPP2/BQNR1KR1 w GDgd - 2 9",
     {31, 1132, 36559, 1261476, 43256823, 1456721391}},  // 99
    {"qbb1rnkr/ppp3pp/4n3/3ppp2/1P3PP1/8/P1PPPN1P/QBB1RNKR w HEhe - 0 9",
     {28, 696, 20502, 541886, 16492398, 456983120}},  // 100
    {"qnbbr1kr/pp1ppp1p/4n3/6p1/2p3P1/2PP1P2/PP2P2P/QNBBRNKR w HEhe - 0 9",
     {25, 655, 16520, 450189, 11767038, 335414976}},  // 101
    {"1nbrnbkr/p1ppp1pp/1p6/5p2/4q1PP/3P4/PPP1PP2/QNBRNBKR w HDhd - 1 9",
     {30, 1162, 33199, 1217278, 36048727, 1290346802}},  // 102
    {"q1brnkrb/p1pppppp/n7/1p6/P7/3P1P2/QPP1P1PP/1NBRNKRB w GDgd - 0 9",
     {32, 827, 26106, 718243, 23143989, 673147648}},  // 103
    {"qbnrb1kr/ppp1pp1p/3p4/2n3p1/1P6/6N1/P1PPPPPP/QBNRB1KR w HDhd - 2 9",
     {29, 751, 23132, 610397, 19555214, 530475036}},  // 104
    {"q1rbbnkr/pppp1p2/2n3pp/2P1p3/3P4/8/PP1NPPPP/Q1RBBNKR w HChc - 2 9",
     {29, 806, 24540, 687251, 21694330, 619907316}},  // 105
    {"q1r1bbkr/pnpp1ppp/2n1p3/1p6/2P2P2/2N1N3/PP1PP1PP/Q1R1BBKR w HChc - 2 9",
     {32, 1017, 32098, 986028, 31204371, 958455898}},  // 106
    {"2rnbkrb/pqppppp1/1pn5/7p/2P5/P1R5/QP1PPPPP/1N1NBKRB w Ggc - 4 9",
     {26, 625, 16506, 434635, 11856964, 336672890}},  // 107
    {"qbnr1kbr/p2ppppp/2p5/1p6/4n2P/P4N2/1PPP1PP1/QBNR1KBR w HDhd - 0 9",
     {27, 885, 23828, 767273, 21855658, 706272554}},  // 108
    {"qnrbnk1r/pp1pp2p/5p2/2pbP1p1/3P4/1P6/P1P2PPP/QNRBNKBR w HChc - 0 9",
     {26, 954, 24832, 892456, 24415089, 866744329}},  // 109
    {"qnrnk1br/p1p2ppp/8/1pbpp3/8/PP2N3/1QPPPPPP/1NR1KBBR w HChc - 0 9",
     {26, 783, 20828, 634267, 17477825, 539674275}},  // 110
    {"qnrnkrbb/Bpppp2p/6p1/5p2/5P2/3PP3/PPP3PP/QNRNKR1B w FCfc - 1 9",
     {28, 908, 25730, 861240, 25251641, 869525254}},  // 111
    {"bbnqrn1r/ppppp2k/5p2/6pp/7P/1QP5/PP1PPPP1/B1N1RNKR w HE - 0 9",
     {33, 643, 21790, 487109, 16693640, 410115900}},  // 112
    {"b1qbrnkr/ppp1pp2/2np4/6pp/4P3/2N4P/PPPP1PP1/BQ1BRNKR w HEhe - 0 9",
     {28, 837, 24253, 745617, 22197063, 696399065}},  // 113
    {"bnqr1bkr/pp1ppppp/2p5/4N3/5P2/P7/1PPPPnPP/BNQR1BKR w HDhd - 3 9",
     {25, 579, 13909, 341444, 8601011, 225530258}},  // 114
    {"b1qr1krb/pp1ppppp/n2n4/8/2p5/2P3P1/PP1PPP1P/BNQRNKRB w GDgd - 0 9",
     {28, 707, 19721, 549506, 15583376, 468399900}},  // 115
    {"nbbqr1kr/1pppp1pp/8/p1n2p2/4P3/PN6/1PPPQPPP/1BB1RNKR w HEhe - 0 9",
     {30, 745, 23416, 597858, 19478789, 515473678}},  // 116
    {"nqbbrn1r/p1pppp1k/1p4p1/7p/4P3/1R3B2/PPPP1PPP/NQB2NKR w H - 0 9",
     {24, 504, 13512, 317355, 9002073, 228726497}},  // 117
    {"nqbr1bkr/p1p1ppp1/1p1n4/3pN2p/1P6/8/P1PPPPPP/NQBR1BKR w HDhd - 0 9",
     {29, 898, 26532, 809605, 24703467, 757166494}},  // 118
    {"nqbrn1rb/pppp1kp1/5p1p/4p3/P4B2/3P2P1/1PP1PP1P/NQ1RNKRB w GD - 0 9",
     {34, 671, 22332, 473110, 15556806, 353235120}},  // 119
    {"nb1r1nkr/ppp1ppp1/2bp4/7p/3P2qP/P6R/1PP1PPP1/NBQRBNK1 w Dhd - 1 9",
     {38, 1691, 60060, 2526992, 88557078, 3589649998}},  // 120
    {"n1rbbnkr/1p1pp1pp/p7/2p1qp2/1B3P2/3P4/PPP1P1PP/NQRB1NKR w HChc - 0 9",
     {24, 913, 21595, 807544, 19866918, 737239330}},  // 121
    {"nqrnbbkr/p2p1p1p/1pp5/1B2p1p1/1P3P2/4P3/P1PP2PP/NQRNB1KR w HChc - 0 9",
     {33, 913, 30159, 843874, 28053260, 804687975}},  // 122
    {"nqr1bkrb/ppp1pp2/2np2p1/P6p/8/2P4P/1P1PPPP1/NQRNBKRB w GCgc - 0 9",
     {24, 623, 16569, 442531, 12681936, 351623879}},  // 123
    {"nb1rnkbr/pqppppp1/1p5p/8/1PP4P/8/P2PPPP1/NBQRNKBR w HDhd - 1 9",
     {31, 798, 24862, 694386, 22616076, 666227466}},  // 124
    {"nqrbnkbr/2p1p1pp/3p4/pp3p2/6PP/3P1N2/PPP1PP2/NQRB1KBR w HChc - 0 9",
     {24, 590, 14409, 383690, 9698432, 274064911}},  // 125
    {"nqrnkbbr/pp1p1p1p/4p1p1/1p6/8/5P1P/P1PPP1P1/NQRNKBBR w HChc - 0 9",
     {30, 1032, 31481, 1098116, 34914919, 1233362066}},  // 126
    {"nqrnkrbb/p2ppppp/1p6/2p5/2P3P1/5P2/PP1PPN1P/NQR1KRBB w FCfc - 1 9",
     {30, 775, 23958, 668000, 21141738, 621142773}},  // 127
    {"bbnrqrk1/pp2pppp/4n3/2pp4/P7/1N5P/BPPPPPP1/B2RQNKR w HD - 2 9",
     {23, 708, 17164, 554089, 14343443, 481405144}},  // 128
    {"bnr1qnkr/p1pp1p1p/1p4p1/4p1b1/2P1P3/1P6/PB1P1PPP/1NRBQNKR w HChc - 1 9",
     {30, 931, 29249, 921746, 30026687, 968109774}},  // 129
    {"b1rqnbkr/ppp1ppp1/3p3p/2n5/P3P3/2NP4/1PP2PPP/B1RQNBKR w HChc - 0 9",
     {24, 596, 15533, 396123, 11099382, 294180723}},  // 130
    {"bnrqnr1b/pp1pkppp/2p1p3/P7/2P5/7P/1P1PPPP1/BNRQNKRB w GC - 0 9",
     {24, 572, 15293, 390903, 11208688, 302955778}},  // 131
    {"n1brq1kr/bppppppp/p7/8/4P1Pn/8/PPPP1P2/NBBRQNKR w HDhd - 0 9",
     {20, 570, 13139, 371247, 9919113, 284592289}},  // 132
    {"1rbbqnkr/ppn1ppp1/3p3p/2p5/3P4/1N4P1/PPPBPP1P/1R1BQNKR w HBhb - 0 9",
     {29, 1009, 29547, 1040816, 31059587, 1111986835}},  // 133
    {"nrbq2kr/ppppppb1/5n1p/5Pp1/8/P5P1/1PPPP2P/NRBQNBKR w HBhb - 1 9",
     {20, 520, 11745, 316332, 7809837, 216997152}},  // 134
    {"nrb1nkrb/pp3ppp/1qBpp3/2p5/8/P5P1/1PPPPP1P/NRBQNKR1 w GBgb - 2 9",
     {32, 850, 25642, 734088, 21981567, 664886187}},  // 135
    {"1br1bnkr/ppqppp1p/1np3p1/8/1PP4P/4N3/P2PPPP1/NBRQB1KR w HChc - 1 9",
     {32, 798, 24765, 691488, 22076141, 670296871}},  // 136
    {"nrqbb1kr/1p1pp1pp/2p3n1/p4p2/3PP3/P5N1/1PP2PPP/NRQBB1KR w HBhb - 0 9",
     {32, 791, 26213, 684890, 23239122, 634260266}},  // 137
    {"nrqn1bkr/ppppp1pp/4b3/8/4P1p1/5P2/PPPP3P/NRQNBBKR w HBhb - 0 9",
     {29, 687, 20223, 506088, 15236287, 398759980}},  // 138
    {"nrqnbrkb/pppp1p2/4p2p/3B2p1/8/1P4P1/PQPPPP1P/NR1NBKR1 w GB - 0 9",
     {37, 764, 27073, 610950, 21284835, 514864869}},  // 139
    {"nbrq1kbr/Bp3ppp/2pnp3/3p4/5P2/2P4P/PP1PP1P1/NBRQNK1R w HChc - 0 9",
     {40, 1271, 48022, 1547741, 56588117, 1850696281}},  // 140
    {"nrqbnkbr/1p2ppp1/p1p4p/3p4/1P6/8/PQPPPPPP/1RNBNKBR w HBhb - 0 9",
     {28, 757, 23135, 668025, 21427496, 650939962}},  // 141
    {"nrqn1bbr/2ppkppp/4p3/pB6/8/2P1P3/PP1P1PPP/NRQNK1BR w HB - 1 9",
     {27, 642, 17096, 442653, 11872805, 327545120}},  // 142
    {"nrqnkrb1/p1ppp2p/1p4p1/4bp2/4PP1P/4N3/PPPP2P1/NRQ1KRBB w FBfb - 1 9",
     {27, 958, 27397, 960350, 28520172, 995356563}},  // 143
    {"1bnrnqkr/pbpp2pp/8/1p2pp2/P6P/3P1N2/1PP1PPP1/BBNR1QKR w HDhd - 0 9",
     {27, 859, 23475, 773232, 21581178, 732696327}},  // 144
    {"b1rbnqkr/1pp1ppp1/2n4p/p2p4/5P2/1PBP4/P1P1P1PP/1NRBNQKR w HChc - 0 9",
     {26, 545, 14817, 336470, 9537260, 233549184}},  // 145
    {"1nrnqbkr/p1pppppp/1p6/8/2b2P2/P1N5/1PP1P1PP/BNR1QBKR w HChc - 2 9",
     {24, 668, 17716, 494866, 14216070, 406225409}},  // 146
    {"1nrnqkrb/2ppp1pp/p7/1p3p2/5P2/N5K1/PPPPP2P/B1RNQ1RB w gc - 0 9",
     {33, 725, 23572, 559823, 18547476, 471443091}},  // 147
    {"nbbr1qkr/p1pppppp/8/1p1n4/3P4/1N3PP1/PPP1P2P/1BBRNQKR w HDhd - 1 9",
     {28, 698, 20527, 539625, 16555068, 458045505}},  // 148
    {"1rbbnqkr/1pnppp1p/p5p1/2p5/2P4P/5P2/PP1PP1PR/NRBBNQK1 w Bhb - 1 9",
     {24, 554, 14221, 362516, 9863080, 269284081}},  // 149
    {"nrb1qbkr/2pppppp/2n5/p7/2p5/4P3/PPNP1PPP/1RBNQBKR w HBhb - 0 9",
     {23, 618, 15572, 443718, 12044358, 360311412}},  // 150
    {"nrb1qkrb/2ppppp1/p3n3/1p1B3p/2P5/6P1/PP1PPPRP/NRBNQK2 w Bgb - 2 9",
     {27, 593, 16770, 401967, 11806808, 303338935}},  // 151
    {"nbrn1qkr/ppp1pp2/3p2p1/3Q3P/b7/8/PPPPPP1P/NBRNB1KR w HChc - 2 9",
     {39, 1056, 40157, 1133446, 42201531, 1239888683}},  // 152
    {"nr1bbqkr/pp1pp2p/1n3pp1/2p5/8/1P4P1/P1PPPPQP/NRNBBK1R w hb - 0 9",
     {25, 585, 15719, 406544, 11582539, 320997679}},  // 153
    {"nr2bbkr/ppp1pppp/1n1p4/8/6PP/1NP4q/PP1PPP2/1RNQBBKR w HBhb - 1 9",
     {22, 742, 15984, 545231, 13287051, 457010195}},  // 154
    {"1rnqbkrb/ppp1p1p1/1n3p2/3p3p/P6P/4P3/1PPP1PP1/NRNQBRKB w gb - 0 9",
     {22, 574, 14044, 379648, 9968830, 281344367}},  // 155
    {"nb1rqkbr/1pppp1pp/4n3/p4p2/6PP/5P2/PPPPPN2/NBR1QKBR w HCh - 0 9",
     {25, 621, 16789, 462600, 13378840, 396575613}},  // 156
    {"nrnbqkbr/2pp2pp/4pp2/pp6/8/1P3P2/P1PPPBPP/NRNBQ1KR w hb - 0 9",
     {25, 656, 16951, 466493, 12525939, 358763789}},  // 157
    {"nrnqkbbr/ppppp1p1/7p/5p2/8/P4PP1/NPPPP2P/NR1QKBBR w HBhb - 0 9",
     {28, 723, 20621, 547522, 15952533, 439046803}},  // 158
    {"1rnqkr1b/ppppp2p/1n3pp1/8/2P3P1/Pb1N4/1P1PPP1P/NR1QKRBB w FBfb - 0 9",
     {26, 713, 19671, 548875, 15865528, 454532806}},  // 159
    {"bbnrnkqr/1pppp1pp/5p2/p7/7P/1P6/PBPPPPPR/1BNRNKQ1 w D - 2 9",
     {26, 649, 17834, 502279, 14375839, 435585252}},  // 160
    {"bnrbk1qr/1ppp1ppp/p2np3/8/P7/2N2P2/1PPPP1PP/B1RBNKQR w HC - 0 9",
     {26, 621, 17569, 451452, 13514201, 364421088}},  // 161
    {"br1nkbqr/ppppppp1/8/n6p/8/N1P2PP1/PP1PP2P/B1RNKBQR w HCh - 1 9",
     {29, 664, 20182, 512316, 16125924, 442508159}},  // 162
    {"bnr1kqrb/pp1pppp1/2n5/2p5/1P4Pp/4N3/P1PPPP1P/BNKR1QRB w gc - 0 9",
     {36, 888, 31630, 789863, 27792175, 719015345}},  // 163
    {"1bbrnkqr/pp1p1ppp/2p1p3/1n6/5P2/3Q4/PPPPP1PP/NBBRNK1R w HDhd - 2 9",
     {36, 891, 31075, 781792, 26998966, 702903862}},  // 164
    {"nrbbnk1r/pp2pppq/8/2pp3p/3P2P1/1N6/PPP1PP1P/1RBBNKQR w HBhb - 0 9",
     {29, 1036, 31344, 1139166, 35627310, 1310683359}},  // 165
    {"nr1nkbqr/ppp3pp/5p2/3pp3/6b1/3PP3/PPP2PPP/NRBNKBQR w hb - 0 9",
     {18, 664, 13306, 483892, 10658989, 386307449}},  // 166
    {"nrbnk1rb/ppp1pq1p/3p4/5pp1/2P1P3/1N6/PP1PKPPP/1RBN1QRB w gb - 2 9",
     {25, 966, 24026, 920345, 23957242, 913710194}},  // 167
    {"1brnbkqr/pppppp2/6p1/7p/1Pn5/P1NP4/2P1PPPP/NBR1BKQR w HChc - 0 9",
     {22, 627, 13760, 395829, 9627826, 285900573}},  // 168
    {"nrnbbk1r/p1pppppq/8/7p/1p6/P5PP/1PPPPPQ1/NRNBBK1R w HBhb - 2 9",
     {29, 888, 26742, 874270, 27229468, 930799376}},  // 169
    {"n1nkb1qr/prppppbp/6p1/1p6/2P2P2/P7/1P1PP1PP/NRNKBBQR w HBh - 1 9",
     {29, 804, 24701, 688520, 21952444, 623156747}},  // 170
    {"nr2bqrb/ppkpp1pp/1np5/5p1P/5P2/2P5/PP1PP1P1/NRNKBQRB w GB - 0 9",
     {22, 530, 13055, 347657, 9244693, 264088392}},  // 171
    {"nbr1kqbr/p3pppp/2ppn3/1p4P1/4P3/1P6/P1PP1P1P/NBRNKQBR w HChc - 1 9",
     {23, 555, 14291, 350917, 9692630, 247479180}},  // 172
    {"nr1bkqbr/1p1pp1pp/pnp2p2/8/6P1/P1PP4/1P2PP1P/NRNBKQBR w HBhb - 0 9",
     {22, 565, 13343, 365663, 9305533, 268612479}},  // 173
    {"nr1kqbbr/np2pppp/p1p5/1B1p1P2/8/4P3/PPPP2PP/NRNKQ1BR w HBhb - 0 9",
     {32, 730, 23391, 556995, 18103280, 454569900}},  // 174
    {"nrnk1rbb/p1p2ppp/3pq3/Qp2p3/1P1P4/8/P1P1PPPP/NRN1KRBB w fb - 2 9",
     {28, 873, 25683, 791823, 23868737, 747991356}},  // 175
    {"bbnrnkrq/pp1ppp1p/6p1/2p5/6P1/P5RP/1PPPPP2/BBNRNK1Q w Dgd - 3 9",
     {37, 1260, 45060, 1542086, 54843403, 1898432768}},  // 176
    {"bnrb1rkq/ppnpppp1/3Q4/2p4p/7P/N7/PPPPPPP1/B1RBNKR1 w GC - 2 9",
     {38, 878, 31944, 800440, 28784300, 784569826}},  // 177
    {"bnrnkbrq/p1ppppp1/1p5p/8/P2PP3/5P2/1PP3PP/BNRNKBRQ w GCgc - 1 9",
     {26, 617, 16992, 419099, 11965544, 311309576}},  // 178
    {"bnrnkrqb/pp2p2p/2pp1pp1/8/P7/2PP1P2/1P2P1PP/BNRNKRQB w FCfc - 0 9",
     {26, 721, 19726, 560824, 15966934, 467132503}},  // 179
    {"nbbrnkr1/1pppp1p1/p6q/P4p1p/8/5P2/1PPPP1PP/NBBRNRKQ w gd - 2 9",
     {18, 556, 10484, 316634, 6629293, 202528241}},  // 180
    {"nrb1nkrq/2pp1ppp/p4b2/1p2p3/P4B2/3P4/1PP1PPPP/NR1BNRKQ w gb - 0 9",
     {24, 562, 14017, 355433, 9227883, 247634489}},  // 181
    {"nrbnkbrq/p3p1pp/1p6/2pp1P2/8/3PP3/PPP2P1P/NRBNKBRQ w GBgb - 0 9",
     {31, 746, 24819, 608523, 21019301, 542954168}},  // 182
    {"nrbnkrqb/pppp1p1p/4p1p1/8/7P/2P1P3/PPNP1PP1/1RBNKRQB w FBfb - 0 9",
     {20, 459, 9998, 242762, 5760165, 146614723}},  // 183
    {"nbrn1krq/ppp1p2p/6b1/3p1pp1/8/4N1PP/PPPPPP2/NBR1BRKQ w gc - 1 9",
     {27, 835, 23632, 766397, 22667987, 760795567}},  // 184
    {"nrnbbkrq/p1pp2pp/5p2/1p6/2P1pP1B/1P6/P2PP1PP/NRNB1KRQ w GBgb - 0 9",
     {24, 646, 16102, 444472, 11489727, 324948755}},  // 185
    {"nrn1bbrq/1ppkppp1/p2p3p/8/1P3N2/4P3/P1PP1PPP/NR1KBBRQ w GB - 2 9",
     {32, 591, 18722, 381683, 12069159, 269922838}},  // 186
    {"n1krbrqb/1ppppppp/p7/8/4n3/P4P1P/1PPPPQP1/NRNKBR1B w FB - 2 9",
     {26, 639, 16988, 417190, 12167153, 312633873}},  // 187
    {"n1rnkrbq/1p1ppp1p/8/p1p1b1p1/3PQ1P1/4N3/PPP1PP1P/NBR1KRB1 w FCfc - 0 9",
     {35, 1027, 35731, 1040417, 35738410, 1060661628}},  // 188
    {"nrnbkrbq/2pp1pp1/pp6/4p2p/P7/5PPP/1PPPP3/NRNBKRBQ w FBfb - 0 9",
     {26, 628, 16731, 436075, 11920087, 331498921}},  // 189
    {"1rnkrbbq/pp1p2pp/1n3p2/1Bp1p3/1P6/1N2P3/P1PP1PPP/1RNKR1BQ w EBeb - 0 9",
     {33, 992, 32244, 983481, 31703749, 980306735}},  // 190
    {"nr1krqbb/p1ppppp1/8/1p5p/1Pn5/5P2/P1PPP1PP/NRNKRQBB w EBeb - 0 9",
     {24, 670, 15985, 445492, 11371067, 325556465}},  // 191
    {"bbq1rkr1/1ppppppp/p1n2n2/8/2P2P2/1P6/PQ1PP1PP/BB1NRKNR w HEe - 3 9",
     {32, 794, 26846, 689334, 24085223, 645633370}},  // 192
    {"b1nbrknr/1qppp1pp/p4p2/1p6/6P1/P2NP3/1PPP1P1P/BQ1BRKNR w HEhe - 1 9",
     {25, 663, 17138, 482994, 13157826, 389603029}},  // 193
    {"bqnrk1nr/pp2ppbp/6p1/2pp4/2P5/5P2/PPQPP1PP/B1NRKBNR w HDhd - 0 9",
     {26, 850, 22876, 759768, 21341087, 719712622}},  // 194
    {"bqnrknrb/1ppp1p1p/p7/6p1/1P2p3/P1PN4/3PPPPP/BQ1RKNRB w GDgd - 0 9",
     {25, 721, 19290, 581913, 16391601, 511725087}},  // 195
    {"q1b1rknr/pp1pppp1/4n2p/2p1b3/1PP5/4P3/PQ1P1PPP/1BBNRKNR w HEhe - 1 9",
     {32, 975, 32566, 955493, 32649943, 962536105}},  // 196
    {"qnbbrknr/1p1ppppp/8/p1p5/5P2/PP1P4/2P1P1PP/QNBBRKNR w HEhe - 0 9",
     {27, 573, 16331, 391656, 11562434, 301166330}},  // 197
    {"q1brkb1r/p1pppppp/np3B2/8/6n1/1P5N/P1PPPPPP/QN1RKB1R w HDhd - 0 9",
     {32, 984, 31549, 1007217, 32597704, 1075429389}},  // 198
    {"qn1rk1rb/p1pppppp/1p2n3/8/2b5/4NPP1/PPPPP1RP/QNBRK2B w Dgd - 4 9",
     {22, 802, 19156, 697722, 17761431, 650603534}},  // 199
    {"qbnrbknr/ppp2p1p/8/3pp1p1/1PP1B3/5N2/P2PPPPP/Q1NRBK1R w HDhd - 0 9",
     {34, 943, 32506, 930619, 32523099, 955802240}},  // 200
    {"qnrbb1nr/pp1p1ppp/2p2k2/4p3/4P3/5PPP/PPPP4/QNRBBKNR w HC - 0 9",
     {20, 460, 10287, 241640, 5846781, 140714047}},  // 201
    {"qnr1bbnr/ppk1p1pp/3p4/2p2p2/8/2P5/PP1PPPPP/QNKRBBNR w - - 1 9",
     {19, 572, 11834, 357340, 7994547, 243724815}},  // 202
    {"qnrkbnrb/1p1p1ppp/2p5/4p3/p7/N1BP4/PPP1PPPP/Q1R1KNRB w gc - 0 9",
     {27, 579, 16233, 375168, 10845146, 268229097}},  // 203
    {"qbnrkn1r/1pppp1p1/p3bp2/2BN3p/8/5P2/PPPPP1PP/QBNRK2R w HDhd - 0 9",
     {40, 1027, 38728, 1059229, 38511307, 1104094381}},  // 204
    {"qnrbknbr/1pp2ppp/4p3/p6N/2p5/8/PPPPPPPP/Q1RBK1BR w HChc - 0 9",
     {22, 510, 11844, 300180, 7403327, 200581103}},  // 205
    {"1qkrnbbr/p1pppppp/2n5/1p6/8/5NP1/PPPPPP1P/QNRK1BBR w HC - 4 9",
     {24, 549, 13987, 352037, 9396521, 255676649}},  // 206
    {"q1rknr1b/1ppppppb/2n5/p2B3p/8/1PN3P1/P1PPPP1P/Q1RKNRB1 w FCfc - 3 9",
     {31, 924, 28520, 861944, 27463479, 847726572}},  // 207
    {"bbnqrk1r/pp1pppp1/2p4p/8/6n1/1N1P1P2/PPP1P1PP/BBQ1RKNR w HEhe - 4 9",
     {24, 804, 20147, 666341, 18024195, 595947631}},  // 208
    {"bn1brknr/ppp1p1pp/5p2/3p4/6qQ/3P3P/PPP1PPP1/BN1BRKNR w HEhe - 4 9",
     {25, 854, 22991, 704173, 20290974, 600195008}},  // 209
    {"1nqrkbnr/2pp1ppp/pp2p3/3b4/2P5/N7/PP1PPPPP/B1QRKBNR w HDhd - 0 9",
     {22, 651, 16173, 479152, 13133439, 390886040}},  // 210
    {"bnqrk1rb/1pp1pppp/p2p4/4n3/2PPP3/8/PP3PPP/BNQRKNRB w GDgd - 1 9",
     {30, 950, 28169, 889687, 27610213, 880739164}},  // 211
    {"nbb1rknr/1ppq1ppp/3p4/p3p3/4P3/1N2R3/PPPP1PPP/1BBQ1KNR w Hhe - 2 9",
     {33, 988, 31293, 967575, 30894863, 985384035}},  // 212
    {"nqbbrknr/2ppp2p/pp4p1/5p2/7P/3P1P2/PPPBP1P1/NQ1BRKNR w HEhe - 0 9",
     {27, 492, 13266, 276569, 7583292, 175376176}},  // 213
    {"1qbrkb1r/pppppppp/8/3n4/4P1n1/PN6/1PPP1P1P/1QBRKBNR w HDhd - 3 9",
     {28, 800, 21982, 630374, 17313279, 507140861}},  // 214
    {"1qbrknrb/1p1ppppp/1np5/8/p4P1P/4P1N1/PPPP2P1/NQBRK1RB w GDgd - 0 9",
     {21, 482, 10581, 267935, 6218644, 168704845}},  // 215
    {"nbqrbkr1/ppp1pppp/8/3p4/6n1/2P2PPN/PP1PP2P/NBQRBK1R w HDd - 1 9",
     {29, 921, 25748, 840262, 24138518, 806554650}},  // 216
    {"nqrb1knr/1ppbpp1p/p7/3p2p1/2P3P1/5P1P/PP1PP3/NQRBBKNR w HChc - 1 9",
     {31, 803, 25857, 665799, 21998733, 583349773}},  // 217
    {"1qrkbbr1/pppp1ppp/1n3n2/4p3/5P2/1N6/PPPPP1PP/1QRKBBNR w HCc - 0 9",
     {25, 715, 19118, 556325, 15514933, 459533767}},  // 218
    {"nqrkb1rb/pp2pppp/2p1n3/3p4/3PP1N1/8/PPP2PPP/NQRKB1RB w GCgc - 0 9",
     {26, 795, 21752, 679387, 19185851, 616508881}},  // 219
    {"nb1rknbr/pp2ppp1/8/2Bp3p/6P1/2P2P1q/PP1PP2P/NBQRKN1R w HDhd - 0 9",
     {35, 1391, 43025, 1726888, 53033675, 2139267832}},  // 220
    {"nqrbkn1r/pp1pp1pp/8/2p2p2/5P2/P3B2P/1PbPP1P1/NQRBKN1R w HChc - 0 9",
     {23, 758, 19439, 653854, 18296195, 628403401}},  // 221
    {"nqrknbbr/pp1pppp1/7p/2p5/7P/1P1N4/P1PPPPPB/NQRK1B1R w HChc - 2 9",
     {29, 824, 23137, 683686, 19429491, 595493802}},  // 222
    {"1qrknrbb/B1p1pppp/8/1p1p4/2n2P2/1P6/P1PPP1PP/NQRKNR1B w FCfc - 0 9",
     {28, 771, 20237, 581721, 16065378, 483037840}},  // 223
    {"bbnrqk1r/1ppppppp/8/7n/1p6/P6P/1BPPPPP1/1BNRQKNR w HDhd - 0 9",
     {25, 601, 15471, 396661, 10697065, 289472497}},  // 224
    {"bnrbqknr/ppp3p1/3ppp1Q/7p/3P4/1P6/P1P1PPPP/BNRB1KNR w HChc - 0 9",
     {32, 845, 26876, 742888, 23717883, 682154649}},  // 225
    {"bn1qkb1r/pprppppp/8/2p5/2PPP1n1/8/PPR2PPP/BN1QKBNR w Hh - 1 9",
     {32, 856, 27829, 768595, 25245957, 727424329}},  // 226
    {"1nrqknrb/p1pp1ppp/1p2p3/3N4/5P1P/5b2/PPPPP3/B1RQKNRB w GCgc - 2 9",
     {33, 873, 27685, 779473, 25128076, 745401024}},  // 227
    {"nbbrqrk1/pppppppp/8/2N1n3/P7/6P1/1PPPPP1P/1BBRQKNR w HD - 3 9",
     {25, 555, 14339, 342296, 9153089, 234841945}},  // 228
    {"1rbbqknr/1ppp1pp1/1n2p3/p6p/4P1P1/P6N/1PPP1P1P/NRBBQK1R w HBhb - 0 9",
     {25, 693, 18652, 528070, 15133381, 439344945}},  // 229
    {"nrq1kbnr/p1pbpppp/3p4/1p6/6P1/1N3N2/PPPPPP1P/1RBQKB1R w HBhb - 4 9",
     {24, 648, 16640, 471192, 12871967, 380436777}},  // 230
    {"nr1qknr1/p1pppp1p/b5p1/1p6/8/P4PP1/1bPPP1RP/NRBQKN1B w Bgb - 0 9",
     {18, 533, 11215, 331243, 7777833, 234905172}},  // 231
    {"nbrqbknr/1ppp2pp/8/4pp2/p2PP1P1/7N/PPP2P1P/NBRQBK1R w HChc - 0 9",
     {29, 803, 24416, 706648, 22305910, 672322762}},  // 232
    {"nr1b1k1r/ppp1pppp/2bp1n2/6P1/2P3q1/5P2/PP1PP2P/NRQBBKNR w HBhb - 1 9",
     {27, 1199, 30908, 1296241, 35121759, 1418677099}},  // 233
    {"nrqkbbnr/2pppp1p/p7/1p6/2P1Pp2/8/PPNP2PP/1RQKBBNR w HBhb - 0 9",
     {28, 613, 17874, 432750, 13097064, 345294379}},  // 234
    {"1rqkbnrb/pp1ppp1p/1n4p1/B1p5/3PP3/4N3/PPP2PPP/NRQK2RB w GBgb - 0 9",
     {33, 723, 23991, 590970, 19715083, 535650233}},  // 235
    {"nbrqkn1r/1pppp2p/5pp1/p2b4/5P2/P2PN3/1PP1P1PP/NBRQK1BR w HChc - 2 9",
     {23, 607, 15482, 400970, 11026383, 290708878}},  // 236
    {"nrqbknbr/pp1pppp1/8/2p4p/P3PP2/8/1PPP2PP/NRQBKNBR w HBhb - 1 9",
     {26, 700, 19371, 556026, 16058815, 485460242}},  // 237
    {"nrqknbbr/p2pppp1/1pp5/6Qp/3P4/1P3P2/P1P1P1PP/NR1KNBBR w HBhb - 0 9",
     {40, 905, 32932, 829746, 29263502, 791963709}},  // 238
    {"nrqknrbb/1p3ppp/p2p4/2p1p3/1P6/3PP1P1/P1P2P1P/NRQKNRBB w FBfb - 0 9",
     {29, 780, 22643, 654495, 19532077, 593181101}},  // 239
    {"1bnrkqnr/p1pppp2/7p/1p4p1/4b3/7N/PPPP1PPP/BBNRKQ1R w HDhd - 0 9",
     {25, 725, 19808, 565006, 16661676, 487354613}},  // 240
    {"bnrbkq1r/pp2p1pp/5n2/2pp1p2/P7/N1PP4/1P2PPPP/B1RBKQNR w HChc - 1 9",
     {24, 745, 18494, 584015, 15079602, 488924040}},  // 241
    {"2rkqbnr/p1pppppp/2b5/1pn5/1P3P1Q/2B5/P1PPP1PP/1NRK1BNR w HChc - 3 9",
     {33, 904, 30111, 840025, 28194726, 801757709}},  // 242
    {"bnrkqnrb/2pppp2/8/pp4pp/1P5P/6P1/P1PPPPB1/BNRKQNR1 w GCgc - 0 9",
     {34, 1059, 34090, 1054311, 33195397, 1036498304}},  // 243
    {"1bbrkq1r/pppp2pp/1n2pp1n/8/2PP4/1N4P1/PP2PP1P/1BBRKQNR w HDhd - 1 9",
     {33, 891, 28907, 814247, 26970098, 788040469}},  // 244
    {"nrbbkqnr/1p2pp1p/p1p3p1/3p4/8/1PP5/P2PPPPP/NRBBKQNR w HBhb - 0 9",
     {21, 567, 13212, 376487, 9539687, 284426039}},  // 245
    {"1rbkqbr1/ppp1pppp/1n5n/3p4/3P4/1PP3P1/P3PP1P/NRBKQBNR w HBb - 1 9",
     {27, 752, 20686, 606783, 16986290, 521817800}},  // 246
    {"nrbkq1rb/1ppp1pp1/4p1n1/p6p/2PP4/5P2/PPK1P1PP/NRB1QNRB w gb - 0 9",
     {35, 697, 23678, 505836, 16906409, 390324794}},  // 247
    {"nbrkbqnr/p2pp1p1/5p2/1pp4p/7P/3P2P1/PPP1PP2/NBKRBQNR w hc - 0 9",
     {25, 679, 17223, 484921, 12879258, 376652259}},  // 248
    {"nrkb1qnr/ppppp1p1/6bp/5p2/1PP1P1P1/8/P2P1P1P/NRKBBQNR w HBhb - 1 9",
     {32, 761, 24586, 632916, 20671433, 568524724}},  // 249
    {"nrk1bbnr/p1q1pppp/1ppp4/8/3P3P/4K3/PPP1PPP1/NR1QBBNR w hb - 0 9",
     {30, 719, 21683, 541389, 16278120, 423649784}},  // 250
    {"nrkqbr1b/1pppp1pp/5pn1/p6N/1P3P2/8/P1PPP1PP/NRKQB1RB w GBb - 0 9",
     {26, 494, 13815, 296170, 8763742, 206993496}},  // 251
    {"nbrkq2r/pppp1bpp/4p1n1/5p2/7P/2P3N1/PP1PPPP1/NBKRQ1BR w hc - 0 9",
     {27, 701, 19536, 535052, 15394667, 443506342}},  // 252
    {"nrkbqnbr/2ppp2p/pp6/5pp1/P1P5/8/1P1PPPPP/NRKBQNBR w HBhb - 0 9",
     {21, 487, 11341, 285387, 7218486, 193586674}},  // 253
    {"nr1qnbbr/pk1pppp1/1pp4p/8/3P4/5P1P/PPP1P1P1/NRKQNBBR w HB - 0 9",
     {22, 546, 13615, 352855, 9587439, 259830255}},  // 254
    {"nrkq1rbb/pp1ppp1p/2pn4/8/PP3Pp1/7P/2PPP1P1/NRKQNRBB w FBfb - 0 9",
     {26, 839, 22075, 723845, 19867117, 658535326}},  // 255
    {"b2rknqr/pp1ppppp/8/2P5/n7/P7/1PPNPPPb/BBNRK1QR w HDhd - 2 9",
     {24, 699, 19523, 575172, 17734818, 535094237}},  // 256
    {"bnrbknqr/pp2p2p/2p3p1/3p1p2/8/3P4/PPPNPPPP/B1RBKNQR w HChc - 0 9",
     {23, 580, 14320, 385917, 10133092, 288041554}},  // 257
    {"bnrknb1r/pppp2pp/8/4pp2/6P1/3P3P/qPP1PPQ1/BNRKNB1R w HChc - 0 9",
     {28, 1100, 31813, 1217514, 36142423, 1361341249}},  // 258
    {"b1rknqrb/ppp1p1p1/2np1p1p/8/4N3/6PQ/PPPPPP1P/B1RKN1RB w GCgc - 0 9",
     {36, 629, 23082, 453064, 16897544, 367503974}},  // 259
    {"nb1rknqr/pbppp2p/6p1/1p3p2/5P2/3KP3/PPPP2PP/NBBR1NQR w hd - 2 9",
     {18, 557, 9779, 300744, 5822387, 180936551}},  // 260
    {"nr1bknqr/1ppb1ppp/p7/3pp3/B7/2P3NP/PP1PPPP1/NRB1K1QR w HBhb - 2 9",
     {28, 688, 19541, 519785, 15153092, 425149249}},  // 261
    {"nrbkn2r/pppp1pqp/4p1p1/8/3P2P1/P3B3/P1P1PP1P/NR1KNBQR w HBhb - 1 9",
     {32, 808, 25578, 676525, 22094260, 609377239}},  // 262
    {"nrbknqrb/2p1ppp1/1p6/p2p2Bp/1P6/3P1P2/P1P1P1PP/NR1KNQRB w GBgb - 0 9",
     {30, 625, 18288, 418895, 12225742, 301834282}},  // 263
    {"nbr1knqr/1pp1p1pp/3p1pb1/8/7P/5P2/PPPPPQP1/NBRKBN1R w HC - 2 9",
     {29, 863, 25767, 800239, 24965592, 799182442}},  // 264
    {"n1kbbnqr/prp2ppp/1p1p4/4p3/1P2P3/3P1B2/P1P2PPP/NRK1BNQR w HBh - 2 9",
     {26, 653, 17020, 449719, 12187583, 336872952}},  // 265
    {"nrknbbqr/pp3p1p/B3p1p1/2pp4/4P3/2N3P1/PPPP1P1P/NRK1B1QR w HBhb - 0 9",
     {29, 683, 19755, 501807, 14684565, 394951291}},  // 266
    {"n1knbqrb/pr1p1ppp/Qp6/2p1p3/4P3/6P1/PPPP1P1P/NRKNB1RB w GBg - 2 9",
     {31, 552, 17197, 371343, 11663330, 283583340}},  // 267
    {"nbrknqbr/p3p1pp/1p1p1p2/2p5/2Q1PP2/8/PPPP2PP/NBRKN1BR w HChc - 0 9",
     {37, 913, 32470, 825748, 28899548, 759875563}},  // 268
    {"nrkb1qbr/pp1pppp1/5n2/7p/2p5/1N1NPP2/PPPP2PP/1RKB1QBR w HBhb - 0 9",
     {25, 712, 18813, 543870, 15045589, 445074372}},  // 269
    {"nrk2bbr/pppqpppp/3p4/8/1P3nP1/3P4/P1P1PP1P/NRKNQBBR w HBhb - 1 9",
     {24, 814, 19954, 670162, 17603960, 592121050}},  // 270
    {"nrknqrbb/1p2ppp1/2pp4/Q6p/P2P3P/8/1PP1PPP1/NRKN1RBB w FBfb - 0 9",
     {34, 513, 16111, 303908, 9569590, 206509331}},  // 271
    {"bbnrk1rq/pp2p1pp/2ppn3/5p2/8/3NNP1P/PPPPP1P1/BB1RK1RQ w GDgd - 1 9",
     {28, 697, 20141, 517917, 15301879, 410843713}},  // 272
    {"bnrbknrq/ppppp2p/6p1/5p2/4QPP1/8/PPPPP2P/BNRBKNR1 w GCgc - 0 9",
     {37, 901, 32612, 877372, 31385912, 903831981}},  // 273
    {"bnkrnbrq/ppppp1p1/B6p/5p2/8/4P3/PPPP1PPP/BNKRN1RQ w - - 0 9",
     {26, 417, 11124, 217095, 5980981, 133080499}},  // 274
    {"bnrk1rqb/2pppp1p/3n4/pp4p1/3Q1P2/2N3P1/PPPPP2P/B1RKNR1B w FCfc - 0 9",
     {49, 1655, 74590, 2512003, 107234294, 3651608327}},  // 275
    {"nbbrk1rq/pp2pppp/2pp4/8/2P2n2/6N1/PP1PP1PP/NBBRKR1Q w Dgd - 0 9",
     {28, 960, 26841, 884237, 26083252, 846682836}},  // 276
    {"nrbb2rq/pppk1ppp/4p1n1/3p4/6P1/1BP5/PP1PPPQP/NRB1KNR1 w GB - 0 9",
     {28, 735, 22048, 593839, 18588316, 512048946}},  // 277
    {"nrbk1brq/p1ppppp1/7p/1p6/4P1nP/P7/1PPP1PP1/NRBKNBRQ w GBgb - 0 9",
     {22, 572, 12739, 351494, 8525056, 247615348}},  // 278
    {"nrbk1rqb/1pp2ppp/5n2/p2pp3/5B2/1N1P2P1/PPP1PP1P/1R1KNRQB w FBfb - 0 9",
     {35, 927, 31559, 849932, 28465693, 783048748}},  // 279
    {"nbrkb1rq/p1pp1ppp/4n3/4p3/Pp6/6N1/1PPPPPPP/NBRKBRQ1 w Cgc - 0 9",
     {20, 456, 10271, 247733, 6124625, 154766108}},  // 280
    {"nrkb1nrq/p2pp1pp/1pp2p2/7b/6PP/5P2/PPPPP2N/NRKBB1RQ w GBgb - 0 9",
     {21, 479, 11152, 264493, 6696458, 165253524}},  // 281
    {"nr1nbbr1/pppkpp1p/6p1/3p4/P6P/1P6/1RPPPPP1/N1KNBBRQ w G - 1 9",
     {20, 498, 11304, 288813, 7197322, 188021682}},  // 282
    {"nrknbrqb/3p1ppp/ppN1p3/8/6P1/8/PPPPPP1P/1RKNBRQB w FBfb - 0 9",
     {32, 526, 17267, 319836, 10755190, 220058991}},  // 283
    {"nbrkn1bq/p1pppr1p/1p6/5pp1/8/1N2PP2/PPPP2PP/1BKRNRBQ w c - 1 9",
     {19, 491, 10090, 277313, 6230616, 180748649}},  // 284
    {"nrkbnrbq/ppppppp1/8/8/7p/PP3P2/2PPPRPP/NRKBN1BQ w Bfb - 0 9",
     {16, 353, 6189, 156002, 3008668, 82706705}},  // 285
    {"nrknrbbq/p4ppp/2p1p3/1p1p4/1P2P3/2P5/P1NP1PPP/1RKNRBBQ w EBeb - 0 9",
     {29, 728, 21915, 587668, 18231199, 511686397}},  // 286
    {"nrknr1bb/pppp1p2/7p/2qPp1p1/8/1P5P/P1P1PPP1/NRKNRQBB w EBeb - 0 9",
     {20, 714, 14336, 500458, 11132758, 386064577}},  // 287
    {"bbqnrrkn/ppp2p1p/3pp1p1/8/1PP5/2Q5/P1BPPPPP/B2NRKRN w GE - 0 9",
     {39, 593, 23446, 424799, 16764576, 346185058}},  // 288
    {"bqn1rkrn/p1p2ppp/1p1p4/4p3/3PP2b/8/PPP2PPP/BQNBRKRN w GEge - 2 9",
     {25, 773, 20042, 616817, 16632403, 515838333}},  // 289
    {"bqnrkb1n/p1p1pprp/3p4/1p2P1p1/2PP4/8/PP3PPP/BQNRKBRN w GDd - 1 9",
     {31, 860, 28102, 810379, 27233018, 813751250}},  // 290
    {"bqr1krnb/ppppppp1/7p/3n4/1P4P1/P4N2/2PPPP1P/BQNRKR1B w FDf - 3 9",
     {31, 709, 22936, 559830, 18608857, 480498340}},  // 291
    {"qbbn1krn/pp3ppp/4r3/2ppp3/P1P4P/8/1P1PPPP1/QBBNRKRN w GEg - 1 9",
     {26, 775, 21100, 649673, 18476807, 582542257}},  // 292
    {"qnbbrkrn/1p1pp2p/p7/2p2pp1/8/4P2P/PPPP1PPK/QNBBRR1N w ge - 0 9",
     {25, 599, 15139, 389104, 10260500, 279222412}},  // 293
    {"qnbrkbrn/1ppp2p1/p3p2p/5p2/P4P2/1P6/2PPP1PP/QNBRKBRN w GDgd - 0 9",
     {27, 588, 16735, 394829, 11640416, 293541380}},  // 294
    {"1nbrkrnb/p1pppp1p/1pq3p1/8/4P3/P1P4N/1P1P1PPP/QNBRKR1B w FDfd - 1 9",
     {18, 609, 11789, 406831, 8604788, 299491047}},  // 295
    {"qb1r1krn/pppp2pp/1n2ppb1/4P3/7P/8/PPPP1PP1/QBNRBKRN w GDgd - 0 9",
     {20, 578, 12205, 349453, 7939483, 229142178}},  // 296
    {"qnr1bkrn/p3pppp/1bpp4/1p6/2P2PP1/8/PP1PPN1P/QNRBBKR1 w GCgc - 0 9",
     {30, 865, 26617, 771705, 24475596, 719842237}},  // 297
    {"1nkrbbrn/qppppppp/8/8/p2P4/1P5P/P1P1PPP1/QNKRBBRN w - - 0 9",
     {27, 672, 18371, 505278, 14065717, 410130412}},  // 298
    {"1qrkbrnb/ppp1p1pp/n2p4/5p2/4N3/8/PPPPPPPP/Q1RKBRNB w Ffc - 2 9",
     {25, 718, 18573, 536771, 14404324, 424279467}},  // 299
    {"q1nrkrbn/pp1pppp1/2p4p/8/P7/5Pb1/BPPPPNPP/Q1NRKRB1 w FDfd - 0 9",
     {22, 558, 12911, 336042, 8516966, 228074630}},  // 300
    {"qnrbkrbn/1p1p1pp1/p1p5/4p2p/8/3P1P2/PPP1P1PP/QNRBKRBN w FCfc - 0 9",
     {28, 669, 17713, 440930, 12055174, 313276304}},  // 301
    {"qnrkr1bn/p1pp1ppp/8/1p2p3/3P1P2/bP4P1/P1P1P2P/QNRKRBBN w ECec - 1 9",
     {23, 845, 20973, 759778, 19939053, 718075943}},  // 302
    {"q1krrnbb/p1p1pppp/2np4/1pB5/5P2/8/PPPPP1PP/QNRKRN1B w EC - 0 9",
     {29, 776, 21966, 631941, 18110831, 549019739}},  // 303
    {"bbn1rkrn/pp1p1ppp/8/2p1p1q1/6P1/P7/BPPPPP1P/B1NQRKRN w GEge - 0 9",
     {26, 936, 25177, 906801, 24984621, 901444251}},  // 304
    {"bn1brkrn/pp1qpp1p/2p3p1/3p4/1PPP4/P7/4PPPP/BNQBRKRN w GEge - 1 9",
     {29, 755, 22858, 645963, 20128587, 600207069}},  // 305
    {"b2rkbrn/p1pppppp/qp6/8/1n6/2B2P2/P1PPP1PP/1NQRKBRN w GDgd - 0 9",
     {24, 878, 21440, 791007, 20840078, 775795187}},  // 306
    {"b2rkrnb/pqp1pppp/n7/1p1p4/P7/N1P2N2/1P1PPPPP/B1QRKR1B w FDfd - 4 9",
     {26, 724, 19558, 571891, 16109522, 492933398}},  // 307
    {"1bbqrkrn/ppppp1p1/8/5p1p/P1n3P1/3P4/1PP1PP1P/NBBQRRKN w ge - 1 9",
     {25, 678, 17351, 461211, 12173245, 329661421}},  // 308
    {"nqb1rrkn/ppp1bppp/3pp3/8/3P4/1P6/PQP1PPPP/N1BBRRKN w - - 1 9",
     {23, 503, 12465, 290341, 7626054, 188215608}},  // 309
    {"nqbrkbr1/p1pppppp/1p6/2N2n2/2P5/5P2/PP1PP1PP/1QBRKBRN w GDgd - 1 9",
     {29, 688, 20289, 506302, 15167248, 399015237}},  // 310
    {"nqbrkrn1/1ppppp2/6pp/p7/1P6/2Q5/P1PPPPPP/N1BRKRNB w FDfd - 0 9",
     {36, 602, 20985, 397340, 13706856, 291708797}},  // 311
    {"nbqrbrkn/pp1p1pp1/2p5/4p2p/2P3P1/1P3P2/P2PP2P/NBQRBKRN w GD - 0 9",
     {34, 655, 22581, 474396, 16613630, 379344541}},  // 312
    {"nqrbbrkn/1p1pppp1/8/p1p4p/4P2P/1N4P1/PPPP1P2/1QRBBKRN w GC - 0 9",
     {23, 597, 14468, 400357, 10096863, 294900903}},  // 313
    {"nqrkbbrn/2p1p1pp/pp1p1p2/8/P2N4/2P5/1P1PPPPP/1QRKBBRN w GCgc - 0 9",
     {32, 744, 23310, 550728, 17597164, 428786656}},  // 314
    {"n1krbrnb/q1pppppp/p7/1p6/3Q4/2P2P2/PP1PP1PP/N1RKBRNB w FC - 1 9",
     {43, 1038, 41327, 1074450, 40918952, 1126603824}},  // 315
    {"nb1rkrbn/p1pp1p1p/qp6/4p1p1/5PP1/P7/1PPPPB1P/NBQRKR1N w FDfd - 2 9",
     {26, 645, 16463, 445464, 11911314, 342563372}},  // 316
    {"nqr1krbn/pppp1ppp/8/8/3pP3/5P2/PPPb1NPP/NQRBKRB1 w FCfc - 3 9",
     {2, 51, 1047, 27743, 612305, 17040200}},  // 317
    {"n1rkrbbn/pqppppp1/7p/1p6/8/1NPP4/PP1KPPPP/1QR1RBBN w ec - 0 9",
     {25, 674, 17553, 505337, 13421727, 403551903}},  // 318
    {"1qrkrnbb/1p1p1ppp/pnp1p3/8/3PP3/P6P/1PP2PP1/NQRKRNBB w ECec - 0 9",
     {24, 688, 17342, 511444, 13322502, 403441498}},  // 319
    {"1bnrqkrn/2ppppp1/p7/1p1b3p/3PP1P1/8/PPPQ1P1P/BBNR1KRN w GDgd - 1 9",
     {35, 925, 32238, 857060, 30458921, 824344087}},  // 320
    {"bnrbqkr1/ppp2pp1/6n1/3pp2p/1P6/2N3N1/P1PPPPPP/B1RBQRK1 w gc - 0 9",
     {23, 704, 17345, 539587, 14154852, 450893738}},  // 321
    {"1nrqkbrn/p1pppppp/8/1p1b4/P6P/5P2/1PPPP1P1/BNRQKBRN w GCgc - 1 9",
     {19, 505, 10619, 281422, 6450025, 175593967}},  // 322
    {"b1rqkrnb/ppppppp1/8/6p1/3n4/NP6/P1PPPP1P/B1RQKRNB w FCfc - 0 9",
     {25, 614, 15578, 377660, 10391021, 259629603}},  // 323
    {"nbbrqkrn/ppp3p1/3pp3/5p1p/1P2P3/P7/2PPQPPP/NBBR1KRN w GDgd - 0 9",
     {30, 833, 25719, 717713, 22873901, 649556666}},  // 324
    {"nr1bqrk1/ppp1pppp/6n1/3pP3/8/5PQb/PPPP2PP/NRBB1KRN w GB - 3 9",
     {26, 734, 20161, 582591, 17199594, 512134836}},  // 325
    {"1rbqkbr1/ppppp1pp/1n6/4np2/3P1P2/6P1/PPPQP2P/NRB1KBRN w GBgb - 1 9",
     {27, 662, 17897, 447464, 13038519, 338365642}},  // 326
    {"nr1qkr1b/ppp1pp1p/4bn2/3p2p1/4P3/1Q6/PPPP1PPP/NRB1KRNB w FBfb - 4 9",
     {33, 939, 30923, 942138, 30995969, 991509814}},  // 327
    {"nb1qbkrn/pprp1pp1/7p/2p1pB2/Q1PP4/8/PP2PPPP/N1R1BKRN w GCg - 2 9",
     {47, 1128, 50723, 1306753, 56747878, 1560584212}},  // 328
    {"nrqb1rkn/pp2pppp/2bp4/2p5/6P1/2P3N1/PP1PPP1P/NRQBBRK1 w - - 3 9",
     {24, 828, 21148, 723705, 19506135, 668969549}},  // 329
    {"nrq1bbrn/ppkpp2p/2p3p1/P4p2/8/4P1N1/1PPP1PPP/NRQKBBR1 w GB - 0 9",
     {25, 525, 13533, 309994, 8250997, 201795680}},  // 330
    {"Br1kbrn1/pqpppp2/8/6pp/3b2P1/1N6/PPPPPP1P/1RQKBRN1 w FBfb - 3 9",
     {20, 790, 18175, 695905, 17735648, 669854148}},  // 331
    {"nbrqkrbn/2p1p1pp/p7/1p1p1p2/4P1P1/5P2/PPPP3P/NBRQKRBN w FCfc - 0 9",
     {29, 771, 22489, 647106, 19192982, 591335970}},  // 332
    {"1rqbkrbn/1ppppp1p/1n6/p1N3p1/8/2P4P/PP1PPPP1/1RQBKRBN w FBfb - 0 9",
     {29, 502, 14569, 287739, 8652810, 191762235}},  // 333
    {"1rqkrbbn/ppnpp1pp/8/2p5/6p1/3P4/PPP1PPPP/NRK1RBBN w eb - 0 9",
     {19, 531, 10812, 300384, 6506674, 184309316}},  // 334
    {"nrqkrnbb/p1pp2pp/5p2/4P3/2p5/4N3/PP1PP1PP/NRQKR1BB w EBeb - 0 9",
     {26, 800, 23256, 756695, 23952941, 809841274}},  // 335
    {"bbnrkqrn/pp3pp1/4p2p/2pp4/4P1P1/1PB5/P1PP1P1P/1BNRKQRN w GDgd - 0 9",
     {33, 915, 30536, 878648, 29602610, 881898159}},  // 336
    {"bnrbkqr1/1p2pppp/6n1/p1pp4/7P/P3P3/1PPPKPP1/BNRB1QRN w gc - 0 9",
     {19, 457, 9332, 238944, 5356253, 144653627}},  // 337
    {"b1rkqbrn/pp1p2pp/2n1p3/2p2p2/3P2PP/8/PPP1PP2/BNKRQBRN w gc - 0 9",
     {30, 985, 30831, 1011700, 32684185, 1080607773}},  // 338
    {"b1rkqrnb/2ppppp1/np6/p6p/1P6/P2P3P/2P1PPP1/BNRKQRNB w FCfc - 0 9",
     {26, 692, 18732, 517703, 14561181, 413226841}},  // 339
    {"nbbrkqrn/1ppp1p2/p6p/4p1p1/5P2/1P5P/P1PPPNP1/NBBRKQR1 w GDgd - 0 9",
     {22, 561, 13222, 367487, 9307003, 273928315}},  // 340
    {"nrbbkqrn/p1pppppp/8/1p6/4P3/7Q/PPPP1PPP/NRBBK1RN w GBgb - 0 9",
     {38, 769, 28418, 632310, 23091070, 560139600}},  // 341
    {"nrbkqbrn/1pppp2p/8/p4pp1/P4PQ1/8/1PPPP1PP/NRBK1BRN w GBgb - 0 9",
     {23, 507, 13067, 321423, 8887567, 237475184}},  // 342
    {"nr1kqr1b/pp2pppp/5n2/2pp4/P5b1/5P2/1PPPPRPP/NRBK1QNB w Bfb - 2 9",
     {18, 626, 12386, 434138, 9465555, 335004239}},  // 343
    {"nbkrbqrn/1pppppp1/8/4P2p/pP6/P7/2PP1PPP/NBRKBQRN w GC - 0 9",
     {22, 329, 8475, 148351, 4160034, 82875306}},  // 344
    {"nrkb1qrn/pp1pp1pp/8/5p1b/P1p4P/6N1/1PPPPPP1/NRKBBQR1 w GBgb - 2 9",
     {16, 479, 9037, 275354, 5862341, 184959796}},  // 345
    {"1rkq1brn/ppppp1pp/1n6/3b1p2/3N3P/5P2/PPPPP1P1/1RKQBBRN w GBgb - 3 9",
     {23, 614, 15324, 418395, 11090645, 313526088}},  // 346
    {"nrk1brnb/pp1ppppp/2p5/3q4/5P2/PP6/1KPPP1PP/NR1QBRNB w fb - 1 9",
     {25, 942, 21765, 792179, 19318837, 685549171}},  // 347
    {"nbrkqr1n/1pppp2p/p4pp1/2Bb4/5P2/6P1/PPPPP2P/NBRKQ1RN w Cfc - 2 9",
     {30, 841, 24775, 677876, 20145765, 557578726}},  // 348
    {"n1kbqrbn/2p1pppp/1r6/pp1p4/P7/3P4/1PP1PPPP/NRKBQRBN w FBf - 2 9",
     {21, 591, 14101, 394289, 10295086, 292131422}},  // 349
    {"nrkqrbb1/ppp1pppp/3p4/8/4P3/2Pn1P2/PP4PP/NRKQRBBN w EBeb - 0 9",
     {4, 88, 3090, 73414, 2640555, 66958031}},  // 350
    {"nrkqrnbb/ppppp1p1/7p/1P3p2/3P4/2P5/P3PPPP/NRKQRNBB w EBeb - 0 9",
     {29, 689, 21091, 508789, 16226660, 408570219}},  // 351
    {"bbnr1rqn/pp2pkpp/2pp1p2/8/4P1P1/8/PPPP1P1P/BBNRKRQN w FD - 0 9",
     {21, 463, 11135, 256244, 6826249, 165025370}},  // 352
    {"bnrbk1qn/1pppprpp/8/p4p1P/6P1/3P4/PPP1PP2/BNRBKRQN w FCc - 0 9",
     {22, 459, 11447, 268157, 7371098, 190583454}},  // 353
    {"1nrkrbqn/p1pp1ppp/4p3/1p6/1PP5/6PB/P2PPPbP/BNRKR1QN w ECec - 0 9",
     {30, 931, 29012, 887414, 28412902, 869228014}},  // 354
    {"b1rkr1nb/pppppqp1/n4B2/7p/8/1P4P1/P1PPPP1P/1NKRRQNB w ec - 1 9",
     {36, 934, 31790, 930926, 30392925, 952871799}},  // 355
    {"nbbrkrqn/p1ppp1p1/8/1p3p1p/2P3PP/8/PP1PPPQ1/NBBRKR1N w FDfd - 0 9",
     {34, 938, 31848, 921716, 31185844, 944483246}},  // 356
    {"1rbbkrqn/ppp1pp2/1n1p2p1/7p/P3P1P1/3P4/1PP2P1P/NRBBKRQN w FBfb - 0 9",
     {26, 646, 18083, 472744, 14006203, 384101783}},  // 357
    {"nrbkrbq1/Qpppp1pp/2n5/5p2/P4P2/6N1/1PPPP1PP/NRBKRB2 w EBeb - 1 9",
     {27, 619, 16713, 421845, 11718463, 313794027}},  // 358
    {"1rbkr1nb/pppp1qpp/1n6/4pp2/1PP1P3/8/PB1P1PPP/NR1KRQNB w EBeb - 1 9",
     {32, 1029, 32970, 1080977, 35483796, 1181835398}},  // 359
    {"nbrk1rqn/p1ppp2p/1p6/5ppb/8/1N2P2P/PPPP1PP1/1BKRBRQN w fc - 0 9",
     {18, 594, 12350, 408544, 9329122, 315021712}},  // 360
    {"nrkbbrqn/3pppp1/7p/ppp5/P7/1N5P/1PPPPPP1/1RKBBRQN w FBfb - 0 9",
     {19, 417, 9026, 218513, 5236331, 137024458}},  // 361
    {"nrkr1bqn/ppp1pppp/3p4/1b6/7P/P7/1PPPPPP1/NRKRBBQN w DBdb - 1 9",
     {17, 457, 9083, 243872, 5503579, 150091997}},  // 362
    {"nrkrbqnb/p4ppp/1p2p3/2pp4/6P1/2P2N2/PPNPPP1P/1RKRBQ1B w DBdb - 0 9",
     {27, 755, 21012, 620093, 17883987, 547233320}},  // 363
    {"nbkrr1bn/ppB2ppp/4p3/2qp4/4P3/5P2/PPPP2PP/NBRKRQ1N w EC - 1 9",
     {37, 1473, 51939, 1956521, 68070015, 2490912491}},  // 364
    {"n1kbrqbn/p1pp1pp1/4p2p/2B5/1r3P2/8/PPPPP1PP/NRKBRQ1N w EBe - 2 9",
     {30, 1029, 30874, 1053163, 32318550, 1106487743}},  // 365
    {"nrkrqbbn/2pppp1p/8/pp6/1P1P2p1/P5P1/2P1PP1P/NRKRQBBN w DBdb - 0 9",
     {22, 421, 10034, 221927, 5754555, 141245633}},  // 366
    {"nrkr1nbb/1ppp2pp/p3q3/4pp2/2P5/P3P3/1PKP1PPP/NR1RQNBB w db - 0 9",
     {22, 619, 13953, 411392, 9905109, 301403003}},  // 367
    {"bbnrkrnq/1pp1p2p/6p1/p2p1p2/8/1P2P3/P1PP1PPP/BBNRKRNQ w FDfd - 0 9",
     {27, 805, 21915, 688224, 19133881, 620749189}},  // 368
    {"bnrbkrn1/pp1ppp2/2p3pp/8/2Pq4/P4PP1/1P1PP2P/BNRBKRNQ w FCfc - 1 9",
     {20, 770, 16593, 577980, 13581691, 456736500}},  // 369
    {"b1rkrbnq/1pp1pppp/2np4/p5N1/8/1P2P3/P1PP1PPP/BNRKRB1Q w ECec - 0 9",
     {37, 740, 27073, 581744, 21156664, 485803600}},  // 370
    {"b1krrnqb/pp1ppp1p/n1p3p1/2N5/6P1/8/PPPPPP1P/B1RKRNQB w EC - 0 9",
     {34, 850, 28494, 752350, 25360295, 698159474}},  // 371
    {"1bbr1rnq/ppppkppp/8/3np3/4P3/3P4/PPP1KPPP/NBBRR1NQ w - - 1 9",
     {27, 704, 18290, 480474, 12817011, 341026662}},  // 372
    {"nrbbk1nq/p1p1prpp/1p6/N2p1p2/P7/8/1PPPPPPP/R1BBKRNQ w Fb - 2 9",
     {23, 552, 13710, 348593, 9236564, 248469879}},  // 373
    {"1rbkrb1q/1pppp1pp/1n5n/p4p2/P3P3/1P6/2PPNPPP/NRBKRB1Q w EBeb - 1 9",
     {22, 415, 10198, 217224, 5735644, 135295774}},  // 374
    {"nrbkr1qb/1pp1pppp/6n1/p2p4/2P1P3/1N4N1/PP1P1PPP/1RBKR1QB w EBeb - 0 9",
     {27, 709, 19126, 506214, 14192779, 380516508}},  // 375
    {"nbrkbrnq/p3p1pp/1pp2p2/3p4/1PP5/4P3/P1KP1PPP/NBR1BRNQ w fc - 0 9",
     {24, 715, 18009, 535054, 14322279, 427269976}},  // 376
    {"nrk1brnq/pp1p1pp1/7p/b1p1p3/1P6/6P1/P1PPPPQP/NRKBBRN1 w FBfb - 2 9",
     {29, 675, 20352, 492124, 15316285, 389051744}},  // 377
    {"nrkr1bnq/1p2pppp/p2p4/1bp5/PP6/1R5N/2PPPPPP/N1KRBB1Q w Ddb - 2 9",
     {27, 744, 20494, 571209, 16188945, 458900901}},  // 378
    {"nrk1b1qb/pppn1ppp/3rp3/3p4/2P3P1/3P4/PPN1PP1P/1RKRBNQB w DBb - 3 9",
     {35, 941, 33203, 935791, 33150360, 968024386}},  // 379
    {"nb1rrnbq/ppkp1ppp/8/2p1p3/P7/1N2P3/1PPP1PPP/1BKRRNBQ w - - 1 9",
     {19, 451, 9655, 235472, 5506897, 139436165}},  // 380
    {"nrkbrnbq/4pppp/1ppp4/p7/2P1P3/3P2N1/PP3PPP/NRKBR1BQ w EBeb - 0 9",
     {29, 591, 17132, 384358, 11245508, 270967202}},  // 381
    {"nrkrnbbq/3p1ppp/1p6/p1p1p3/3P2P1/P4Q2/1PP1PP1P/NRKRNBB1 w DBdb - 0 9",
     {38, 792, 28597, 640961, 22654797, 540864616}},  // 382
    {"nr1rnqbb/ppp1pp1p/3k2p1/3p4/1P5P/3P1N2/P1P1PPP1/NRKR1QBB w DB - 1 9",
     {25, 758, 18547, 543643, 13890077, 402109399}},  // 383
    {"bbqrnnkr/1ppp1p1p/5p2/p5p1/P7/1P4P1/2PPPP1P/1BQRNNKR w HDhd - 0 9",
     {20, 322, 7224, 145818, 3588435, 82754650}},  // 384
    {"bqrb2k1/pppppppr/5nnp/8/3P1P2/4P1N1/PPP3PP/BQRBN1KR w HCc - 1 9",
     {25, 597, 15872, 397970, 11162476, 295682250}},  // 385
    {"bqrnn1kr/1pppbppp/8/4p3/1p6/2P1N2P/P2PPPP1/BQR1NBKR w HChc - 1 9",
     {34, 921, 31695, 864023, 30126510, 850296236}},  // 386
    {"bqr1nkr1/pppppp2/2n3p1/7p/1P1b1P2/8/PQP1P1PP/B1RNNKRB w GCgc - 0 9",
     {23, 788, 21539, 686795, 20849374, 645694580}},  // 387
    {"qbbrnn1r/1pppp1pk/p7/5p1p/P2P3P/3N4/1PP1PPP1/QBBR1NKR w HD - 0 9",
     {34, 713, 24475, 562189, 19494094, 482645160}},  // 388
    {"qrbb2kr/p1pppppp/1p1n4/8/1P3n2/P7/Q1PPP1PP/1RBBNNKR w HBhb - 0 9",
     {28, 977, 26955, 949925, 27802999, 992109168}},  // 389
    {"qrb2bkr/1pp1pppp/2np1n2/pN6/3P4/4B3/PPP1PPPP/QR2NBKR w HBhb - 0 9",
     {27, 730, 20534, 585091, 17005916, 507008968}},  // 390
    {"qrbnnkrb/pp2pp1p/8/2pp2p1/7P/P1P5/QP1PPPP1/1RBNNKRB w GBgb - 0 9",
     {24, 813, 21142, 707925, 19615756, 655850285}},  // 391
    {"1brnb1kr/p1pppppp/1p6/8/4q2n/1P2P1P1/PNPP1P1P/QBR1BNKR w HChc - 3 9",
     {17, 734, 13462, 530809, 11032633, 416356876}},  // 392
    {"1rnbbnkr/1pp1pppp/1q1p4/p7/4P3/5PN1/PPPP1BPP/QRNB2KR w HBhb - 1 9",
     {26, 809, 21764, 706677, 20292750, 675408811}},  // 393
    {"qrnnbb1Q/ppp1pk1p/3p2p1/5p2/PP6/5P2/2PPP1PP/1RNNBBKR w HB - 0 9",
     {37, 751, 27902, 603931, 22443036, 515122176}},  // 394
    {"qrnnbkrb/p3p1pp/3p1p2/1pp5/PP2P3/8/2PP1PPP/QRNNBRKB w gb - 0 9",
     {30, 906, 27955, 872526, 27658191, 890966633}},  // 395
    {"qbrnnkbr/1p2pp1p/p1p3p1/3p4/6P1/P1N4P/1PPPPP2/QBR1NKBR w HChc - 0 9",
     {26, 701, 18930, 521377, 14733245, 416881799}},  // 396
    {"qr1b1kbr/1p1ppppp/1n1n4/p1p5/4P3/5NPP/PPPP1P2/QRNB1KBR w HBhb - 1 9",
     {26, 649, 17235, 451997, 12367604, 342165821}},  // 397
    {"qrnnkb1r/1pppppp1/7p/p4b2/4P3/5P1P/PPPP2PR/QRNNKBB1 w Bhb - 1 9",
     {34, 941, 31720, 901240, 30307554, 888709821}},  // 398
    {"qr1nkrbb/p2ppppp/1pp5/8/3Pn3/1NP3P1/PP2PP1P/QR1NKRBB w FBfb - 1 9",
     {19, 505, 11107, 294251, 7046501, 190414579}},  // 399
    {"bbrqn1kr/1pppp1pp/4n3/5p2/p5P1/3P4/PPP1PPKP/BBRQNN1R w hc - 0 9",
     {24, 573, 12963, 335845, 8191054, 227555387}},  // 400
    {"brqb1nkr/pppppp1p/8/4N1pn/5P2/6P1/PPPPP2P/BRQB1NKR w HBhb - 0 9",
     {26, 550, 14338, 331666, 8903754, 223437427}},  // 401
    {"brqnn1kr/pp3ppp/2pbp3/3p4/8/2NPP3/PPP1BPPP/BRQ1N1KR w HBhb - 0 9",
     {27, 780, 20760, 589328, 16243731, 463883447}},  // 402
    {"brq1nkrb/ppp2ppp/8/n2pp2P/P7/4P3/1PPP1PP1/BRQNNKRB w GBgb - 1 9",
     {17, 426, 8295, 235162, 5048497, 153986034}},  // 403
    {"rbbqn1kr/pp2p1pp/6n1/2pp1p2/2P4P/P7/BP1PPPP1/R1BQNNKR w HAha - 0 9",
     {27, 916, 25798, 890435, 26302461, 924181432}},  // 404
    {"1qbbn1kr/1ppppppp/r3n3/8/p1P5/P7/1P1PPPPP/RQBBNNKR w HAh - 1 9",
     {29, 817, 24530, 720277, 22147642, 670707652}},  // 405
    {"rqbnnbkr/ppp1ppp1/7p/3p4/PP6/7P/1NPPPPP1/RQB1NBKR w HAa - 1 9",
     {23, 572, 14509, 381474, 10416981, 288064942}},  // 406
    {"r1bnnkrb/q1ppp1pp/p7/1p3pB1/2P1P3/3P4/PP3PPP/RQ1NNKRB w GAga - 2 9",
     {31, 925, 27776, 860969, 26316355, 843078864}},  // 407
    {"rbqnb1kr/ppppp1pp/5p2/5N2/7P/1n3P2/PPPPP1P1/RBQNB1KR w HAha - 1 9",
     {32, 864, 27633, 766551, 24738875, 707188107}},  // 408
    {"rqnbbn1r/ppppppp1/6k1/8/6Pp/2PN4/PP1PPPKP/RQ1BBN1R w - - 0 9",
     {27, 566, 15367, 347059, 9714509, 234622128}},  // 409
    {"rqnnbbkr/p1p2pp1/1p1p3p/4p3/4NP2/6P1/PPPPP2P/RQN1BBKR w HAha - 0 9",
     {27, 631, 17923, 452734, 13307890, 356279813}},  // 410
    {"1qnnbrkb/rppp1ppp/p3p3/8/4P3/2PP1P2/PP4PP/RQNNBKRB w GA - 1 9",
     {24, 479, 12135, 271469, 7204345, 175460841}},  // 411
    {"rbqnn1br/p1pppk1p/1p4p1/5p2/8/P1P2P2/1PBPP1PP/R1QNNKBR w HA - 0 9",
     {31, 756, 23877, 625194, 20036784, 554292502}},  // 412
    {"rqnbnkbr/1ppppp2/p5p1/8/1P4p1/4PP2/P1PP3P/RQNBNKBR w HAha - 0 9",
     {24, 715, 18536, 575589, 16013189, 515078271}},  // 413
    {"rq1nkbbr/1p2pppp/p2n4/2pp4/1P4P1/P2N4/2PPPP1P/RQ1NKBBR w HAha - 1 9",
     {27, 694, 19840, 552904, 16685687, 494574415}},  // 414
    {"r1nnkrbb/pp1pppp1/2p3q1/7p/8/1PPP3P/P3PPP1/RQNNKRBB w FAfa - 1 9",
     {18, 520, 10808, 329085, 7508201, 235103697}},  // 415
    {"bbrnqk1r/pppp3p/6p1/4pp2/3P2P1/8/PPP1PP1P/BBRN1NKR w HC - 0 9",
     {22, 566, 12965, 362624, 8721079, 259069471}},  // 416
    {"brnb1nkr/pppqpp2/3p2pp/8/3PP3/1P6/PBP2PPP/1RNBQNKR w HBhb - 0 9",
     {32, 859, 28517, 817464, 27734108, 829785474}},  // 417
    {"brnq1b1r/ppp1ppkp/3p1np1/8/8/5P1P/PPPPPKPR/BRNQNB2 w - - 0 9",
     {21, 511, 10951, 273756, 6372681, 167139732}},  // 418
    {"brnq1rkb/1pppppp1/3n3p/p7/8/P4NP1/1PPPPPRP/BRNQ1K1B w B - 0 9",
     {25, 548, 14049, 341208, 9015901, 235249649}},  // 419
    {"rbb1qnkr/p1ppp1pp/1p3p2/6n1/8/1PN1P2P/P1PP1PP1/RBB1QNKR w HAha - 0 9",
     {25, 673, 16412, 467660, 12099119, 361714466}},  // 420
    {"rnbb1nkr/1ppp1ppp/4p3/p5q1/6P1/1PP5/PB1PPP1P/RN1BQNKR w HAha - 1 9",
     {19, 663, 14149, 489653, 11491355, 399135495}},  // 421
    {"rnbqnbkr/1pp1p2p/3p1p2/p5p1/5PP1/2P5/PPNPP2P/RNBQ1BKR w HAha - 0 9",
     {24, 647, 16679, 461931, 12649636, 361157611}},  // 422
    {"rnb2krb/pppqppnp/8/3p2p1/1P4P1/7P/P1PPPPB1/RNBQNKR1 w GAga - 1 9",
     {24, 722, 18749, 605229, 16609220, 563558512}},  // 423
    {"rbnqb1kr/pppn1pp1/3p3p/4p3/1P6/P7/R1PPPPPP/1BNQBNKR w Hha - 1 9",
     {20, 538, 12277, 345704, 8687621, 255304141}},  // 424
    {"rnqb1nkr/p1pbp1pp/8/1pPp1p2/P2P4/8/1P2PPPP/RNQBBNKR w HAha - 1 9",
     {35, 764, 26952, 632796, 22592380, 564255328}},  // 425
    {"rnq1bbkr/1p1ppp1p/4n3/p1p3p1/P1PP4/8/RP2PPPP/1NQNBBKR w Hha - 0 9",
     {29, 709, 21296, 570580, 17597398, 506140370}},  // 426
    {"1nqnbkrb/1pppp2p/r7/p4pp1/3P4/8/PPPBPPPP/RNQNK1RB w g - 0 9",
     {27, 1028, 28534, 1050834, 30251988, 1096869832}},  // 427
    {"rbnqnkbr/p1pp1p1p/8/1p2p3/3P2pP/2P5/PP2PPP1/RBNQNKBR w HAha - 0 9",
     {32, 832, 27120, 750336, 24945574, 724171581}},  // 428
    {"rnq1nkbr/1p1p1ppp/2p1pb2/p7/7P/2P5/PPNPPPPB/RNQB1K1R w HAha - 2 9",
     {31, 779, 24010, 638640, 19919434, 551494771}},  // 429
    {"rnqnk1br/p1ppp1bp/1p3p2/6p1/4N3/P5P1/1PPPPP1P/R1QNKBBR w HAha - 2 9",
     {25, 717, 19396, 576577, 16525239, 507175842}},  // 430
    {"rnq1krbb/p1p1pppp/8/1p1p4/1n5B/2N2P2/PPPPP1PP/RNQ1KR1B w FAfa - 0 9",
     {28, 867, 24029, 735686, 21112751, 654808184}},  // 431
    {"bbrnnqkr/1pp1pppp/3p4/p7/P3P3/7P/1PPP1PP1/BBRNNQKR w HChc - 0 9",
     {24, 405, 11025, 210557, 6196438, 131401224}},  // 432
    {"brnbnqkr/p1ppp3/1p5p/5Pp1/5P2/3N4/PPPPP2P/BRNB1QKR w HBhb g6 0 9",
     {25, 785, 21402, 698331, 20687969, 695850727}},  // 433
    {"br1nqbkr/1ppppp2/pn6/6pp/2PP4/1N4P1/PP2PP1P/BR1NQBKR w HBhb - 0 9",
     {25, 596, 16220, 421882, 12185361, 337805606}},  // 434
    {"1rnnqkrb/p2ppp1p/1pp5/2N3p1/8/1P6/P1PPPPKP/BR1NQ1RB w gb - 0 9",
     {38, 960, 34831, 913665, 32490040, 880403591}},  // 435
    {"rbbnnqkr/pp3pp1/2p1p3/3p3p/3P3P/1PP5/P3PPP1/RBBNNQKR w HAha - 0 9",
     {30, 785, 23079, 656618, 19885037, 599219582}},  // 436
    {"rn1bnqkr/p1ppppp1/8/1p5p/P4P1P/3N4/1PPPP1b1/RNBB1QKR w HAha - 0 9",
     {27, 752, 21735, 613194, 18862234, 547415271}},  // 437
    {"1nbnqbkr/1p1p1ppp/r3p3/p1p5/P3P3/3Q4/1PPP1PPP/RNBN1BKR w HAh - 2 9",
     {33, 721, 24278, 572535, 19648535, 496023732}},  // 438
    {"rnbnqkrb/2pppppp/1p6/p7/1PP5/4N2P/P2PPPP1/RNB1QKRB w GAg - 0 9",
     {23, 570, 14225, 374196, 10022614, 279545007}},  // 439
    {"rbnnbq1r/ppppppkp/6p1/N7/4P3/P7/1PPP1PPP/RB1NBQKR w HA - 5 9",
     {27, 620, 18371, 440594, 13909432, 349478320}},  // 440
    {"r1nbbqkr/pppppp1p/8/8/1n3Pp1/3N1QP1/PPPPP2P/RN1BB1KR w HAha - 0 9",
     {31, 791, 25431, 682579, 22408813, 636779732}},  // 441
    {"rnq1bbkr/pp1p1ppp/2pnp3/8/7P/1QP5/PP1PPPPR/RNN1BBK1 w Aha - 2 9",
     {28, 559, 16838, 390887, 12242780, 315431511}},  // 442
    {"rnnqbrkb/2ppppp1/1p1N4/p6p/4P3/8/PPPP1PPP/R1NQBKRB w GA - 0 9",
     {32, 638, 20591, 438792, 14395828, 331782223}},  // 443
    {"rbnnq1br/pppp1kp1/4pp2/7p/PP6/2PP4/4PPPP/RBNNQKBR w HA - 0 9",
     {21, 521, 12201, 320429, 8239159, 227346638}},  // 444
    {"rnnbqkbr/p2ppp2/7p/1pp3p1/2P2N2/8/PP1PPPPP/RN1BQKBR w HAha - 0 9",
     {25, 528, 13896, 326094, 9079829, 232750602}},  // 445
    {"rnn1kbbr/ppppqp2/6p1/2N1p2p/P7/2P5/1P1PPPPP/RN1QKBBR w HAha - 2 9",
     {27, 801, 22088, 707078, 20334071, 682580976}},  // 446
    {"rnnqkrbb/p1p1p1pp/1p3p2/8/3p2Q1/P1P1P3/1P1P1PPP/RNN1KRBB w FAfa - 0 9",
     {37, 1014, 34735, 998999, 32921537, 988770109}},  // 447
    {"bbrnk1qr/1pppppp1/p4n1p/8/P2P2N1/8/1PP1PPPP/BBR1NKQR w HC - 1 9",
     {21, 481, 11213, 279993, 7015419, 187564853}},  // 448
    {"brnbnkqr/1pp1p1p1/p2p1p2/7p/1P4PP/8/PBPPPP2/1RNBNKQR w HBhb - 0 9",
     {31, 743, 24260, 660177, 22391185, 653721389}},  // 449
    {"br2kbqr/ppppp1pp/3n1p2/3P4/3n3P/3N4/PPP1PPP1/BR1NKBQR w HBhb - 3 9",
     {25, 872, 22039, 748726, 20281962, 685749952}},  // 450
    {"br1nkqrb/ppppppp1/8/7p/4P3/n1P2PP1/PP1P3P/BRNNKQRB w GBgb - 0 9",
     {28, 607, 16934, 396483, 11607818, 294181806}},  // 451
    {"rbbn1kqr/pp1pp1p1/2pn3p/5p2/5P2/1P1N4/PNPPP1PP/RBB2KQR w HAha - 1 9",
     {27, 725, 21543, 616082, 19239812, 581716972}},  // 452
    {"rnbbnk1r/pp1ppp1p/6q1/2p5/PP4p1/4P3/2PP1PPP/RNBBNKQR w HAha - 1 9",
     {25, 1072, 26898, 1088978, 28469879, 1122703887}},  // 453
    {"rnbnkbqr/1pp3pp/3p4/p3pp2/3P2P1/2N1N3/PPP1PP1P/R1B1KBQR w HAha - 0 9",
     {31, 1028, 32907, 1095472, 36025223, 1211187800}},  // 454
    {"r1bnkqrb/1ppppppp/p3n3/8/6P1/4N3/PPPPPPRP/RNB1KQ1B w Aga - 1 9",
     {23, 457, 11416, 250551, 6666787, 159759052}},  // 455
    {"rbn1bkqr/p1pp1pp1/1pn5/4p2p/7P/1PBP4/P1P1PPP1/RBNN1KQR w HAha - 0 9",
     {23, 470, 11649, 264274, 6963287, 172833738}},  // 456
    {"rnnbbkqr/3ppppp/p7/1pp5/P6P/6P1/1PPPPP2/RNNBBKQR w HAha - 0 9",
     {26, 569, 15733, 375556, 11008114, 284485303}},  // 457
    {"r1nk1bqr/1pppp1pp/2n5/p4p1b/5P2/1N4B1/PPPPP1PP/RN1K1BQR w HAha - 2 9",
     {25, 824, 21983, 738366, 20904119, 716170771}},  // 458
    {"r1nkbqrb/p2pppp1/npp4p/8/4PP2/2N4P/PPPP2P1/R1NKBQRB w GAga - 0 9",
     {31, 548, 17480, 349633, 11469548, 255067638}},  // 459
    {"rbnnkqbr/ppppp2p/5p2/6p1/2P1B3/P6P/1P1PPPP1/R1NNKQBR w HAha - 1 9",
     {31, 809, 24956, 680747, 21247414, 606221516}},  // 460
    {"1r1bkqbr/pppp1ppp/2nnp3/8/2P5/N4P2/PP1PP1PP/1RNBKQBR w Hh - 0 9",
     {28, 810, 22844, 694599, 20188622, 636748147}},  // 461
    {"rn1kqbbr/p1pppp1p/1p4p1/1n6/1P2P3/4Q2P/P1PP1PP1/RNNK1BBR w HAha - 1 9",
     {39, 848, 30100, 724426, 25594662, 659615710}},  // 462
    {"rn1kqrbb/pppppppp/8/8/2nP2P1/1P2P3/P1P2P1P/RNNKQRBB w FAfa - 1 9",
     {29, 766, 21701, 567971, 16944425, 456898648}},  // 463
    {"b1rnnkrq/bpppppp1/7p/8/1p6/2B5/PNPPPPPP/1BR1NKRQ w GCgc - 2 9",
     {25, 667, 17253, 472678, 12865247, 365621294}},  // 464
    {"brnb1krq/pppppppp/8/5P2/2P1n2P/8/PP1PP1P1/BRNBNKRQ w GBgb - 1 9",
     {23, 620, 14882, 402561, 10776855, 300125003}},  // 465
    {"b1nnkbrq/pr1pppp1/1p5p/2p5/P2N1P2/8/1PPPP1PP/BR1NKBRQ w GBg - 0 9",
     {24, 472, 12181, 267398, 7370758, 178605165}},  // 466
    {"br1nkrqb/p1p1p1pp/3n4/1p1p1p2/5N1P/4P3/PPPP1PP1/BR1NKRQB w FBfb - 0 9",
     {24, 775, 19398, 624309, 16429837, 539767605}},  // 467
    {"rbbnnkrq/p2pp1pp/2p5/5p2/1pPP1B2/P7/1P2PPPP/RB1NNKRQ w GAga - 0 9",
     {34, 921, 30474, 849933, 28095833, 806446436}},  // 468
    {"rnbbnkr1/1p1ppp1p/2p3p1/p7/2Pq4/1P1P4/P2BPPPP/RN1BNKRQ w GAga - 2 9",
     {26, 1139, 29847, 1204863, 32825932, 1281760240}},  // 469
    {"1rbnkbrq/pppppp2/n5pp/2P5/P7/4N3/1P1PPPPP/RNB1KBRQ w GAg - 2 9",
     {23, 574, 14146, 391413, 10203438, 301874034}},  // 470
    {"1nbnkr1b/rppppppq/p7/7p/1P5P/3P2P1/P1P1PP2/RNBNKRQB w FAf - 1 9",
     {33, 823, 26696, 724828, 23266182, 672294132}},  // 471
    {"rbn1bkrq/ppppp3/4n2p/5pp1/1PN5/2P5/P2PPPPP/RBN1BKRQ w GAga - 0 9",
     {27, 859, 24090, 796482, 23075785, 789152120}},  // 472
    {"r1nbbkrq/1ppp2pp/2n2p2/p3p3/5P2/1N4BP/PPPPP1P1/RN1B1KRQ w GAga - 0 9",
     {25, 774, 20141, 618805, 16718577, 515864053}},  // 473
    {"rnnkbbrq/1pppp1p1/5p2/7p/p6P/3N1P2/PPPPP1PQ/RN1KBBR1 w GAga - 0 9",
     {29, 673, 20098, 504715, 15545590, 416359581}},  // 474
    {"r1nkbrqb/pppp1p2/n3p1p1/7p/2P2P2/1P6/P2PPQPP/RNNKBR1B w FAfa - 0 9",
     {27, 722, 21397, 593762, 18742426, 537750982}},  // 475
    {"rbnnkr1q/1ppp2pp/p4p2/P2bp3/4P2P/8/1PPP1PP1/RBNNKRBQ w FAfa - 1 9",
     {26, 848, 23387, 741674, 21591790, 675163653}},  // 476
    {"rn1bkrb1/1ppppp1p/pn4p1/8/P2q3P/3P4/NPP1PPP1/RN1BKRBQ w FAfa - 1 9",
     {22, 803, 18322, 632920, 15847763, 536419559}},  // 477
    {"rn1krbbq/pppp1npp/4pp2/8/4P2P/3P2P1/PPP2P2/RNNKRBBQ w EAea - 1 9",
     {29, 810, 23968, 670500, 20361517, 575069358}},  // 478
    {"rnn1rqbb/ppkp1pp1/2p1p2p/2P5/8/3P1P2/PP2P1PP/RNNKRQBB w EA - 0 9",
     {22, 506, 11973, 292344, 7287368, 189865944}},  // 479
    {"bbqr1knr/pppppp1p/8/4n1p1/2P1P3/6P1/PPQP1P1P/BB1RNKNR w HDhd - 0 9",
     {26, 650, 18253, 481200, 14301029, 394943978}},  // 480
    {"bq1bnknr/pprppp1p/8/2p3p1/4PPP1/8/PPPP3P/BQRBNKNR w HCh - 0 9",
     {24, 548, 14021, 347611, 9374021, 250988458}},  // 481
    {"bqrnkb1r/1p2pppp/p1pp3n/5Q2/2P4P/5N2/PP1PPPP1/B1RNKB1R w HChc - 0 9",
     {46, 823, 33347, 673905, 26130444, 582880996}},  // 482
    {"bq1rknrb/pppppp1p/4n3/6p1/4P1P1/3P1P2/PPP4P/BQRNKNRB w GCg - 0 9",
     {23, 618, 14815, 419474, 10606831, 315124518}},  // 483
    {"q1brnknr/pp1pp1p1/8/2p2p1p/5b2/P4N2/1PPPP1PP/QBBRK1NR w hd - 0 9",
     {22, 675, 15778, 473994, 12077228, 368479752}},  // 484
    {"qrbbnknr/1p1ppp1p/p1p5/8/1P2P1p1/3P1B2/P1P2PPP/QRB1NKNR w HBhb - 0 9",
     {32, 722, 24049, 569905, 19584539, 484814878}},  // 485
    {"qrb1kbnr/p3pppp/2n5/1ppp4/7P/3P1P2/PPP1P1PR/QRBNKBN1 w Bhb - 0 9",
     {26, 831, 22606, 724505, 20500804, 662608969}},  // 486
    {"qrbnknrb/ppp1pp2/6p1/7p/PPNp4/8/2PPPPPP/QRB1KNRB w GBgb - 0 9",
     {31, 840, 26762, 742772, 24422614, 701363800}},  // 487
    {"qbrnbknr/pp1pp1pp/8/2p2p2/3Q4/PP6/2PPPPPP/1BRNBKNR w HChc - 0 9",
     {38, 1121, 39472, 1198438, 41108769, 1285503872}},  // 488
    {"qr1bbk1r/pppppp1p/1n6/5np1/4B3/1PP5/P2PPPPP/QRN1BKNR w HBhb - 0 9",
     {25, 694, 16938, 472950, 12164609, 345122090}},  // 489
    {"qrnkbbnr/1p1pp2p/p7/2p1Npp1/6P1/7P/PPPPPP2/QR1KBBNR w HBhb - 0 9",
     {27, 586, 16348, 393391, 11409633, 298054792}},  // 490
    {"qrnkbnrb/pp1p1p2/2p1p1pp/4N3/P4P2/8/1PPPP1PP/QR1KBNRB w GBgb - 0 9",
     {32, 645, 20737, 460319, 15037464, 358531599}},  // 491
    {"qbrnknbr/1pppppp1/p6p/8/1P6/3PP3/PQP2PPP/1BRNKNBR w HChc - 3 9",
     {26, 595, 16755, 415022, 12214768, 323518628}},  // 492
    {"qrnbk1br/1ppppp1p/p5p1/8/4Pn2/4K1P1/PPPP1P1P/QRNB1NBR w hb - 0 9",
     {24, 609, 13776, 359415, 8538539, 230364479}},  // 493
    {"qrnk1bbr/1pnp1ppp/p1p1p3/8/3Q4/1P1N3P/P1PPPPP1/1RNK1BBR w HBhb - 0 9",
     {43, 1106, 42898, 1123080, 41695761, 1113836402}},  // 494
    {"qrnknrb1/pppppp2/8/6pp/4P2P/3P1P2/PbP3P1/QRNKNRBB w FBfb - 0 9",
     {24, 658, 17965, 488373, 14457245, 400971226}},  // 495
    {"bbrqnrk1/ppp2ppp/7n/3pp3/8/P4N1N/1PPPPPPP/BBRQ1RK1 w - - 1 9",
     {22, 503, 12078, 310760, 8080951, 224960353}},  // 496
    {"brqbnk1r/1ppp1ppp/8/p3pn2/8/2PP1P2/PP2PKPP/BRQBN1NR w hb - 1 9",
     {25, 745, 19387, 570459, 15520298, 460840861}},  // 497
    {"brqnkbnr/pp2pp1p/3p4/2p5/5p2/3P3P/PPP1PPP1/B1RNKBNR w Hhb - 0 9",
     {19, 516, 10755, 312996, 6995034, 214340699}},  // 498
    {"brq1kn1b/1ppppprp/2n3p1/p7/P1N5/6P1/1PPPPP1P/BRQNK1RB w GBb - 2 9",
     {29, 557, 16739, 352277, 10840256, 249999654}},  // 499
    {"rbbq1k1r/ppp1pppp/7n/1n1p4/5P2/P2P4/1PPBP1PP/RB1QNKNR w HAha - 1 9",
     {25, 769, 20110, 638340, 17438715, 570893953}},  // 500
    {"r1bbnk1r/qpp1pppp/p6n/3p4/1P6/5N1P/P1PPPPP1/RQBBK1NR w ha - 0 9",
     {23, 728, 18209, 587364, 16053564, 529082811}},  // 501
    {"rqbnkbnr/1pp2p1p/3p4/p3p1p1/8/2P2P2/PP1PPNPP/RQBNKB1R w HAha - 0 9",
     {26, 772, 21903, 653704, 19571559, 593915677}},  // 502
    {"r1bnknrb/pqppp1p1/1p5p/5p2/7P/3P2N1/PPP1PPP1/RQBNK1RB w GAga - 2 9",
     {27, 748, 20291, 597105, 16324542, 506453626}},  // 503
    {"rbqnbknr/pp1pppp1/8/2p5/3P3p/5N1P/PPP1PPPR/RBQNBK2 w Aha - 0 9",
     {30, 859, 26785, 819631, 26363334, 842796987}},  // 504
    {"rqnbbrk1/ppppppp1/8/5n1p/3P3P/2B3P1/PPP1PP2/RQNB1KNR w HA - 0 9",
     {22, 505, 11452, 283464, 7055215, 186760784}},  // 505
    {"rqnkbbnr/pp2p1p1/8/2pp1p1p/3PPP2/8/PPP1N1PP/RQNKBB1R w HAha - 0 9",
     {28, 832, 23142, 722857, 20429246, 663183060}},  // 506
    {"rqnkbnr1/pppp2bp/6p1/4pp2/1P2P3/3NN3/P1PP1PPP/RQ1KB1RB w GAga - 0 9",
     {28, 641, 18835, 459993, 14038570, 364210162}},  // 507
    {"rbq2kbr/pppppppp/2n5/P7/3P1n2/2P5/1P2PPPP/RBQNKNBR w HA - 1 9",
     {31, 889, 27028, 766181, 24299415, 692180754}},  // 508
    {"rq1bkn1r/ppppp2p/3n4/5pp1/2b3P1/1N1P1P2/PPP1P2P/RQ1BKNBR w HAha - 1 9",
     {28, 810, 22667, 657520, 18719949, 556282676}},  // 509
    {"r1nknbbr/p2ppp1p/1pp3p1/8/1P6/4P3/P1PPNPPq/R1QKNBBR w HAha - 0 9",
     {24, 797, 22144, 719069, 21862776, 716521139}},  // 510
    {"rqnknrbb/ppp1p3/5ppp/2Np4/2P5/4P3/PP1P1PPP/RQNK1RBB w FAfa - 0 9",
     {34, 686, 23277, 515541, 17664543, 423574794}},  // 511
    {"1brnqknr/2p1pppp/p2p4/1P6/6P1/4Nb2/PP1PPP1P/BBR1QKNR w HChc - 1 9",
     {34, 1019, 32982, 1003103, 33322477, 1043293394}},  // 512
    {"brn1qknr/1p1pppp1/pb5p/Q1p5/3P3P/8/PPP1PPPR/BRNB1KN1 w Bhb - 2 9",
     {32, 642, 20952, 464895, 15454749, 371861782}},  // 513
    {"brnqkbnr/pppppp2/8/6pp/6P1/P2P1P2/1PP1P2P/BRNQKBNR w HBhb - 0 9",
     {20, 441, 9782, 240220, 5770284, 153051835}},  // 514
    {"2nqknrb/1rpppppp/5B2/pp6/1PP1b3/3P4/P3PPPP/1RNQKNRB w GBg - 1 9",
     {35, 1042, 36238, 1101159, 38505058, 1202668717}},  // 515
    {"rb1nqknr/1pp1pppp/8/3p4/p2P4/6PN/PPPQPP1P/RBBN1K1R w HAha - 0 9",
     {29, 692, 21237, 555018, 17820605, 497251206}},  // 516
    {"rnbbqknr/pppp4/5p2/4p1pp/P7/2N2PP1/1PPPP2P/R1BBQKNR w HAha - 0 9",
     {23, 595, 14651, 415772, 10881112, 329010121}},  // 517
    {"rn1qkbnr/p1p1pp1p/bp4p1/3p4/1P6/4P3/P1PP1PPP/RNBQKBNR w HAha - 0 9",
     {30, 794, 24319, 690811, 21657601, 647745807}},  // 518
    {"r1bqk1rb/pppnpppp/5n2/3p4/2P3PP/2N5/PP1PPP2/R1BQKNRB w GAga - 1 9",
     {32, 821, 27121, 733155, 24923473, 710765657}},  // 519
    {"rbnqbknr/1p1ppp1p/6p1/p1p5/7P/3P4/PPP1PPP1/RBNQBKNR w HAha - 0 9",
     {24, 720, 18842, 575027, 15992882, 501093456}},  // 520
    {"r1qbbk1r/pp1ppppp/n1p5/5n2/B1P3P1/8/PP1PPP1P/RNQ1BKNR w HAha - 0 9",
     {27, 831, 22293, 698986, 19948650, 637973209}},  // 521
    {"rnqkbb1r/p1pppppp/8/8/1p4n1/PP4PP/2PPPP2/RNQKBBNR w HAha - 0 9",
     {18, 463, 9519, 256152, 6065231, 172734380}},  // 522
    {"rnqk1nrb/pppbpp2/7p/3p2p1/4B3/2N1N1P1/PPPPPP1P/R1QKB1R1 w GAga - 0 9",
     {34, 1171, 38128, 1318217, 42109356, 1465473753}},  // 523
    {"rbnqknbr/1pp1ppp1/3p4/7p/p2P2PP/2P5/PP2PP2/RBNQKNBR w HAha - 0 9",
     {32, 867, 28342, 798722, 26632459, 781067145}},  // 524
    {"rn1bknbr/pq2pppp/1p6/2pp4/P7/1P1P4/2PNPPPP/RNQBK1BR w HAha - 0 9",
     {24, 627, 16652, 462942, 13200921, 385193532}},  // 525
    {"r1qk1bbr/ppp1pp1p/2np1n2/6p1/2PP4/3BP3/PP3PPP/RNQKN1BR w HAha - 2 9",
     {31, 992, 30213, 986631, 30397368, 1011631987}},  // 526
    {"r1qknrbb/pppp1p2/2n3p1/4p2p/8/QPP5/P1NPPPPP/RN1K1RBB w FAfa - 2 9",
     {30, 702, 21563, 532939, 16813114, 438096194}},  // 527
    {"bbkr1qnr/2pppppp/2n5/pp6/8/PPN5/1BPPPPPP/1BR1KQNR w HC - 2 9",
     {25, 573, 15183, 380910, 10554668, 283975400}},  // 528
    {"1rnbkqnr/1bpppppp/1p6/7P/p2P4/5P2/PPP1P1P1/BRNBKQNR w HBhb - 0 9",
     {21, 503, 11790, 301084, 7679979, 207799378}},  // 529
    {"brnkqbnr/2p1pppp/1p6/3p4/1pP5/P6P/3PPPP1/BRNKQBNR w HBhb - 0 9",
     {28, 743, 21054, 587192, 17354516, 507176753}},  // 530
    {"br1kqnrb/npp1pppp/8/3p4/p4N2/PP6/2PPPPPP/BR1KQNRB w GBgb - 0 9",
     {31, 808, 25585, 698475, 22376575, 640362920}},  // 531
    {"rbbnkq1r/pppppp1p/7n/6p1/P5P1/2P2N2/1P1PPP1P/RBBNKQ1R w HAha - 1 9",
     {29, 580, 17585, 404831, 12730970, 325226128}},  // 532
    {"rnbbk1nr/pp2qppp/2ppp3/8/3P4/P1N4N/1PP1PPPP/R1BBKQ1R w HAha - 0 9",
     {29, 838, 24197, 721884, 21100580, 646624429}},  // 533
    {"rnbk1b1r/ppppn1pp/4pp2/7q/7P/P5PB/1PPPPP2/RNBKQ1NR w HAha - 3 9",
     {20, 729, 16633, 576199, 14507076, 498621813}},  // 534
    {"r2kqnrb/pbppppp1/np5p/8/4Q1P1/3P4/PPP1PP1P/RNBK1NRB w GAga - 2 9",
     {47, 1219, 55009, 1486353, 65239153, 1834391369}},  // 535
    {"rbnkbq1r/p1p2ppp/1p2pn2/3p4/P3P3/3P4/1PP1KPPP/RBN1BQNR w ha - 2 9",
     {29, 923, 27179, 883866, 26202752, 868565895}},  // 536
    {"rk1bb1nr/ppppqppp/n7/1N2p3/6P1/7N/PPPPPP1P/R1KBBQ1R w HA - 6 9",
     {27, 703, 19478, 559525, 16049807, 492966455}},  // 537
    {"rnkqbbnr/p1ppp2p/1p4p1/8/1B3p1P/2NP4/PPP1PPP1/R1KQ1BNR w HAha - 0 9",
     {29, 610, 18855, 438277, 14020041, 355083962}},  // 538
    {"rnkqb1rb/pp1p1ppp/4p3/2P3n1/8/1PP5/P3PPPP/RNKQBNRB w GAga - 0 9",
     {29, 675, 20699, 535821, 17000613, 476598337}},  // 539
    {"rb1kqnbr/pp1pp1p1/1np2p2/7p/P1P3PP/8/1P1PPP2/RBNKQNBR w HAha - 0 9",
     {31, 1077, 33661, 1183381, 37415304, 1328374620}},  // 540
    {"rnkbq1br/ppp2ppp/3p4/Q3p1n1/5P2/3P2P1/PPP1P2P/RNKB1NBR w HAha - 0 9",
     {41, 1201, 46472, 1420367, 52991625, 1675608008}},  // 541
    {"rn1qnbbr/pp2pppp/2ppk3/8/2PP4/3Q1N2/PP2PPPP/RNK2BBR w HA - 1 9",
     {34, 666, 22474, 472299, 15860369, 353831792}},  // 542
    {"rnkqnr1b/ppppp1pp/5p2/8/Q1P2P2/8/PP1P2PP/RbK1NRBB w FAfa - 0 9",
     {36, 876, 31987, 788580, 29022529, 736717252}},  // 543
    {"bbrn1nqr/ppp1k1pp/5p2/3pp3/7P/3PN3/PPP1PPP1/BBRK1NQR w - - 1 9",
     {24, 583, 15063, 383532, 10522064, 280707118}},  // 544
    {"brnbkn1r/1pppp1p1/4q3/p4p1p/7P/1N3P2/PPPPP1PQ/BR1BKN1R w HBhb - 2 9",
     {27, 935, 26120, 885699, 26000648, 873063158}},  // 545
    {"br1knbqr/pp2p1pp/1n6/2pp1p2/6P1/2P4B/PP1PPPQP/BRNKN2R w HBhb - 0 9",
     {27, 681, 19202, 510687, 14954779, 415624943}},  // 546
    {"brnk1qrb/p1ppppp1/1p5p/8/P3n3/1N4P1/1PPPPPRP/BR1KNQ1B w Bgb - 0 9",
     {22, 638, 13991, 412346, 9760752, 293499724}},  // 547
    {"rbbnknqr/pppp3p/5pp1/8/1P1pP3/7P/P1P2PP1/RBBNKNQR w HAha - 0 9",
     {29, 756, 21616, 614074, 17602252, 528140595}},  // 548
    {"1nbbknqr/rpp1ppp1/1Q1p3p/p7/2P2PP1/8/PP1PP2P/RNBBKN1R w HAh - 2 9",
     {37, 977, 34977, 944867, 33695089, 940198007}},  // 549
    {"rnb2bqr/ppkpppp1/3n3p/2p5/6PP/2N2P2/PPPPP3/R1BKNBQR w HA - 2 9",
     {30, 647, 20365, 467780, 15115531, 369257622}},  // 550
    {"rn1k1qrb/p1pppppp/bp6/8/4n3/P4BPP/1PPPPP2/RNBKNQR1 w GAga - 2 9",
     {22, 670, 14998, 451517, 11199653, 339919682}},  // 551
    {"rb2bnqr/nppkpppp/3p4/p7/1P6/P2N2P1/2PPPP1P/RB1KBNQR w HA - 3 9",
     {22, 479, 11475, 264739, 6831555, 167329117}},  // 552
    {"r1kbb1qr/2pppppp/np2n3/p7/2P3P1/8/PP1PPPQP/RNKBBN1R w HAha - 1 9",
     {32, 723, 23953, 581832, 19472074, 504622114}},  // 553
    {"rnknbb1r/p1ppp1pp/8/1p1P1p1q/8/P1P5/1P2PPPP/RNKNBBQR w HAha - 1 9",
     {19, 607, 12733, 417451, 9753617, 325177085}},  // 554
    {"rnkn1qrb/pp1bp1pp/2p5/1N1p1p2/8/2P5/PPKPPPPP/R2NBQRB w ga - 2 9",
     {27, 533, 14549, 330747, 9206957, 232664675}},  // 555
    {"r1nknqbr/pp2p1pp/2p2p2/3p4/6P1/PP1P4/2P1PP1b/RBNKNQBR w HAha - 0 9",
     {20, 582, 13777, 409166, 10708639, 326565393}},  // 556
    {"rnkb1qbr/p1pp1p1p/1p2pn2/1Q4p1/4P3/N4P2/PPPP2PP/R1KBN1BR w HAha - 0 9",
     {40, 1038, 39356, 1051441, 39145902, 1079612614}},  // 557
    {"rn2qbbr/1pkppp1p/p3n1p1/8/8/2P2P2/PP1PP1PP/RNKN1BBR w HA - 0 9",
     {24, 605, 14888, 385964, 9687507, 260874068}},  // 558
    {"rn1nqrbb/p1kppp1p/8/1pp3p1/1P6/2N1P3/P1PP1PPP/RK1NQRBB w - - 0 9",
     {21, 540, 12489, 337997, 8436136, 237525904}},  // 559
    {"bbrnknrq/1pp3pp/p2p1p2/4p3/P7/1P2N3/2PPPPPP/BBRN1RKQ w gc - 0 9",
     {24, 527, 13900, 326175, 9139962, 226253685}},  // 560
    {"brnb1nrq/pppp1kpp/4p3/8/5p1P/P1P3P1/1P1PPP2/BRNBKNRQ w GB - 1 9",
     {29, 773, 23904, 638768, 20503775, 560338709}},  // 561
    {"br1k1brq/ppppp2p/1n1n1pp1/8/P1P5/3P2P1/1P2PP1P/BRNKNBRQ w GBgb - 0 9",
     {28, 811, 23550, 664880, 19913758, 565143976}},  // 562
    {"1r1knrqb/n1pppppp/p1b5/1p6/8/3N1P2/PPPPP1PP/BRNK1RQB w fb - 3 9",
     {29, 753, 23210, 620019, 20044474, 558383603}},  // 563
    {"rbbnk1rq/pppppppp/8/3Pn3/8/4P1P1/PPP2P1P/RBBNKNRQ w GAga - 1 9",
     {22, 551, 12619, 324608, 8204171, 217689974}},  // 564
    {"rnbbk1rq/2pppp1p/p3n1p1/1p6/P3N3/8/1PPPPPPP/RNBB1KRQ w ga - 0 9",
     {26, 742, 20061, 599527, 16787080, 525678162}},  // 565
    {"rnbkn1rq/ppppppb1/6p1/7p/2B2P2/1P2P3/P1PP2PP/RNBKN1RQ w GAga - 1 9",
     {28, 799, 23210, 689436, 20755098, 639632905}},  // 566
    {"rn1knrqb/p2pppp1/b1p5/1p5p/2P2P2/1P6/P2PP1PP/RNBKNRQB w FAfa - 1 9",
     {30, 579, 18481, 397545, 13257198, 311282465}},  // 567
    {"rbnkbnrq/pp2p1Np/2p2p2/8/3p4/8/PPPPPPPP/RBNKBR1Q w Aga - 0 9",
     {23, 670, 16435, 501883, 13012378, 411860744}},  // 568
    {"rk1bbnrq/ppp1pppp/n7/3p4/5P2/3P2NP/PPP1P1P1/RNKBB1RQ w GA - 0 9",
     {26, 597, 16238, 402506, 11269462, 296701249}},  // 569
    {"r1knbbrq/pppp2p1/2n1p2p/5p2/4P3/P1PP4/1P3PPP/RNKNBBRQ w GAga - 1 9",
     {20, 596, 13091, 399069, 9416862, 293659781}},  // 570
    {"rnknbrqb/p1p1pp1p/3p4/1p1N2p1/8/N7/PPPPPPPP/1RK1BRQB w Ffa - 0 9",
     {26, 724, 18942, 552040, 15257204, 461293885}},  // 571
    {"rbnknrb1/1p1ppp1p/p1p3p1/8/1P3P2/1R6/PqPPP1PP/RBNKN1BQ w Afa - 0 9",
     {31, 1183, 34723, 1289502, 38722152, 1421492227}},  // 572
    {"rnkbnrbq/2p1ppp1/p7/1p1p3p/3P4/1P4P1/P1P1PP1P/RNKBNRBQ w FAfa - 0 9",
     {24, 506, 12748, 301464, 8086100, 207129256}},  // 573
    {"r1knrbbq/pp1ppppp/2p1n3/8/2P3P1/P7/1PKPPP1P/RN1NRBBQ w ea - 0 9",
     {28, 570, 16037, 352471, 10278695, 242592363}},  // 574
    {"rnknrq1b/ppp1p1p1/4b3/3p1p1p/6P1/P4P2/1PPPPQ1P/RNKNR1BB w EAea - 2 9",
     {30, 739, 23124, 594962, 19252739, 521629794}},  // 575
    {"bbqr1krn/pppp1p1p/5n2/4p1p1/3P4/P3QP2/1PP1P1PP/BB1RNKRN w GDgd - 0 9",
     {31, 799, 25627, 674913, 22172123, 609277274}},  // 576
    {"bq1b1krn/pp1ppppp/3n4/2r5/3p3N/6N1/PPP1PPPP/BQRB1KR1 w GCg - 2 9",
     {21, 798, 18571, 688429, 17546069, 647165916}},  // 577
    {"bqrnkbrn/2pp1pp1/p7/1p2p2p/1P6/4N3/P1PPPPPP/BQR1KBRN w GCgc - 0 9",
     {27, 783, 22327, 670798, 20059741, 624462073}},  // 578
    {"bqr1krnb/1np1pppp/8/pp1p4/8/2P2N2/PP1PPPPP/BQRNKR1B w FCfc - 0 9",
     {28, 636, 18874, 461104, 14237097, 372181570}},  // 579
    {"qbb1rkrn/1ppppppp/p7/7n/8/P2P4/1PP1PPPP/QBBRNKRN w Gg - 0 9",
     {25, 547, 13837, 332918, 8849383, 229112926}},  // 580
    {"1rbbnkrn/p1p1pp1p/2q5/1p1p2p1/8/2P3P1/PP1PPP1P/QRBBNKRN w GBgb - 2 9",
     {24, 1010, 24370, 983770, 24328258, 961371180}},  // 581
    {"qrb1kbrn/ppp1p2p/4npp1/3p4/8/1PP4P/PR1PPPP1/Q1BNKBRN w Ggb - 1 9",
     {18, 451, 9291, 247310, 5568106, 155744022}},  // 582
    {"qr2krnb/p1p1pppp/b1np4/1p6/3NP3/7P/PPPP1PP1/QRBNKR1B w FBfb - 2 9",
     {25, 667, 17081, 476030, 12458875, 361495148}},  // 583
    {"qbrnbkrn/ppp3pp/3p4/5p2/2P1pP2/6PP/PP1PP3/QBRNBKRN w GCgc - 0 9",
     {24, 650, 16835, 445263, 12187382, 326834539}},  // 584
    {"qrnb1krn/ppp1p1pp/5p2/2Np4/b2P4/2P5/PP2PPPP/QR1BBKRN w GBgb - 0 9",
     {27, 641, 17490, 432041, 12103076, 310695797}},  // 585
    {"qrnkbbrn/pp2pp2/8/2pp2pp/6PP/3P4/PPPKPP2/QRN1BBRN w gb - 0 9",
     {22, 554, 13116, 357404, 9014737, 258925091}},  // 586
    {"qrnkbrnb/p1p1ppp1/1p6/3p4/3P3p/5N1P/PPP1PPP1/QRNKBR1B w FBfb - 0 9",
     {24, 529, 13205, 318722, 8295874, 213856651}},  // 587
    {"qbr1krbn/1pppp1pp/p7/5pn1/2PP4/8/PPB1PPPP/Q1RNKRBN w FCfc - 0 9",
     {26, 831, 21651, 696830, 18961456, 621884383}},  // 588
    {"1rnbkrbn/1qp1pppp/3p4/pp6/4P3/1NP4P/PP1P1PP1/QR1BKRBN w FBfb - 0 9",
     {24, 597, 15089, 404761, 10832084, 307793179}},  // 589
    {"q1rkrbbn/ppp1pppp/8/3p4/1PnP4/P7/1RP1PPPP/Q1NKRBBN w Ee - 1 9",
     {20, 520, 10769, 278067, 6452205, 170268300}},  // 590
    {"qrnkrn1b/ppppp1pp/4b3/7P/6p1/P7/1PPPPP2/QRNKRNBB w EBeb - 0 9",
     {26, 566, 15623, 381312, 10940750, 287987207}},  // 591
    {"bbr1nkrn/ppp1pppp/3q4/3p4/8/P7/1PPPPPPP/BBRQNRKN w gc - 5 9",
     {19, 661, 13895, 460396, 10870247, 356399665}},  // 592
    {"brqbnkrn/pp1pp2p/5pp1/2p5/4P3/P2P1N2/1PP2PPP/BRQB1KRN w GBgb - 0 9",
     {27, 679, 19916, 527306, 16391730, 455940859}},  // 593
    {"2qnkbrn/p1pppppp/8/1r6/1p2bP2/7N/PPPPP1PP/BR1QKBRN w GBg - 4 9",
     {18, 774, 15713, 635461, 14371755, 559579332}},  // 594
    {"r1qnkr1b/p1pppppp/7n/1p6/8/1P3b1N/PRPPPPPP/B1QNK1RB w f - 5 9",
     {21, 677, 15437, 501520, 12463801, 410795298}},  // 595
    {"rbbqn1rn/pppp1pp1/3k4/4p2Q/2PPP3/8/PP3PPP/RBB1NKRN w GA - 1 9",
     {40, 742, 28757, 579833, 21852196, 471452088}},  // 596
    {"rqbbnkrn/3pppp1/p1p4p/1p6/5P2/P2N4/1PPPP1PP/RQBBK1RN w ga - 0 9",
     {23, 665, 16400, 492544, 12794736, 396640086}},  // 597
    {"r2nkbrn/pp2pppp/8/2ppqb2/2P3P1/5P2/PP1PPN1P/RQB1KBRN w GAga - 3 9",
     {28, 1108, 31164, 1194581, 34780853, 1292405738}},  // 598
    {"rqbnk1nb/p1pppr1p/5p2/1p4p1/1PP1P3/8/P2P1PPP/RQBNKRNB w FAa - 1 9",
     {26, 650, 18208, 491403, 14565370, 416833400}},  // 599
    {"rbqnb1rn/p1pp1kpp/1p2pp2/8/4P2P/P5P1/1PPP1P2/RBQNBKRN w GA - 0 9",
     {20, 437, 9423, 222154, 5282124, 132309824}},  // 600
    {"rqnbbkrn/p1p1pppp/3p4/1p5B/8/1P1NP3/P1PP1PPP/RQ2BKRN w GAga - 0 9",
     {30, 606, 18382, 422491, 12989786, 326601372}},  // 601
    {"rqnkbbr1/ppppp1pp/5p2/7n/8/2PNP2P/PP1P1PP1/RQ1KBBRN w GAga - 1 9",
     {23, 482, 12506, 297869, 8430874, 217797292}},  // 602
    {"r1nkbrnb/2ppppp1/1q6/pp5p/1P6/P3P3/2PPKPPP/RQN1BRNB w fa - 2 9",
     {25, 827, 21518, 701071, 19290675, 632892337}},  // 603
    {"rbqnkrbn/p1ppppp1/7p/1p6/7P/2N1P3/PPPP1PPB/RBQ1KR1N w FAfa - 1 9",
     {30, 627, 18566, 440217, 12976682, 337377291}},  // 604
    {"r1nbkrbn/p1qp1ppp/8/1pp1p3/2P1P3/6P1/PP1PBP1P/RQN1KRBN w FAfa - 2 9",
     {22, 616, 14503, 431199, 10850952, 335943324}},  // 605
    {"rqnkr1bn/ppp1ppb1/3p2pp/8/P7/2P2P2/1PKPP1PP/RQN1RBBN w ea - 1 9",
     {31, 679, 21365, 493500, 15661072, 379844460}},  // 606
    {"r2krnbb/qppp1ppp/1n6/p3p3/PP6/4N3/N1PPPPPP/RQ1KR1BB w EAea - 4 9",
     {24, 645, 17054, 487028, 13837270, 416239106}},  // 607
    {"bbr1qk1n/1ppppp1p/2n5/p7/P7/1P2P3/2PP1PrP/1BRNQKRN w GCc - 0 9",
     {18, 520, 10680, 304462, 7215306, 207612575}},  // 608
    {"brnbq1rn/2ppppkp/p5p1/1p6/8/1BP3P1/PP1PPP1P/BRN1QRKN w - - 0 9",
     {21, 625, 13989, 419667, 9929336, 300902534}},  // 609
    {"brn1kbrn/pp2p1pp/3p4/q1p2p2/2P4P/6P1/PP1PPP2/BRNQKBRN w GBgb - 1 9",
     {18, 477, 10205, 273925, 6720181, 187205941}},  // 610
    {"brn1krnb/p3pppp/1qpp4/1p6/2P3P1/1P6/P2PPP1P/BRNQKRNB w FBfb - 1 9",
     {30, 835, 24761, 716151, 21806428, 654487872}},  // 611
    {"r1b1qkrn/1p1ppppp/p1p1n3/8/4P3/1PN5/P1PPQPPb/RBB2KRN w GAga - 0 9",
     {28, 825, 24536, 716585, 22079005, 647939781}},  // 612
    {"r1bbqk1n/p1pppprp/n7/1p4p1/5P2/2N3N1/PPPPP1PP/1RBBQKR1 w Ga - 4 9",
     {25, 545, 14657, 358854, 10271111, 273864588}},  // 613
    {"rnbqkbrn/p1pp1pp1/4p3/7p/2p4P/2P5/PP1PPPP1/R1BQKBRN w GAga - 0 9",
     {17, 445, 9076, 255098, 5918310, 174733195}},  // 614
    {"rnbqkrnb/1p1pp1p1/2p4p/p4p2/3P2P1/7N/PPPBPP1P/RN1QKR1B w FAfa - 0 9",
     {34, 746, 25319, 623133, 21285553, 569141201}},  // 615
    {"rbnqbkr1/1ppppp2/p5n1/6pp/4P3/1N6/PPPP1PPP/RBQ1BRKN w ga - 2 9",
     {18, 466, 9683, 260864, 6051500, 170135726}},  // 616
    {"rnqb1krn/ppppp1p1/7p/7b/P1P2pPP/8/1P1PPP2/RNQBBKRN w GAga - 0 9",
     {24, 575, 15400, 385825, 11039042, 291243811}},  // 617
    {"rnqkbbr1/p1pp1ppp/4p3/1p6/P3P2n/5P2/1PPP1NPP/RNQKBBR1 w GAga - 2 9",
     {27, 803, 22883, 694449, 20666099, 638696065}},  // 618
    {"rn1kbrnb/1qppp1pp/1p6/p4p2/1B1P4/1P5N/P1P1PPPP/RNQK1R1B w FAfa - 0 9",
     {37, 1209, 43015, 1425600, 49748034, 1671593862}},  // 619
    {"rbnqkrbn/Bppp1p2/p5pp/4p3/5P2/6PP/PPPPP3/RBNQKR1N w FAfa - 0 9",
     {29, 720, 20434, 534148, 15384362, 421343249}},  // 620
    {"rnqbkr1n/1p1ppbpp/3p1p2/p7/8/1P6/P1PPPPPP/R1QBKRBN w FAfa - 0 9",
     {20, 657, 14424, 492678, 11843134, 413965054}},  // 621
    {"rnqkrb1n/ppppp3/6p1/5p1p/2b2P2/P1N5/1PPPP1PP/RQ1KRBBN w EAea - 1 9",
     {28, 749, 20684, 543151, 15379233, 417191461}},  // 622
    {"rnqk1nbb/1pp2ppp/3pr3/p3p3/3P1P2/2N3N1/PPP1P1PP/R1QKR1BB w EAa - 1 9",
     {29, 883, 26412, 815098, 25144295, 789705382}},  // 623
    {"bbr1kqrn/p1p1ppp1/1p2n2p/3p4/1P1P4/2N5/P1P1PPPP/BBR1KQRN w GCgc - 0 9",
     {22, 485, 11475, 271271, 6825123, 171793012}},  // 624
    {"brnbkq1n/ppp1ppr1/7p/3p2p1/2P3PP/8/PPBPPP2/BRN1KQRN w GBb - 2 9",
     {30, 634, 19017, 442537, 13674310, 345386924}},  // 625
    {"brnkqbr1/1pppp1pp/5p2/p7/P1P1P2n/8/1P1P1PP1/BRNKQBRN w GBgb - 0 9",
     {21, 504, 11672, 305184, 7778289, 217596497}},  // 626
    {"b1rkqrnb/p1ppp1pp/1p1n4/5p2/5P2/PN5P/1PPPP1P1/BR1KQRNB w FBf - 0 9",
     {23, 688, 17259, 531592, 14228372, 451842354}},  // 627
    {"1bbnkqrn/rppppp2/p5p1/7p/7P/P1P1P3/1P1P1PP1/RBBNKQRN w GAg - 1 9",
     {25, 450, 12391, 263946, 7752404, 185393913}},  // 628
    {"rnbbkqr1/1pppppp1/7p/p3n3/PP5P/8/1BPPPPP1/RN1BKQRN w GAga - 0 9",
     {23, 543, 12224, 305812, 7549008, 199883770}},  // 629
    {"r1bkqbrn/ppppp1pp/8/5p2/3nPP2/1P4N1/P1PP2PP/RNBKQBR1 w GAga - 1 9",
     {27, 751, 21158, 600417, 17989920, 527273615}},  // 630
    {"rnbkqr1b/1p1pp1pp/p4p1n/2p5/1P5P/N4P2/P1PPP1P1/R1BKQRNB w FAfa - 0 9",
     {21, 498, 11738, 302278, 7808375, 216224115}},  // 631
    {"rbnkbqrn/p1p3pp/1p1p4/B3pp2/3P2P1/6N1/PPP1PP1P/RBNK1QR1 w GAga - 0 9",
     {34, 977, 33464, 961128, 33318567, 978991050}},  // 632
    {"r1kbbqrn/ppp3pp/2np1p2/1P2p3/3P1P2/8/P1P1P1PP/RNKBBQRN w GAga - 0 9",
     {32, 920, 28916, 844881, 26763259, 797524786}},  // 633
    {"rk1qbbrn/p2npppp/1p6/2p4Q/8/4P3/PPPP1PPP/RNK1B1RN w GA - 2 9",
     {35, 657, 22359, 495406, 16662477, 419496845}},  // 634
    {"rnk1brnb/pp1p1pp1/8/q1p1p2p/5P2/NP6/P1PPP1PP/R1KQBRNB w FAfa - 1 9",
     {26, 774, 20215, 610661, 16987110, 523437649}},  // 635
    {"rb1kqrbn/npp1ppp1/p7/3P3p/2PP4/8/PP3PPP/RBNKQRBN w FAfa - 0 9",
     {35, 775, 27395, 661118, 23983464, 625669222}},  // 636
    {"rnkb1rbn/pp1p2pp/8/2p1pp1q/P6P/1PN5/2PPPPP1/R1KBQRBN w FAfa - 1 9",
     {22, 899, 21188, 850597, 21518343, 857951339}},  // 637
    {"rnkqrbbn/1pppp1p1/8/p2N1p1p/2P4P/8/PP1PPPP1/R1KQRBBN w EAea - 0 9",
     {29, 585, 17571, 393221, 12238776, 299752383}},  // 638
    {"rnk1r1bb/pp1ppppp/1q4n1/2p5/5P1P/3PP3/PPP3P1/RNKQRNBB w EAea - 1 9",
     {27, 884, 24613, 811915, 23698701, 790239502}},  // 639
    {"bbrnkrqn/1ppp1p2/6pp/p3p3/5PP1/2PB4/PP1PP2P/B1RNKRQN w FCfc - 0 9",
     {37, 693, 25425, 550527, 20138432, 481498664}},  // 640
    {"b1rbkrqn/ppp2ppp/1n2p3/3p4/6P1/2PP4/PP2PP1P/BRNBKRQN w FBf - 1 9",
     {21, 463, 10610, 253204, 6307276, 159025909}},  // 641
    {"brnkrb1n/1pp1p1pp/3p4/p1Nq1p2/2P5/8/PP1PPPPP/BRK1RBQN w eb - 2 9",
     {27, 725, 17842, 496072, 12604078, 362747791}},  // 642
    {"brn1r1nb/ppppkppp/4p3/8/2PP1P2/8/PP1KP1PP/BRN1RQNB w - - 1 9",
     {25, 623, 16874, 426659, 12290985, 317097424}},  // 643
    {"rbb1krqn/1pp1pp1p/p3n1p1/3pP3/8/1PN5/P1PP1PPP/RBB1KRQN w FAfa d6 0 9",
     {23, 529, 12641, 310277, 7861413, 202594556}},  // 644
    {"r1bbkrqn/p1pppppp/8/4n3/1p5P/P2P2P1/1PP1PP2/RNBBKRQN w FAfa - 0 9",
     {23, 571, 13133, 346793, 8699448, 243460643}},  // 645
    {"rnbkrbqn/p1pp1ppp/4p3/1p6/8/BPN3P1/P1PPPP1P/R2KRBQN w EAea - 2 9",
     {29, 692, 20014, 500375, 14904192, 386694739}},  // 646
    {"rnbkrqn1/pppppp2/8/1Q2b1pp/P3P3/5P2/1PPP2PP/RNBKR1NB w EAea - 0 9",
     {37, 1001, 36440, 987842, 35626426, 993747544}},  // 647
    {"rbnkbrqn/p1pppp2/7p/1p4pP/3P1P2/8/PPP1P1P1/RBNKBRQN w FAfa - 0 9",
     {30, 564, 17143, 381364, 11859538, 293703269}},  // 648
    {"1nkbbrqn/3ppppp/r1p5/pp6/8/4PP2/PPPPN1PP/RNKBBRQ1 w FAf - 2 9",
     {26, 546, 14641, 344592, 9556962, 245137199}},  // 649
    {"rnkrbbq1/pppppnp1/7p/8/1B1Q1p2/3P1P2/PPP1P1PP/RNKR1B1N w DAda - 2 9",
     {43, 887, 36240, 846858, 33185346, 851927292}},  // 650
    {"1rkrbqnb/pppppp2/2n3p1/7p/3P3P/P4N2/1PP1PPP1/RNKRBQ1B w DAd - 0 9",
     {26, 622, 16049, 403921, 10786140, 285233838}},  // 651
    {"rbnkr1bn/pp1pqp1p/2p1p3/6p1/3P4/7P/PPP1PPP1/RBNKRQBN w EAea - 0 9",
     {19, 566, 12257, 381197, 9107175, 293397389}},  // 652
    {"r1kbrqb1/pppp2pp/2n1p1n1/5p1B/4PP2/P7/1PPP2PP/RNK1RQBN w EAea - 2 9",
     {39, 1359, 53626, 1876028, 73871486, 2633945690}},  // 653
    {"rnkrqbbn/p1p3pp/1p1ppp2/8/1P6/3P2P1/PKP1PP1P/RN1RQBBN w da - 0 9",
     {26, 776, 20735, 611907, 16884013, 503561996}},  // 654
    {"rnkrqnbb/ppp2p1p/3p4/4p1p1/3P3P/N1Q5/PPP1PPP1/R1KR1NBB w DAda - 0 9",
     {40, 1175, 45637, 1375884, 52620163, 1633655838}},  // 655
    {"bbrnkrn1/p1pppp2/1p6/6pp/3q4/1P3QP1/P1PPPP1P/BBRNKRN1 w FCfc - 0 9",
     {34, 1398, 45749, 1712950, 57268492, 2059942014}},  // 656
    {"br1bkrnq/1p2pppp/pnp5/3p4/P1P5/5P2/1P1PPKPP/BRNB1RNQ w fb - 2 9",
     {24, 501, 12237, 284936, 7049659, 177940764}},  // 657
    {"brnkrbn1/pppppp1q/B6p/6p1/8/1P2PP2/P1PP2PP/BRNKR1NQ w EBeb - 0 9",
     {34, 815, 25868, 700970, 22006883, 639803952}},  // 658
    {"br1krnqb/pppppp1p/1n4p1/8/8/P2NN3/2PPPPPP/BR1K1RQB w Beb - 2 9",
     {37, 1029, 36748, 1025712, 36214583, 1026195877}},  // 659
    {"rbbnkr1q/p1p2ppp/1p1ppn2/8/1PP4P/8/P2PPPP1/RBBNKRNQ w FAfa - 0 9",
     {28, 755, 22623, 605106, 18972778, 513486101}},  // 660
    {"r1b1krnq/pp2pppp/1bn5/2pp4/4N3/5P2/PPPPPRPP/R1BBK1NQ w Afa - 0 9",
     {24, 705, 17427, 532521, 13532966, 426443376}},  // 661
    {"1nbkrbn1/rpppppqp/p7/6p1/4P3/3P2P1/PPP1KP1P/RNB1RBNQ w e - 1 9",
     {31, 800, 24748, 693366, 21193292, 625757852}},  // 662
    {"r1bkrnqb/pp3ppp/n1ppp3/8/1P5P/P7/R1PPPPP1/1NBKRNQB w Eea - 0 9",
     {21, 482, 11417, 275339, 7112890, 180378139}},  // 663
    {"rbnkbrnq/ppp1p2p/5p2/3p2p1/1B1P4/1N4P1/PPP1PP1P/RB1K1RNQ w FAfa - 0 9",
     {33, 780, 25532, 628945, 20756770, 535497008}},  // 664
    {"rnk1brnq/pp1ppppp/2p5/b7/8/1P2P2P/P1PP1PPQ/RNKBBRN1 w FAfa - 3 9",
     {29, 648, 19043, 449637, 13722785, 341389148}},  // 665
    {"rnkrbbnq/p1p3pp/5p2/1p1pp3/P7/1PN2P2/2PPP1PP/R1KRBBNQ w DAda - 0 9",
     {26, 827, 21865, 683167, 18916370, 589161126}},  // 666
    {"r1krbnqb/p1pp1ppp/2n1p3/8/1p4P1/PPP5/3PPP1P/RNKRBNQB w DAda - 1 9",
     {25, 540, 14709, 331332, 9491817, 225389422}},  // 667
    {"rbnkrnbq/ppp1pp2/3p2p1/2N5/P6p/2P5/1P1PPPPP/RB1KRNBQ w EAea - 0 9",
     {32, 790, 25107, 661207, 20906017, 578332225}},  // 668
    {"rnkbrn1q/1ppppppb/8/p4N1p/8/P1N5/1PPPPPPP/R1KBR1BQ w EAea - 0 9",
     {31, 691, 20813, 510665, 15308408, 404129987}},  // 669
    {"rnkrnbbq/p1p2ppp/3pp3/1p6/6P1/4PQ1B/PPPP1P1P/RNKRN1B1 w DAda - 0 9",
     {29, 558, 16800, 352887, 10825379, 246965507}},  // 670
    {"rnkrnqbb/pp2p1p1/3p3p/2p2p2/5P2/1P1N4/P1PPPQPP/RNKR2BB w DAda - 0 9",
     {29, 762, 23210, 644936, 20522675, 596067005}},  // 671
    {"bb1rknnr/ppqppppp/8/2p5/3P1N2/1P6/P1P1PPPP/BBQRKN1R w HDhd - 1 9",
     {33, 963, 32279, 1000890, 34552118, 1124738493}},  // 672
    {"bqrbknnr/ppp1p2p/8/3p1p2/5p2/P3N2P/1PPPP1P1/BQRBK1NR w HChc - 0 9",
     {20, 398, 9009, 194859, 4834319, 113660536}},  // 673
    {"b1rk1bnr/qpp1pppp/p4n2/3p4/3PPP2/7N/PPP3PP/BQRKNB1R w HChc - 1 9",
     {25, 648, 16587, 455720, 12200870, 351766307}},  // 674
    {"bqkrnnrb/pppp2p1/4pp2/4P2p/6P1/7P/PPPP1P2/BQRKNNRB w GC - 1 9",
     {30, 493, 15118, 280726, 8786998, 181492621}},  // 675
    {"q1brknnr/1p1ppppp/p7/2p5/8/1PPP4/P2RPPPP/QBB1KNNR w Hhd - 0 9",
     {25, 501, 13206, 290463, 7982978, 192717198}},  // 676
    {"qrb1k1nr/ppppb1pp/6n1/4ppN1/3P4/4N3/PPP1PPPP/QRBBK2R w HBhb - 2 9",
     {31, 872, 26191, 739276, 22493014, 646855304}},  // 677
    {"1rbknbnr/1ppp1pp1/q6p/p3p3/5P2/2PPB3/PP2P1PP/QR1KNBNR w HBhb - 0 9",
     {28, 1020, 28147, 984000, 27484692, 947786800}},  // 678
    {"qrbk2rb/1ppp1ppp/5nn1/p3p3/1N6/P7/1PPPPPPP/QRB1KNRB w gb - 0 9",
     {23, 592, 14398, 395716, 10098215, 293988585}},  // 679
    {"qbrk1nnr/1pp1pppp/2b5/p2p4/P2P2P1/8/1PP1PP1P/QBKRBNNR w hc - 1 9",
     {26, 654, 18103, 471653, 13740891, 373081138}},  // 680
    {"qrkbbnnr/ppp2p1p/4p3/3p2p1/P7/2PP4/1P2PPPP/QRKBBNNR w HBhb - 0 9",
     {25, 626, 16616, 431634, 12079406, 324006164}},  // 681
    {"qr1kbbnr/ppp1pp1p/4n1p1/2Pp4/6P1/4N3/PP1PPP1P/QRK1BBNR w HB d6 0 9",
     {26, 699, 18068, 497152, 13353359, 375702908}},  // 682
    {"qrk1b1rb/p1pppppp/3nnQ2/1p6/1P3P2/3P4/P1P1P1PP/1RKNBNRB w GBgb - 3 9",
     {43, 1369, 55463, 1831200, 71514365, 2427477375}},  // 683
    {"qbrk1nbr/pppp3p/5n2/4ppp1/3P1P2/4N3/PPP1P1PP/QBKRN1BR w hc - 0 9",
     {25, 752, 20165, 615263, 17493373, 543180234}},  // 684
    {"qrkb1nbr/1pppppQp/3n4/p7/5p2/1P1N4/P1PPP1PP/1RKB1NBR w HBhb - 0 9",
     {45, 946, 40100, 966903, 39736157, 1051910977}},  // 685
    {"qrk1nbbr/ppp1p1p1/4n2p/3p1p2/1P5P/3P2P1/P1P1PP2/QRKNNBBR w HBhb - 1 9",
     {32, 770, 25367, 646977, 21717615, 577979364}},  // 686
    {"qrkn1rbb/pp2pppp/2p5/3p4/P2Qn1P1/1P6/2PPPP1P/1RKNNRBB w FBfb - 0 9",
     {38, 943, 35335, 868165, 31909835, 798405123}},  // 687
    {"bbrqknnr/ppp4p/3pp3/5pp1/4PP2/5Q2/PPPP2PP/BBR1KNNR w HChc - 0 9",
     {36, 843, 29974, 758528, 26828059, 723306114}},  // 688
    {"1rqbkn1r/p1p1pppp/1p5n/P2p4/3Pb1P1/8/1PP1PP1P/BRQBKNNR w HBhb - 0 9",
     {23, 778, 19482, 649789, 17337683, 579112676}},  // 689
    {"br1knbnr/1qp1pppp/pp1p4/8/8/PP6/2PPPPPP/BRQKNBNR w HBhb - 2 9",
     {26, 697, 18835, 546622, 15280079, 473071890}},  // 690
    {"brqk2rb/ppppp1pp/4np2/8/2n5/3P1Q2/PP2PPPP/BR1KNNRB w GBgb - 0 9",
     {32, 948, 30434, 885713, 29821322, 874251866}},  // 691
    {"r1bqknnr/pp1pp1p1/5p1p/2p1b2N/2P5/8/PPQPPPPP/RBB1K1NR w HAha - 0 9",
     {31, 785, 25549, 659952, 22244193, 592797491}},  // 692
    {"rqbbknnr/ppppp2p/5pp1/8/8/1P3PP1/PQPPP2P/R1BBKNNR w HAha - 0 9",
     {23, 391, 10163, 198450, 5576671, 121267576}},  // 693
    {"rqbknbnr/1pp1p2p/p7/3p1pp1/7N/1PP5/P2PPPPP/RQBK1BNR w HAha - 0 9",
     {27, 676, 19606, 522428, 15955388, 448477218}},  // 694
    {"rqb1nnrb/2ppkppp/1p2p3/p7/2PPP3/1P6/P4PPP/RQBKNNRB w GA - 1 9",
     {31, 727, 22895, 570647, 18361051, 483248153}},  // 695
    {"rb1kbn1r/p1ppppp1/qp5n/7p/P7/RPP5/3PPPPP/1BQKBNNR w Hha - 2 9",
     {29, 837, 23815, 730083, 21279560, 682863811}},  // 696
    {"rqkbb1nr/p1p2ppp/1p1p2n1/3Np3/4P3/5N2/PPPP1PPP/RQKBB2R w HAha - 0 9",
     {28, 717, 20663, 550987, 16347343, 453153783}},  // 697
    {"rqknbbr1/p1pppp1p/1p3np1/8/4P3/2P2P1P/PP1P2P1/RQKNBBNR w HAa - 0 9",
     {27, 650, 18231, 475303, 13847463, 383256006}},  // 698
    {"r1k1bnrb/1qpppppp/1p2n3/p7/1P5P/6P1/P1PPPP2/RQKNBNR1 w GAga - 1 9",
     {24, 806, 20693, 713220, 19382263, 686009788}},  // 699
    {"rb1knnbr/1pp1ppp1/p2p3p/5q2/3B2P1/3P1P2/PPP1P2P/RBQKNN1R w HAha - 0 9",
     {34, 1360, 44096, 1605706, 51973672, 1837704407}},  // 700
    {"rqkb1nbr/p1p1ppp1/1p3n1p/2Qp4/8/2P5/PP1PPPPP/R1KBNNBR w HAha - 2 9",
     {39, 983, 38218, 940989, 36347815, 918801645}},  // 701
    {"rqknnbbr/2pppp2/pp5p/6p1/1P1P4/4PP2/P1P3PP/RQKNNBBR w HAha - 0 9",
     {26, 628, 17638, 464924, 13787303, 386125234}},  // 702
    {"rqkn1rbb/1pp1pppp/p7/3p4/3Pn3/2P1PP2/PP4PP/RQKNNRBB w FAfa - 1 9",
     {20, 527, 12216, 321533, 8082183, 219311659}},  // 703
    {"bbrkqn1r/1pppppp1/5n2/p7/1PP2P1p/7N/P2PP1PP/BBRKQN1R w HChc - 1 9",
     {36, 963, 35291, 973839, 35907489, 1034223364}},  // 704
    {"brkbqn1r/p2ppppp/7n/1p6/P1p3PP/8/1PPPPP1N/BRKBQ1NR w HBhb - 0 9",
     {18, 583, 11790, 394603, 8858385, 304339862}},  // 705
    {"brkq1bnr/pp1ppp1p/8/2p2np1/P7/8/1PPPPPPP/BRKQNBNR w HBhb - 0 9",
     {19, 552, 11811, 354260, 8432183, 262293169}},  // 706
    {"brkqnnrb/1ppppppp/8/8/p3P3/5N2/PPPP1PPP/BRKQ1NRB w GBgb - 3 9",
     {21, 397, 9653, 204350, 5489836, 128389738}},  // 707
    {"rbbkq1nr/1p2pppp/p1p3nB/3p4/1Q1P4/6N1/PPP1PPPP/RB1K2NR w HAha - 0 9",
     {40, 1132, 43404, 1260470, 47425783, 1415578783}},  // 708
    {"rkbbq1nr/1pppp1p1/4np2/p6p/8/PP3P2/1KPPP1PP/R1BBQNNR w ha - 0 9",
     {24, 596, 15220, 402121, 10822049, 302056813}},  // 709
    {"r1bqn1nr/pkpppp1p/1p4pb/8/PN6/R7/1PPPPPPP/1KBQ1BNR w H - 2 9",
     {33, 794, 25450, 649150, 20919309, 561073410}},  // 710
    {"rkb1nnrb/1pppq1pp/p4p2/4p3/5P2/1P1PB3/P1P1P1PP/RK1QNNRB w GAga - 0 9",
     {26, 625, 17050, 442036, 12515042, 342967558}},  // 711
    {"rbkqbn1r/pppp1p1p/2n1p1p1/8/8/1P1PP1N1/P1P2PPP/RBKQB1NR w HAha - 1 9",
     {30, 660, 20308, 492714, 15348335, 403323883}},  // 712
    {"rkqbb1n1/pppppppr/8/6np/5P2/8/PPPPP1PP/RKQBBNNR w HAa - 6 9",
     {23, 500, 12154, 292936, 7519117, 196524441}},  // 713
    {"rkqnbbnr/ppppppp1/8/7p/3N4/6PP/PPPPPP2/RKQNBB1R w HAa - 0 9",
     {24, 484, 12495, 284570, 7775173, 193947530}},  // 714
    {"rkqnb1rb/p1p1pppp/1p1p4/2n5/3P4/2P1N1N1/PP2PPPP/RKQ1B1RB w GAga - 0 9",
     {28, 1020, 29124, 1027904, 30515456, 1073711823}},  // 715
    {"rbk1nnbr/1ppq1ppp/p2p4/4p3/P3B2P/2P5/1P1PPPP1/R1KQNNBR w HAha - 2 9",
     {38, 998, 37265, 1047592, 38552638, 1139322479}},  // 716
    {"r1qbn1br/k1pppppp/6n1/pp6/5P1P/P7/1PPPP1PB/RKQBNN1R w HA - 1 9",
     {22, 549, 12867, 348574, 8725809, 251613569}},  // 717
    {"rkqnn1br/pppp3p/4p1pb/5p2/P2P4/7P/1PP1PPPB/RKQNNB1R w HAha - 1 9",
     {32, 659, 21249, 469701, 15434721, 365761521}},  // 718
    {"rk1nnrbb/p1p1pppp/1p6/3p1q2/P3P3/2NN4/1PPP1PPP/RKQ2RBB w FAfa - 3 9",
     {29, 989, 29087, 980477, 29643404, 998848556}},  // 719
    {"bbrk1q1r/ppppppp1/3n4/7p/3Pn3/6PN/PPP1PPNP/BBRK1Q1R w HChc - 2 9",
     {23, 712, 16551, 516177, 12995202, 411077508}},  // 720
    {"brkbnq1r/p1ppp2p/5ppn/1p6/5P2/1P1P2P1/P1P1P2P/BRKBNQNR w HBhb - 0 9",
     {28, 856, 24984, 780503, 23529352, 754501112}},  // 721
    {"br1k1bnr/ppppp1pp/4np2/1B2P2q/3P4/8/PPP2PPP/BRKNQ1NR w HB - 3 9",
     {36, 1214, 40615, 1328331, 45096834, 1470987023}},  // 722
    {"brk1qnrb/pnppp1p1/1p6/5p1p/8/5PPP/PPPPP1R1/BRKNQN1B w Bgb - 0 9",
     {22, 551, 13111, 353317, 9040545, 259643605}},  // 723
    {"rbbkn1nr/1ppp2pp/p3p3/2q2p2/3P4/6P1/PPPBPP1P/RB1KNQNR w HAha - 0 9",
     {31, 1060, 31332, 1015099, 30314172, 976268967}},  // 724
    {"rkbbn1nr/ppppp1pp/8/6N1/5p2/1q6/P1PPPPPP/RKBBN1QR w HAha - 0 9",
     {3, 72, 1919, 50827, 1400832, 39654253}},  // 725
    {"rkb2bnr/pp2pppp/2p1n3/3p4/q2P4/5NP1/PPP1PP1P/RKBNQBR1 w Aha - 0 9",
     {29, 861, 24504, 763454, 22763215, 731511256}},  // 726
    {"rkbq1nrb/ppppppp1/7p/8/1P1n4/P4P1P/2PPP1P1/RKBNQNRB w GAga - 0 9",
     {25, 672, 17631, 473864, 12954224, 361237536}},  // 727
    {"rbknb1nr/ppp1qp1p/6p1/3pp3/3P3P/2B1P3/PPP2PP1/RBKN1QNR w HAha - 1 9",
     {27, 857, 24688, 792538, 23790033, 768247869}},  // 728
    {"rknbbq1r/p1pppppp/1p2N3/8/3n4/2P5/PP1PPPPP/RK1BBQNR w HAha - 4 9",
     {29, 763, 22138, 574054, 16926075, 447896703}},  // 729
    {"r1nqbbnr/1pppp1pp/1k6/p4p2/8/4P3/PPPP1PPP/RKN1BBNR w HA - 0 9",
     {26, 658, 17302, 464039, 12380488, 349047256}},  // 730
    {"rkn2qrb/ppp1pppp/6n1/1b1p4/1P6/4PPB1/P1PP2PP/RKNQ1NRB w GAga - 3 9",
     {23, 574, 14070, 370324, 9501401, 263870337}},  // 731
    {"rbkn2br/ppppp1p1/4np1p/1P5q/8/2P1N3/P2PPPPP/RBK1QNBR w HAha - 1 9",
     {29, 992, 29506, 999564, 30148787, 1045942540}},  // 732
    {"1knbqnbr/1ppppp1p/r5p1/p7/7P/2PN2P1/PP1PPP2/RK1BQNBR w HAh - 2 9",
     {26, 698, 19395, 512023, 14848229, 402599313}},  // 733
    {"rk1qnbbr/pnpppp1p/6p1/1p6/3P4/1P6/P1P1PPPP/RKNQNBBR w HAha - 1 9",
     {20, 480, 11159, 287539, 7425917, 203194521}},  // 734
    {"rknqnrbb/pp1p2p1/5p1p/2p1p3/2P1P3/P2P4/1P3PPP/RKNQNRBB w FAfa - 0 9",
     {26, 679, 18116, 494953, 13790137, 392629571}},  // 735
    {"bbrk2qr/pp1p1ppp/3n2n1/2p1p3/3P1P2/6N1/PPP1P1PP/BBRKN1QR w HChc - 0 9",
     {26, 790, 21521, 673269, 19259490, 617563700}},  // 736
    {"b1krnnqr/1p1ppppp/p1p5/b6B/P7/4P1N1/1PPP1PPP/BRK1N1QR w HB - 2 9",
     {26, 625, 16451, 415452, 11490615, 304805107}},  // 737
    {"1rknnbqr/3ppppp/p7/1pp5/4b2P/P4P2/1PPPP1PR/BRKNNBQ1 w Bhb - 1 9",
     {24, 757, 19746, 618777, 17275100, 544309489}},  // 738
    {"br1nn1rb/pppkpqpp/3p1p2/8/PP6/4N3/1KPPPPPP/BR2NQRB w - - 3 9",
     {24, 682, 17129, 482711, 13057308, 375033550}},  // 739
    {"rbbkn1qr/pppp2p1/6np/4pp2/7N/7P/PPPPPPPR/RBBK1NQ1 w Aha - 0 9",
     {22, 586, 14158, 409891, 10607781, 324452612}},  // 740
    {"rk1bn1qr/pppbpppp/4n3/4p3/4P3/5P2/PPPP2PP/RKBB1NQR w HAha - 1 9",
     {22, 530, 13440, 348004, 9514787, 259898748}},  // 741
    {"rkbnnbqr/1ppp1ppp/p7/4p3/8/QP3P2/P1PPP1PP/RKBNNB1R w HAha - 0 9",
     {29, 705, 21511, 551042, 17524731, 472356665}},  // 742
    {"1kbnnqrb/1pp1p1pp/r4p2/p2p4/N4P2/3P4/PPP1P1PP/RKB1NQRB w GAg - 2 9",
     {21, 623, 14979, 437554, 11601134, 343214006}},  // 743
    {"rbknbn1r/pppp1p1p/4p1q1/8/P1P3Pp/8/1P1PPP2/RBKNBNQR w HAha - 0 9",
     {30, 813, 24959, 708454, 23379040, 692576573}},  // 744
    {"rk1bb1qr/2pppppp/p2nn3/1p4P1/6QP/8/PPPPPP2/RKNBBN1R w HAha - 2 9",
     {36, 857, 30124, 757524, 26485812, 696999449}},  // 745
    {"rkn1bbqr/p2ppppp/2p1n3/1p6/4PP2/6PP/PPPP4/RKNNBBQR w HAha - 0 9",
     {33, 687, 22744, 511018, 17101732, 412778368}},  // 746
    {"rkn1bqrb/pnp1pppp/3p4/8/Pp6/1N2NP2/1PPPP1PP/RK2BQRB w GAga - 0 9",
     {28, 591, 17174, 406025, 12182448, 312575205}},  // 747
    {"rbk1n1br/ppp1ppqp/2n5/2Np2p1/8/2P5/PPBPPPPP/R1KN1QBR w HAha - 4 9",
     {35, 930, 30663, 844433, 27160490, 780616047}},  // 748
    {"rknbn1br/1ppp1ppp/p3p3/8/1q6/2P2N1P/P2PPPP1/RKNB1QBR w HAha - 0 9",
     {4, 157, 3697, 138102, 3454704, 125373395}},  // 749
    {"rkn1qbbr/pp3ppp/4n3/2ppp3/4P1P1/P2P4/1PP2P1P/RKNNQBBR w HAha - 0 9",
     {28, 840, 24437, 771328, 23200961, 756489357}},  // 750
    {"rkn1qrbb/pp1ppp2/2p1n1p1/7p/2P2P1P/6P1/PP1PP3/RKNNQRBB w FAfa - 1 9",
     {32, 867, 27595, 757836, 24485663, 688115847}},  // 751
    {"b1rknnrq/bpppp1p1/p6p/5p1P/6P1/4N3/PPPPPP2/BBRKN1RQ w GCgc - 1 9",
     {33, 851, 28888, 763967, 26686205, 731944177}},  // 752
    {"brkb1nr1/pppppp2/3n2pp/3B4/1P6/4P3/PqPP1PPP/BRK1NNRQ w GBgb - 2 9",
     {4, 98, 2965, 76143, 2352530, 64251468}},  // 753
    {"brk1nbrq/1ppppn1p/6p1/p4p2/P5P1/5R2/1PPPPP1P/BRKNNB1Q w Bgb - 0 9",
     {29, 922, 27709, 879527, 27463717, 888881062}},  // 754
    {"brkn1rqb/1p1ppppp/3n4/p1p5/1P3P2/8/PNPPP1PP/BR1KNRQB w fb - 1 9",
     {29, 633, 19399, 469818, 15076198, 396737074}},  // 755
    {"rb1k1nrq/pbp1pppp/1p1p1n2/8/5P2/4NN1P/PPPPP1P1/RBBK2RQ w GAga - 2 9",
     {28, 841, 24056, 710751, 20772996, 613798447}},  // 756
    {"rkbbnnrq/p1pp3p/4p1p1/1p3p2/P6P/1P6/1BPPPPP1/RK1BNNRQ w GAga - 0 9",
     {33, 957, 30668, 907217, 29735654, 903933626}},  // 757
    {"rk2nbrq/p1ppppp1/bpn5/7p/6P1/2N2P2/PPPPP1QP/RKB1NBR1 w GAga - 2 9",
     {24, 687, 18206, 544627, 15518417, 484217179}},  // 758
    {"rkbn1r1b/pp1pppnp/6q1/2p3p1/5P1P/4N3/PPPPP1P1/RKB1NRQB w FAfa - 1 9",
     {23, 831, 21254, 754622, 21126103, 744755212}},  // 759
    {"rbknb1rq/ppp1p1p1/3pnp1p/8/6PP/2PP4/PP2PP2/RBKNBNRQ w GAga - 0 9",
     {31, 838, 26800, 736910, 24008129, 677776408}},  // 760
    {"rknbb1rq/p1pn1ppp/4p3/1p1p4/2P5/1P2N1P1/P2PPP1P/RKNBB1RQ w GAga - 1 9",
     {29, 830, 24798, 721630, 22243832, 660040360}},  // 761
    {"rk1nbbrq/pp1p1ppp/3n4/P3p3/2p4P/8/1PPPPPP1/RKNNBBRQ w GAga - 1 9",
     {24, 484, 12776, 297419, 8379748, 214004367}},  // 762
    {"rknnbr1b/ppp2pqp/3p4/4p1p1/7P/3P1P2/PPP1P1P1/RKNNBRQB w FAfa - 0 9",
     {32, 838, 26408, 740701, 23472124, 699211365}},  // 763
    {"rb1k1rbq/ppppN1pp/2nn4/5p2/7P/8/PPPPPPP1/RBK1NRBQ w FA - 1 9",
     {27, 800, 22785, 701742, 20804424, 660917073}},  // 764
    {"r1nbnrbq/kppppp1p/6p1/8/p1PP1P2/4P3/PP4PP/RKNBNRBQ w FA - 1 9",
     {28, 757, 21198, 602699, 17180857, 507618340}},  // 765
    {"rkn1rbbq/p1pppppp/2n5/1pP5/8/1N2P3/PP1P1PPP/RK1NRBBQ w EAea - 1 9",
     {22, 483, 11890, 283679, 7497674, 191130942}},  // 766
    {"rknnrqbb/2pppppp/8/p7/Np3P2/3P4/PPP1P1PP/RKN1RQBB w EAea - 0 9",
     {25, 536, 14456, 339180, 9694947, 245669668}},  // 767
    {"bb1rknrn/1qppppp1/1p4B1/p6N/8/2P5/PP1PPPPP/B1QRK1RN w GDgd - 1 9",
     {32, 715, 22421, 575008, 17860156, 502410909}},  // 768
    {"b1rbknrn/qpp1ppp1/p6p/3p4/2P5/1P1P1P2/P3P1PP/BQRBKNRN w GCgc - 0 9",
     {30, 818, 24421, 688711, 20981488, 611986786}},  // 769
    {"bqkrnbrn/1pp1pp1p/p7/1B1p2p1/4P3/7P/PPPP1PP1/BQKRN1RN w - - 0 9",
     {28, 676, 18366, 478054, 13126287, 363765666}},  // 770
    {"bqrknrnb/1p2ppp1/p1pp3p/8/3P1P2/1PP5/P3P1PP/BQRKNRNB w FCfc - 0 9",
     {31, 646, 20686, 455607, 14984618, 349082278}},  // 771
    {"qbbrkn1r/pppppp1p/8/6p1/2P1Pn1P/6N1/PP1P1PP1/QBBRKNR1 w GDd - 3 9",
     {20, 532, 11581, 303586, 7512432, 202967948}},  // 772
    {"1rbbknr1/p1ppp1pp/1pq2pn1/8/3P4/P3P3/QPP2PPP/1RBBKNRN w GBgb - 3 9",
     {31, 1002, 30581, 999607, 30642468, 1009228283}},  // 773
    {"qrbkn1rn/pppp1ppp/8/6b1/P1P1Pp2/8/1P1P2PP/QRBKNBRN w GBgb - 0 9",
     {22, 505, 12447, 304863, 8192621, 214730959}},  // 774
    {"qrbk1rnb/p2ppp1p/5n2/1pp3p1/8/7P/PPPPPPPN/QRBKR1NB w Bfb - 0 9",
     {20, 619, 13448, 449630, 10571176, 369603424}},  // 775
    {"qbrkb1r1/ppp2ppp/3pn1n1/P3p3/4P3/3P4/1PP2PPP/QBRKBNRN w GCgc - 1 9",
     {26, 755, 20596, 604483, 17164382, 510878835}},  // 776
    {"qrkbb1r1/ppp1pnpp/3p2n1/5p2/1P3P2/2Q3N1/P1PPP1PP/1RKBB1RN w GBgb - 0 9",
     {35, 918, 32244, 870888, 30933394, 867833733}},  // 777
    {"qrknbbrn/ppp1ppp1/8/7p/2Bp4/4PPP1/PPPP3P/QRKNB1RN w GBgb - 0 9",
     {27, 593, 16168, 376808, 10422676, 258348640}},  // 778
    {"qrk1brnb/ppppp3/4n2p/5pp1/2PP4/2N4P/PP2PPP1/QRK1BRNB w FBfb - 2 9",
     {24, 672, 17447, 506189, 13765777, 414930519}},  // 779
    {"qbrknrb1/p2ppppp/2p3n1/8/p4P2/6PP/1PPPP3/QBRKNRBN w FCfc - 0 9",
     {29, 759, 23235, 634493, 20416668, 584870558}},  // 780
    {"1rkb1rbn/p1pp1ppp/3np3/1p6/4qP2/3NB3/PPPPPRPP/QRKB3N w Bfb - 0 9",
     {22, 923, 22585, 914106, 24049880, 957218571}},  // 781
    {"1rknrbbn/p1pp1p1p/8/1p2p1p1/4qPP1/2P5/PP1PP1BP/QRKNR1BN w EBeb - 0 9",
     {28, 1309, 36355, 1568968, 44576409, 1846382333}},  // 782
    {"qrk1rn1b/ppppp2p/4n3/3b1pp1/4P2P/5BP1/PPPP1P2/QRKNRNB1 w EBeb - 3 9",
     {26, 839, 22189, 726354, 19978260, 661207281}},  // 783
    {"bbrqk1rn/pp1ppppp/8/2p5/2P1P3/5n1P/PPBP1PP1/B1RQKNRN w GCgc - 1 9",
     {3, 95, 2690, 85038, 2518864, 80775549}},  // 784
    {"brqbk2n/pppppprp/8/6p1/1P3n2/5P2/P1PPP1PP/R1QBKNRN w Gb - 2 9",
     {22, 593, 13255, 362760, 8922397, 253271592}},  // 785
    {"brqknbr1/pp3ppp/3p2n1/2p1p3/2P5/5P2/PPKPP1PP/BRQ1NBRN w gb - 0 9",
     {21, 590, 13190, 397355, 9581695, 304103516}},  // 786
    {"1rqknrnb/2pp1ppp/p3p3/1p6/P2P4/5bP1/1PP1PP1P/BRQKNRNB w FBfb - 0 9",
     {24, 737, 20052, 598439, 17948681, 536330341}},  // 787
    {"rbb1k1rn/p1pqpppp/6n1/1p1p4/5P2/3PP3/PPP1K1PP/RBBQ1NRN w ga - 3 9",
     {24, 694, 16773, 513782, 13094823, 419402704}},  // 788
    {"rqbbknr1/1ppp2pp/p5n1/4pp2/P7/1PP5/1Q1PPPPP/R1BBKNRN w GAga - 0 9",
     {24, 600, 15347, 408207, 11029596, 308553169}},  // 789
    {"rqbknbrn/2pppppp/6Q1/pp6/8/2P5/PP1PPPPP/R1BKNBRN w GAga - 2 9",
     {40, 949, 34100, 889887, 31296485, 881529007}},  // 790
    {"rqbknr1b/pp1ppp2/2p2n1p/6p1/8/3P1PPP/PPP1P3/RQBKNRNB w FAfa - 0 9",
     {20, 560, 12275, 373921, 8687544, 277906201}},  // 791
    {"rbqkbnrn/p3pppp/1p6/3p4/P1p3P1/1P6/1QPPPP1P/RB1KBNRN w GAga - 0 9",
     {30, 1155, 35865, 1351455, 43092716, 1614019629}},  // 792
    {"rqkbb1rn/p1p1pppn/1p1p4/7p/4PP2/7P/PPPPB1P1/RQK1BNRN w GAga - 1 9",
     {30, 701, 20804, 515942, 15450970, 401499189}},  // 793
    {"rqknbbrn/1p2pp1p/3p2p1/p1p5/P2P4/1P6/1KP1PPPP/RQ1NBBRN w ga - 0 9",
     {28, 756, 21655, 610320, 17989811, 525585996}},  // 794
    {"rqknbrnb/1pp3pp/5p2/p2pp3/P7/3PPN2/1PP2PPP/RQKNBR1B w FAfa - 0 9",
     {26, 731, 19509, 550395, 15209404, 439767476}},  // 795
    {"rbqkr1bn/p1pppp1p/1p1n4/6p1/7P/3P1PP1/PPP1P3/RBQKNRBN w FAa - 0 9",
     {27, 586, 16282, 381604, 10905865, 274364342}},  // 796
    {"rqk1nrb1/ppbp1ppp/4p1n1/2p5/7P/1PP5/P2PPPP1/RQKBNRBN w FAfa - 1 9",
     {27, 749, 21480, 602318, 18084787, 520547029}},  // 797
    {"rqknrbbn/pp1p1ppp/4p3/2p5/3P2P1/7P/PPP1PP2/RQKNRBBN w EAa - 0 9",
     {20, 533, 11829, 336248, 8230417, 245871540}},  // 798
    {"rqknrnbb/pp1ppp1p/2p3p1/8/8/1P2P1NP/P1PP1PP1/RQKNR1BB w EAea - 0 9",
     {22, 633, 14480, 441877, 10827868, 343525739}},  // 799
    {"1brkq1rn/2pppppp/1p2n3/p2bN3/8/7P/PPPPPPP1/BBRKQ1RN w GCgc - 2 9",
     {27, 748, 20134, 580054, 16010135, 475206624}},  // 800
    {"brkbqnrn/2pp1ppp/8/1p2p3/Pp2N3/8/2PPPPPP/BRKBQNR1 w GBgb - 0 9",
     {30, 827, 25308, 757837, 23746165, 751690068}},  // 801
    {"brk1nbrn/pp1ppppp/2p5/7P/5P2/q2P4/PPP1P1P1/BRKQNBRN w GBgb - 1 9",
     {15, 471, 8716, 276424, 5960901, 190316951}},  // 802
    {"brkqnrnb/1p1pp1p1/p4p2/2p4p/8/P2PP3/1PP1QPPP/BRK1NRNB w FBfb - 0 9",
     {24, 479, 12584, 280081, 7830230, 190419716}},  // 803
    {"rbbkqnrn/2ppp2p/pp3p2/6p1/P6P/8/RPPPPPP1/1BBKQNRN w Gga - 0 9",
     {21, 523, 12125, 328733, 8322614, 242240658}},  // 804
    {"rkbbqr1n/1ppppppn/7p/p7/4P3/2P2P2/PP1PB1PP/RKB1QNRN w GAa - 3 9",
     {27, 563, 16026, 372148, 11105151, 283211800}},  // 805
    {"rkbqnbrn/ppppp3/8/5ppp/2P3P1/7P/PPQPPP2/RKB1NBRN w GAga - 0 9",
     {28, 639, 19250, 469250, 14872172, 384663405}},  // 806
    {"rkb1nrnb/pppp1pp1/5q1p/8/P3p3/4R1P1/1PPPPP1P/1KBQNRNB w Ffa - 0 9",
     {28, 873, 23690, 720814, 20209424, 625281937}},  // 807
    {"rbkqb1rn/1p1ppppp/4n3/p1p5/8/3PBP2/PPP1P1PP/RBKQ1NRN w GAga - 0 9",
     {26, 798, 21416, 667496, 18475618, 591681956}},  // 808
    {"rk1qbnrn/1p1ppppp/1b6/p1p5/P7/2P3NP/1P1PPPP1/RKQBB1RN w GAga - 0 9",
     {22, 506, 12313, 301029, 7891676, 205739580}},  // 809
    {"rk1nbbrn/ppp1ppp1/8/3p3p/1P1P2q1/5PB1/P1P1P1PP/RKQN1BRN w GAga - 1 9",
     {31, 956, 29219, 903799, 27827461, 876341492}},  // 810
    {"rkqnbr1b/pp1pppp1/7p/2p2n2/P2P4/7N/RPP1PPPP/1KQNBR1B w Ffa - 0 9",
     {31, 750, 24267, 646252, 21639104, 617064197}},  // 811
    {"rbkq1rbn/2p1pppp/pp3n2/3p4/5P2/3N2N1/PPPPP1PP/RBKQR1B1 w Afa - 2 9",
     {26, 647, 18027, 465119, 13643783, 369702807}},  // 812
    {"rkqbr1bn/p2ppppp/1pp2n2/8/5P2/3P1N2/PPP1PRPP/RKQB2BN w Aa - 3 9",
     {24, 574, 14593, 371597, 10066892, 271121237}},  // 813
    {"rk1qrbbn/p1ppp1pp/1p2n3/5p2/1P6/K3N3/P1PPPPPP/R1Q1RBBN w ea - 0 9",
     {25, 548, 14069, 340734, 9043111, 235545764}},  // 814
    {"rkqnrnbb/pp1pp3/2p5/5ppp/8/PP4NP/2PPPPP1/RKQNR1BB w EAea - 0 9",
     {23, 727, 18228, 566572, 15078056, 471296844}},  // 815
    {"bbrknq1r/ppppppp1/8/7p/5n2/3P4/PPP1PNPP/BBKRNQR1 w c - 0 9",
     {21, 610, 13300, 394705, 9605845, 293532398}},  // 816
    {"brkbnqr1/2pppnpp/pp3p2/8/4PPPP/8/PPPP4/BRKBNQRN w GBgb - 1 9",
     {30, 757, 23908, 621332, 20360394, 548380577}},  // 817
    {"brk1qb1n/ppppppr1/2n3pp/8/2P3P1/2N5/PP1PPP1P/BR1KQBRN w b - 1 9",
     {26, 570, 15537, 352883, 10081351, 242864559}},  // 818
    {"brknq1nb/pp2prpp/8/2pP1p2/6P1/2N5/PPPP1P1P/BRK1QRNB w FBb - 1 9",
     {33, 830, 27897, 764915, 26262884, 765831403}},  // 819
    {"rbbk1qrn/ppp1p1pp/5p2/3p1n2/7N/P7/1PPPPPPP/RBB1KQRN w ga - 0 9",
     {21, 562, 13060, 378883, 9520963, 290579255}},  // 820
    {"rk1b1qrn/ppp1pppp/5n2/3pN3/P6P/7b/1PPPPPP1/RKBB1QRN w GAga - 4 9",
     {28, 677, 19235, 488740, 14354779, 383207197}},  // 821
    {"rkbnqbrn/pp1ppp1p/2p5/6p1/P7/4P3/KPPPQPPP/R1BN1BRN w - - 3 9",
     {28, 585, 17443, 401483, 12574541, 310495538}},  // 822
    {"rk1nqrnb/pbpppp2/1p4p1/7p/P7/5NP1/1PPPPPBP/RKBNQR2 w FAfa - 2 9",
     {26, 774, 21626, 645200, 19093408, 576325868}},  // 823
    {"rbknb1rn/p1pp2pp/1p6/4pp2/1q3P1B/2N5/PPPPPNPP/RBK2QR1 w GAga - 2 9",
     {31, 1206, 36940, 1374158, 42849564, 1555711209}},  // 824
    {"rk1bbqrn/pp1pp1pp/3n4/5p2/3p4/1PP5/PK2PPPP/R1NBBQRN w ga - 0 9",
     {21, 629, 14059, 429667, 10587910, 332632033}},  // 825
    {"rknqbbr1/p1pp1pp1/1p4n1/4p2p/4P1P1/6RB/PPPP1P1P/RKNQB2N w Aga - 0 9",
     {27, 753, 20918, 593155, 17318772, 507563675}},  // 826
    {"rknqbr1b/pppp1ppp/4p2n/8/1P3P2/4P3/P1PPN1PP/RKNQBR1B w FAfa - 2 9",
     {26, 623, 17177, 460663, 13389799, 383508368}},  // 827
    {"r2kqrbn/bppppppp/2n5/p4B2/5P2/2P5/PP1PP1PP/1RKNQRBN w F - 2 9",
     {39, 1026, 37800, 1011922, 35946987, 992756232}},  // 828
    {"rk1bqrb1/ppppppp1/1n6/7p/2P2P1n/4P1Q1/PP1P2PP/RKNB1RBN w FAfa - 0 9",
     {35, 760, 25817, 610557, 21014787, 536852043}},  // 829
    {"rkq1rb1n/ppppp1pp/1n6/5p2/PPb2P2/8/1KPPP1PP/R1NQRBBN w ea - 1 9",
     {27, 754, 21009, 568788, 16461795, 448313956}},  // 830
    {"rknqr2b/pppnp1pp/3p4/3b1p2/8/1N1P2N1/PPP1PPPP/RKQ1R1BB w EAea - 1 9",
     {27, 803, 23708, 700453, 21875031, 654754840}},  // 831
    {"bbrknrqn/ppppp1pB/8/2P2p1p/8/5N2/PP1PPPPP/B1RK1RQN w FCfc - 0 9",
     {30, 799, 23923, 671112, 20532790, 603059376}},  // 832
    {"brkbnrq1/1pppp1p1/6np/p4p2/4P3/1PP5/P1KP1PPP/BR1BNRQN w fb - 1 9",
     {27, 726, 19329, 555622, 15156662, 457601127}},  // 833
    {"brknrbq1/1p1p1ppp/p3p1n1/2p5/8/1P1BPP2/P1PP2PP/BRKNR1QN w EBeb - 0 9",
     {36, 786, 27868, 655019, 22852433, 577223409}},  // 834
    {"brknrqnb/p2ppp1p/2p5/1p6/3P2p1/P1P1N3/1P2PPPP/BRK1RQNB w EBeb - 0 9",
     {23, 649, 15169, 440504, 10687843, 320881984}},  // 835
    {"rbbk1rqn/1ppppppp/3n4/p7/2P5/3N4/PP1PPPPP/RBB1KRQN w fa - 1 9",
     {20, 478, 11094, 275250, 7094988, 185488058}},  // 836
    {"rkbbnrqn/p2p1ppp/1p2p3/8/P1p1P3/1BP5/1P1P1PPP/RKB1NRQN w FAfa - 0 9",
     {22, 570, 13295, 346811, 8671852, 229898448}},  // 837
    {"rkb1rb1n/ppppppqp/8/2n3p1/2P1P1P1/8/PP1P1P1P/RKBNRBQN w EAea - 1 9",
     {23, 663, 16212, 490748, 12900485, 404944553}},  // 838
    {"rkb1rqnb/pppp3p/2n3p1/4pp2/P2P3P/2P5/1P2PPP1/RKBNRQNB w EAea - 0 9",
     {25, 845, 22188, 741972, 20276176, 683290790}},  // 839
    {"rbk1brqn/ppp1pppp/8/3p4/7P/1P4P1/2PPPP2/RBKNBRQN w FAfa - 0 9",
     {24, 526, 13862, 322175, 9054028, 222704171}},  // 840
    {"rknbbrqn/pp3pp1/4p3/2pp3p/2P5/8/PPBPPPPP/RKN1BRQN w FAfa - 0 9",
     {26, 756, 19280, 559186, 14697705, 433719427}},  // 841
    {"1knrbbqn/rp1p1ppp/p3p3/2p5/8/5P1P/PPPPP1P1/RKNRBBQN w DAd - 0 9",
     {26, 539, 15194, 345070, 10223443, 248715580}},  // 842
    {"rknr1qnb/ppp1p1pp/3p2b1/8/4p3/1P3P1P/P1PP2P1/RKNRBQNB w DAda - 0 9",
     {25, 701, 18969, 561369, 16047041, 496340789}},  // 843
    {"rbk1r1bn/ppppp1pp/4n3/5p2/1P3P2/4N2P/PqPPP1P1/RBK1RQBN w EAea - 1 9",
     {2, 60, 1319, 41765, 1017864, 33183408}},  // 844
    {"r1nbrqbn/k1ppp1pp/1p6/p4p2/2P5/6PQ/PP1PPP1P/RKNBR1BN w EA - 0 9",
     {27, 699, 20436, 561765, 17192121, 499247248}},  // 845
    {"rknrqbbn/1pp1pp2/p5p1/3p3p/6P1/PN5P/1PPPPP2/RK1RQBBN w DAda - 0 9",
     {23, 611, 15515, 435927, 11917036, 352885930}},  // 846
    {"rknrqn1b/p1pp1ppb/8/1p2p1Qp/3P4/3N4/PPP1PPPP/RK1R1NBB w DAda - 0 9",
     {45, 1170, 48283, 1320341, 52213677, 1500007485}},  // 847
    {"bbkrnrnq/p2p1ppp/2p1p3/1p6/1P2Q3/6P1/P1PPPP1P/BBKRNRN1 w - - 0 9",
     {41, 1035, 39895, 1035610, 38555608, 1037686769}},  // 848
    {"brkbnr2/1ppppp1p/7n/p5N1/P2q4/8/1PPPPPPP/BRKBNRQ1 w FBfb - 1 9",
     {22, 869, 19234, 679754, 16453359, 567287944}},  // 849
    {"brknrbnq/p1ppppp1/1p6/7p/2PP4/5P2/PPK1P1PP/BR1NRBNQ w eb - 1 9",
     {23, 641, 14748, 422240, 10192718, 302864305}},  // 850
    {"brk1r1qb/pp1ppnpp/2p2pn1/8/6N1/2N3P1/PPPPPP1P/BRK1R1QB w EBeb - 3 9",
     {32, 863, 28379, 773191, 25848794, 720443112}},  // 851
    {"rbbk1rnq/pppp1pp1/4p2p/8/3P2n1/4BN1P/PPP1PPP1/RB1K1RNQ w FAfa - 3 9",
     {26, 628, 16151, 411995, 11237919, 300314373}},  // 852
    {"rkbbnr1q/p1pppppp/5n2/1p5B/PP6/4P3/2PP1PPP/RKB1NRNQ w FAfa - 0 9",
     {30, 692, 21036, 519283, 16025428, 420887328}},  // 853
    {"rkb1rbnq/1pppp1pp/5p2/p7/5n1P/1PN3P1/P1PPPP2/RKB1RBNQ w EAea - 0 9",
     {32, 825, 27130, 697251, 23593363, 622249676}},  // 854
    {"rkbnrnqb/1ppp1p1p/p5p1/4p3/4P3/2N2P2/PPPP2PP/RKBR1NQB w Aea - 0 9",
     {24, 487, 13300, 301989, 8782713, 215787079}},  // 855
    {"rbknbr1q/pppp2pp/4p3/5p1n/1P2P2N/8/P1PP1PPP/RBKNBR1Q w FAfa - 0 9",
     {23, 571, 13799, 365272, 9224232, 257288920}},  // 856
    {"rknbb1nq/pppppr2/5pp1/7p/8/1N4P1/PPPPPP1P/RK1BBRNQ w FAa - 2 9",
     {26, 548, 15618, 350173, 10587626, 253006082}},  // 857
    {"rknr1bnq/p2pp1pp/1p3p2/2p4b/6PP/2P2N2/PP1PPP2/RKNRBB1Q w DAda - 1 9",
     {25, 502, 13150, 279098, 7824941, 175766730}},  // 858
    {"rknrb1qb/ppp1pppp/3p4/8/4P1nP/2P5/PPKP1PP1/R1NRBNQB w da - 1 9",
     {23, 643, 14849, 426616, 10507328, 312096061}},  // 859
    {"rbk1rnbq/pppp1npp/4p3/5p2/4P1P1/7P/PPPP1P1N/RBKNR1BQ w EAea - 1 9",
     {24, 591, 15178, 376988, 10251465, 263574861}},  // 860
    {"rknbrnb1/p1pppp1p/1p6/3N2p1/P3q1P1/8/1PPPPP1P/RKNBR1BQ w EAea - 1 9",
     {28, 948, 27343, 864588, 26241141, 812343987}},  // 861
    {"rknrn1b1/ppppppqp/8/6p1/2P5/2P1BP2/PP2P1PP/RKNRNB1Q w DAda - 1 9",
     {31, 807, 24360, 672973, 20455205, 588518645}},  // 862
    {"1k1rnqbb/npppppp1/r7/p2B3p/5P2/1N4P1/PPPPP2P/RK1RNQB1 w DAd - 0 9",
     {40, 1122, 44297, 1249989, 48711073, 1412437357}},  // 863
    {"bbqr1rkn/pp1ppppp/8/2p5/1P2P1n1/7N/P1PP1P1P/BBQRKR1N w FD - 0 9",
     {26, 841, 22986, 746711, 21328001, 705170410}},  // 864
    {"bqkr1rnn/1ppp1ppp/p4b2/4p3/P7/3PP2N/1PP2PPP/BQRBKR1N w FC - 3 9",
     {24, 500, 12802, 293824, 7928916, 197806842}},  // 865
    {"bqrkrbnn/1pp1ppp1/8/p6p/3p4/P3P2P/QPPP1PP1/B1RKRBNN w ECec - 0 9",
     {31, 592, 18585, 396423, 12607528, 298629240}},  // 866
    {"bqkrrnnb/2p1pppp/p7/1P1p4/8/2R3P1/PP1PPP1P/BQ1KRNNB w E - 0 9",
     {42, 1124, 45187, 1276664, 50052573, 1483524894}},  // 867
    {"qbbrkrn1/p1pppn1p/8/1p3Pp1/2P5/8/PP1PPP1P/QBBRKRNN w FDfd - 0 9",
     {21, 577, 13244, 392131, 9683808, 300294295}},  // 868
    {"qrbbkrnn/pp1p2pp/4p3/5p2/2p2P1P/2P5/PP1PP1P1/QRBBKRNN w FBfb - 0 9",
     {21, 571, 12736, 345681, 8239872, 228837930}},  // 869
    {"qrbkrbn1/1pp1pppp/p2p4/8/5PPn/2P5/PP1PP3/QRBKRBNN w EBeb - 0 9",
     {18, 466, 9443, 257776, 5679073, 162883949}},  // 870
    {"qrb1rnnb/pp1p1ppp/2pk4/4p3/1P2P3/1R6/P1PP1PPP/Q1BKRNNB w E - 4 9",
     {37, 760, 26863, 562201, 19486022, 421740856}},  // 871
    {"qbrkbrn1/p1pppp1p/6n1/1p4p1/1P6/5P2/P1PPPBPP/QBRK1RNN w FCfc - 1 9",
     {33, 824, 27385, 750924, 25176664, 734656217}},  // 872
    {"qrkbbr2/2pppppp/5nn1/pp1Q4/P7/3P4/1PP1PPPP/1RKBBRNN w FBfb - 0 9",
     {42, 1147, 44012, 1311247, 48216013, 1522548864}},  // 873
    {"qrkrbbnn/pp2pp2/2pp2pp/1B6/P7/4P3/1PPP1PPP/QRKRB1NN w DBdb - 0 9",
     {26, 464, 12653, 242892, 6928220, 142507795}},  // 874
    {"qrkrbnnb/p1pp1pp1/1p5p/4p3/1P6/6PN/PKPPPP1P/QR1RBN1B w db - 0 9",
     {29, 705, 20000, 529810, 15055365, 419552571}},  // 875
    {"qbrkr1bn/p1p1pp1p/1p1p2n1/6p1/3P1P2/4P3/PPP3PP/QBKRRNBN w ec - 2 9",
     {23, 613, 14835, 426484, 10747407, 323905533}},  // 876
    {"qrk1rnb1/p1pp1ppp/1p2Bbn1/8/4P3/6P1/PPPP1P1P/QRK1RNBN w EBeb - 1 9",
     {28, 927, 24887, 846839, 23063284, 807913585}},  // 877
    {"1qkrnbbn/1rpppppp/pp6/5N2/P4P2/8/1PPPP1PP/QRKRNBB1 w DBd - 3 9",
     {30, 542, 16646, 345172, 10976745, 251694423}},  // 878
    {"qrkr2bb/pppppppp/8/1n2n3/1N5P/1P6/P1PPPPP1/QRKR1NBB w DBdb - 1 9",
     {28, 719, 21048, 562015, 17351761, 479400272}},  // 879
    {"bbrqkrnn/3ppppp/8/ppp5/6P1/4P2N/PPPPKP1P/BBRQ1R1N w fc - 0 9",
     {21, 704, 16119, 546215, 13676371, 470796854}},  // 880
    {"brqbkrnn/1pp2p1p/3pp1p1/p5N1/8/1P6/P1PPPPPP/BRQBK1RN w Bfb - 0 9",
     {34, 688, 22827, 505618, 16639723, 402140795}},  // 881
    {"br1krb1n/2qppppp/pp3n2/8/1P4P1/8/P1PPPP1P/1RQKRBNN w EBeb - 0 9",
     {24, 945, 23943, 926427, 25019636, 959651619}},  // 882
    {"brqkr1nb/2ppp1pp/1p2np2/p7/2P1PN2/8/PP1P1PPP/BRQKRN1B w EBeb - 0 9",
     {28, 675, 19728, 504128, 15516491, 417396563}},  // 883
    {"rbbqkrnn/3pppp1/p7/1pp4p/2P1P2P/8/PP1P1PP1/RBBQKRNN w FAfa - 0 9",
     {26, 671, 18164, 496806, 14072641, 404960259}},  // 884
    {"rqbbkr1n/pp1p1p1p/4pn2/2p3p1/4P1P1/3P3P/PPP2P2/RQBBKRNN w FAfa - 0 9",
     {22, 633, 14629, 441809, 10776416, 335689685}},  // 885
    {"rqbkrbnn/p1ppp3/1p3pp1/7p/3P4/P1P5/1PQ1PPPP/R1BKRBNN w EAea - 0 9",
     {32, 607, 20339, 454319, 15586203, 383515709}},  // 886
    {"rqbkrnn1/pp2ppbp/3p4/2p3p1/2P5/1P3N1P/P2PPPP1/RQBKRN1B w EAea - 1 9",
     {29, 943, 28732, 908740, 28761841, 907579129}},  // 887
    {"rbqkb1nn/1ppppr1p/p5p1/5p2/1P6/2P4P/P1KPPPP1/RBQ1BRNN w a - 1 9",
     {22, 441, 10403, 231273, 5784206, 140934555}},  // 888
    {"rqkb1rnn/1pp1pp1p/p5p1/1b1p4/3P4/P5P1/RPP1PP1P/1QKBBRNN w Ffa - 1 9",
     {21, 505, 11592, 290897, 7147063, 188559137}},  // 889
    {"rq1rbbnn/pkp1ppp1/3p3p/1p2N1P1/8/8/PPPPPP1P/RQKRBB1N w DA - 0 9",
     {27, 608, 16419, 387751, 10808908, 268393274}},  // 890
    {"rqkrb2b/p2ppppp/2p3nn/1p6/5P2/PP1P4/2P1P1PP/RQKRBNNB w DAda - 1 9",
     {30, 749, 21563, 581531, 16916813, 485406712}},  // 891
    {"rbqkr1bn/pp1ppp2/2p1n2p/6p1/8/4BPNP/PPPPP1P1/RBQKRN2 w EAea - 0 9",
     {23, 600, 15082, 410057, 11041820, 314327867}},  // 892
    {"rqkbrnb1/2ppp1pp/pp3pn1/8/5P2/B2P4/PPP1P1PP/RQKBRN1N w EAea - 2 9",
     {22, 569, 13541, 371471, 9395816, 269460607}},  // 893
    {"rqkrnbb1/p1p1pppp/1p4n1/3p4/7P/P3P3/1PPPBPP1/RQKRN1BN w DAda - 0 9",
     {27, 579, 15565, 373079, 10238486, 266047417}},  // 894
    {"rqkrn1bb/p1ppp1pp/4n3/1p6/6p1/4N3/PPPPPPPP/RQKR2BB w DAda - 0 9",
     {20, 462, 10234, 274162, 6563859, 193376359}},  // 895
    {"bbrkqr2/pppp1ppp/6nn/8/2P1p3/3PP2N/PP3PPP/BBRKQR1N w FCfc - 0 9",
     {28, 724, 21688, 619064, 19318355, 593204629}},  // 896
    {"brk1qrnn/1pppbppp/4p3/8/1p6/P1P4P/3PPPP1/BRKBQRNN w FBfb - 1 9",
     {24, 662, 16920, 468215, 12610387, 355969349}},  // 897
    {"1r1qrbnn/p1pkpppp/1p1p4/8/3P1PP1/P4b2/1PP1P2P/BRKQRBNN w EB - 1 9",
     {22, 696, 17021, 510247, 13697382, 401903030}},  // 898
    {"1rkqrnnb/p1p1p1pp/1p1p4/3b1p1N/4P3/5N2/PPPP1PPP/BRKQR2B w EBeb - 1 9",
     {29, 887, 27035, 816176, 26051242, 791718847}},  // 899
    {"rbbkq1rn/pppppppp/7n/8/P7/3P3P/1PPKPPP1/RBB1QRNN w a - 3 9",
     {22, 417, 9900, 216855, 5505063, 134818483}},  // 900
    {"rkbbqr1n/1p1pppp1/2p2n2/p4NBp/8/3P4/PPP1PPPP/RK1BQRN1 w FAfa - 0 9",
     {37, 832, 30533, 728154, 26676373, 673756141}},  // 901
    {"rkbqrb1n/3pBppp/ppp2n2/8/8/P2P4/1PP1PPPP/RK1QRBNN w EAea - 0 9",
     {28, 685, 19718, 543069, 16033316, 482288814}},  // 902
    {"rkb1rn1b/ppppqppp/4p3/8/1P2n1P1/5Q2/P1PP1P1P/RKB1RNNB w EAea - 2 9",
     {37, 1158, 40114, 1234768, 44672979, 1389312729}},  // 903
    {"r1kqbrnn/pp1pp1p1/7p/2P2p2/5b2/3P4/P1P1P1PP/RBKQBRNN w FAfa - 0 9",
     {5, 161, 4745, 154885, 4734999, 157499039}},  // 904
    {"rkqbbr1n/ppp1ppp1/8/Q2p3p/4n3/3P1P2/PPP1P1PP/RK1BBRNN w FAfa - 2 9",
     {38, 1144, 40433, 1236877, 43832975, 1366087771}},  // 905
    {"rkqrbbn1/p1ppppp1/Bp5p/8/P6n/2P1P3/1P1P1PPP/RKQRB1NN w DAda - 0 9",
     {28, 551, 15488, 350861, 9944107, 251179183}},  // 906
    {"rkqrb1nb/1ppp1ppp/p7/4p3/5n2/3P2N1/PPPQPPPP/RK1RB1NB w DAda - 0 9",
     {26, 690, 19877, 513628, 15965907, 418191735}},  // 907
    {"rbkqrnbn/pppp1p2/4p1p1/7p/7P/P2P4/BPP1PPP1/R1KQRNBN w EAea - 0 9",
     {27, 515, 13992, 309727, 8792550, 218658292}},  // 908
    {"rkqbrnbn/pp1ppp2/8/2p3p1/P1P4p/5P2/1PKPP1PP/R1QBRNBN w ea - 0 9",
     {27, 627, 16843, 431101, 11978698, 328434174}},  // 909
    {"rkqrnbbn/1p2pp1p/3p2p1/p1p5/P5PP/3N4/1PPPPP2/RKQR1BBN w DAda - 0 9",
     {23, 624, 15512, 451860, 11960861, 367311176}},  // 910
    {"rk2rnbb/ppqppppp/2pn4/8/1P3P2/6P1/P1PPP1NP/RKQR1NBB w DAa - 1 9",
     {27, 727, 20206, 581003, 16633696, 505212747}},  // 911
    {"b1krrqnn/pp1ppp1p/2p3p1/8/P3Pb1P/1P6/2PP1PP1/BBRKRQNN w EC - 0 9",
     {32, 943, 30759, 865229, 28672582, 800922511}},  // 912
    {"1rkbrqnn/p1pp1ppp/1p6/8/P2Pp3/8/1PPKPPQP/BR1BR1NN w eb - 0 9",
     {28, 916, 24892, 817624, 22840279, 759318058}},  // 913
    {"brkrqb1n/1pppp1pp/p7/3n1p2/P5P1/3PP3/1PP2P1P/BRKRQBNN w DBdb - 0 9",
     {27, 669, 18682, 484259, 13956472, 380267099}},  // 914
    {"brkrqnnb/3pppp1/1p6/p1p4p/2P3P1/6N1/PP1PPP1P/BRKRQ1NB w DBdb - 0 9",
     {29, 699, 20042, 512639, 15093909, 406594531}},  // 915
    {"r1bkrq1n/pp2pppp/3b1n2/2pp2B1/6P1/3P1P2/PPP1P2P/RB1KRQNN w EAea - 2 9",
     {27, 835, 22848, 713550, 19867800, 631209313}},  // 916
    {"rk1brq1n/p1p1pppp/3p1n2/1p3b2/4P3/2NQ4/PPPP1PPP/RKBBR2N w EAea - 4 9",
     {36, 1004, 35774, 979608, 35143142, 966310885}},  // 917
    {"rkbrqbnn/1p2ppp1/B1p5/p2p3p/4P2P/8/PPPP1PP1/RKBRQ1NN w DAda - 0 9",
     {27, 748, 21005, 597819, 17597073, 515304215}},  // 918
    {"rkbrqn1b/pp1pp1pp/2p2p2/5n2/8/2P2P2/PP1PP1PP/RKBRQ1NB w DAda - 0 9",
     {20, 479, 10485, 266446, 6253775, 167767913}},  // 919
    {"rbkrbnn1/ppppp1pp/5q2/5p2/5P2/P3P2N/1PPP2PP/RBKRBQ1N w DAda - 3 9",
     {28, 947, 26900, 876068, 26007841, 838704143}},  // 920
    {"rkr1bqnn/1ppp1p1p/p5p1/4p3/3PP2b/2P2P2/PP4PP/RKRBBQNN w CAca - 0 9",
     {31, 1004, 32006, 1006830, 32688124, 1024529879}},  // 921
    {"rkrqbbnn/pppp3p/8/4ppp1/1PP4P/8/P2PPPP1/RKRQBBNN w CAca - 0 9",
     {24, 717, 18834, 564137, 15844525, 484884485}},  // 922
    {"rkrqbn1b/pppp2pp/8/4pp2/1P1P2n1/5N2/P1P1PP1P/RKRQBN1B w CAca - 0 9",
     {25, 718, 19654, 587666, 17257753, 537354146}},  // 923
    {"rbkrqnbn/p1p1ppp1/1p1p4/8/3PP2p/2PB4/PP3PPP/R1KRQNBN w DAda - 0 9",
     {30, 754, 23298, 611322, 19338246, 532603566}},  // 924
    {"1krbqnbn/1p2pppp/r1pp4/p7/8/1P1P2PP/P1P1PP2/RKRBQNBN w CAc - 0 9",
     {21, 566, 13519, 375128, 9700847, 279864836}},  // 925
    {"rkrq1b2/pppppppb/3n2np/2N5/4P3/7P/PPPP1PP1/RKRQ1BBN w CAca - 1 9",
     {33, 654, 21708, 479678, 15990307, 382218272}},  // 926
    {"rkr1nnbb/ppp2p1p/3p1qp1/4p3/P5P1/3PN3/1PP1PP1P/RKRQN1BB w CAca - 1 9",
     {28, 715, 20361, 555328, 16303092, 468666425}},  // 927
    {"bbrkrnqn/1p1ppppp/8/8/p2pP3/PP6/2P2PPP/BBRKRNQN w ECec - 0 9",
     {24, 757, 19067, 603231, 15957628, 509307623}},  // 928
    {"brkbrnqn/ppp2p2/4p3/P2p2pp/6P1/5P2/1PPPP2P/BRKBRNQN w EBeb - 0 9",
     {25, 548, 14563, 348259, 9688526, 247750144}},  // 929
    {"brkr1bqn/1pppppp1/3n3p/1p6/P7/4P1P1/1PPP1P1P/BRKRN1QN w DBdb - 0 9",
     {19, 359, 7430, 157099, 3521652, 81787718}},  // 930
    {"brkr1qnb/pppp2pp/2B1p3/5p2/2n5/6PP/PPPPPPN1/BRKR1QN1 w DBdb - 1 9",
     {27, 854, 23303, 741626, 20558538, 667089231}},  // 931
    {"rbbkrnqn/p1p1p1pp/8/1p1p4/1P1Pp3/6N1/P1P2PPP/RBBKRNQ1 w EAea - 0 9",
     {28, 723, 19844, 514440, 14621108, 397454100}},  // 932
    {"rkbbrn1n/pppppp2/5q1p/6p1/3P3P/4P3/PPP2PP1/RKBBRNQN w EAea - 1 9",
     {25, 741, 19224, 585198, 15605840, 485037906}},  // 933
    {"rkbr1bq1/ppnppppp/6n1/2p5/2P1N2P/8/PP1PPPP1/RKBRNBQ1 w DAda - 3 9",
     {24, 547, 14359, 339497, 9410221, 234041078}},  // 934
    {"1kbrnqnb/r1ppppp1/8/pp5p/8/1P1NP3/P1PP1PPP/RKB1RQNB w Ad - 2 9",
     {26, 618, 17305, 442643, 13112297, 357030697}},  // 935
    {"rbkrb1qn/1pp1ppp1/3pn2p/pP6/8/4N1P1/P1PPPP1P/RBKRB1QN w DAda - 0 9",
     {21, 544, 12492, 338832, 8381483, 236013157}},  // 936
    {"rkrbbnqn/ppppp3/5p2/6pp/5PBP/4P3/PPPP2P1/RKR1BNQN w CAca - 0 9",
     {30, 891, 25435, 764356, 21894752, 669256602}},  // 937
    {"rkr1bb1n/ppppp1pp/5p2/4n3/3QP3/5P2/RPPP2PP/1KRNBB1N w Cca - 1 9",
     {45, 1172, 51766, 1332060, 57856784, 1501852662}},  // 938
    {"rkr1bqnb/pp1ppppp/8/2pN4/1P6/5N2/P1PPnPPP/RKR1BQ1B w CAca - 0 9",
     {28, 730, 20511, 559167, 16323242, 463032124}},  // 939
    {"rbkrnqb1/2ppppp1/p5np/1p6/8/3N4/PPPPPPPP/RBKRQNB1 w DAda - 2 9",
     {20, 417, 9159, 217390, 5180716, 133936564}},  // 940
    {"rkrbnqb1/p1pppnpp/5p2/1p6/2P5/1P1P1N2/P3PPPP/RKRB1QBN w CAca - 0 9",
     {25, 546, 14039, 330316, 8813781, 222026485}},  // 941
    {"rkr1qbbn/ppppppp1/4n3/7p/8/P7/KPPPPPPP/R1RNQBBN w ca - 0 9",
     {22, 484, 11458, 267495, 6633319, 163291279}},  // 942
    {"rkrnqnb1/1ppppp2/p5p1/7p/8/P1bPP3/1PP1QPPP/RKRN1NBB w CAca - 0 9",
     {22, 636, 15526, 441001, 11614241, 331083405}},  // 943
    {"b2krn1q/p1rppppp/1Q3n2/2p1b3/1P4P1/8/P1PPPP1P/BBRKRNN1 w ECe - 3 9",
     {36, 1192, 42945, 1406795, 50382104, 1650202838}},  // 944
    {"brkbrnn1/pp1pppp1/7q/2p5/6Pp/4P1NP/PPPP1P2/BRKBR1NQ w EBeb - 2 9",
     {30, 978, 29593, 942398, 29205057, 936568065}},  // 945
    {"brkrnb1q/pp1p1ppp/2p1p3/5n2/1P6/5N1N/P1PPPPPP/BRKR1B1Q w DBdb - 1 9",
     {31, 897, 27830, 810187, 25423729, 755334868}},  // 946
    {"brkr1nqb/pp1p1pp1/2pn3p/P3p3/4P3/6P1/1PPP1P1P/BRKRNNQB w DBdb - 0 9",
     {19, 382, 8052, 182292, 4232274, 103537333}},  // 947
    {"r1bkrn1q/ppbppppp/5n2/2p5/3P4/P6N/1PP1PPPP/RBBKRNQ1 w EAea - 3 9",
     {27, 822, 22551, 678880, 19115128, 578210135}},  // 948
    {"rkbbrnnq/pp2pppp/8/2pp4/P1P5/1P3P2/3PP1PP/RKBBRNNQ w EAea - 1 9",
     {23, 643, 15410, 442070, 11170489, 329615708}},  // 949
    {"rkbr1b1q/p1pppppp/1p1n4/7n/5QP1/3N4/PPPPPP1P/RKBR1BN1 w DAda - 4 9",
     {37, 943, 34382, 880474, 31568111, 842265141}},  // 950
    {"rkbr1nqb/pppp2np/8/4ppp1/1P6/6N1/P1PPPPPP/RKBRN1QB w DAda - 1 9",
     {23, 574, 13260, 362306, 9020291, 261247606}},  // 951
    {"rbkr1nnq/p1p1pp1p/1p4p1/3p4/b3P3/4N3/PPPPNPPP/RBKRB1Q1 w DAda - 0 9",
     {26, 900, 23414, 805006, 21653203, 745802405}},  // 952
    {"rkrbb1nq/p2pppp1/1p4n1/2p4p/3N4/4P1P1/PPPP1P1P/RKRBBN1Q w CAca - 0 9",
     {32, 697, 22231, 531121, 17150175, 441578567}},  // 953
    {"rkrnbb1q/pp2pp1p/6pn/2pp4/2B1P2P/8/PPPP1PP1/RKRNB1NQ w CAca - 0 9",
     {28, 854, 23853, 755990, 21823412, 712787248}},  // 954
    {"rk2bnqb/pprpppp1/4n2p/2p5/P7/3P2NP/1PP1PPP1/RKRNB1QB w CAa - 1 9",
     {26, 596, 16251, 414862, 11758184, 323043654}},  // 955
    {"r1krnnbq/pp1ppp1p/6p1/2p5/2P5/P3P3/Rb1P1PPP/1BKRNNBQ w Dda - 0 9",
     {2, 61, 1312, 40072, 937188, 28753562}},  // 956
    {"1krbnnbq/1pp1p1pp/r7/p2p1p2/3PP3/2P3P1/PP3P1P/RKRBNNBQ w CAc - 0 9",
     {30, 953, 28033, 860530, 25531358, 787205262}},  // 957
    {"rkr1nbbq/2ppp1pp/1pn5/p4p2/P6P/3P4/1PP1PPPB/RKRNNB1Q w CAca - 1 9",
     {24, 645, 15689, 446423, 11484012, 341262639}},  // 958
    {"rkrnnqbb/p1ppp2p/Qp6/4Pp2/5p2/8/PPPP2PP/RKRNN1BB w CAca - 0 9",
     {35, 929, 32020, 896130, 31272517, 915268405}},  // 959
    {"bbq1nr1r/pppppk1p/2n2p2/6p1/P4P2/4P1P1/1PPP3P/BBQNNRKR w HF - 1 9",
     {23, 589, 14744, 387556, 10316716, 280056112}}  // 960
};
}  // namespace

TEST(ChessBoard, FRC) {
  // Up to 6 is possible, but too long, so keeping at 4.
  for (int i = 0; i < 4; ++i) {
    for (const auto& x : kChess960Positions) {
      ChessBoard board;
      board.SetFromFen(x.fen);
      EXPECT_EQ(Perft(board, i + 1), x.perft[i])
          << "Position: [" << x.fen << "] Depth: " << i + 1 << "\n"
          << board.DebugString();
    }
  }
}

TEST(ChessBoard, HasMatingMaterialStartPosition) {
  ChessBoard board;
  board.SetFromFen(ChessBoard::kStartposFen);
  EXPECT_TRUE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialBareKings) {
  ChessBoard board;
  board.SetFromFen("8/8/8/4k3/8/8/2K5/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialSingleMinorPiece) {
  ChessBoard board;
  board.SetFromFen("8/8/8/4k3/1N6/8/2K5/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/4k3/7b/8/2K5/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialSingleMajorPieceOrPawn) {
  ChessBoard board;
  board.SetFromFen("8/8/8/4k3/8/5R2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/4k3/8/5q2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/4k3/8/5P2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialTwoKnights) {
  ChessBoard board;
  board.SetFromFen("8/8/8/3nk3/8/5N2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/3nk3/8/5n2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialBishopAndKnight) {
  ChessBoard board;
  board.SetFromFen("8/8/8/3bk3/8/5N2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/3Bk3/8/5N2/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialMultipleBishopsSameColor) {
  ChessBoard board;
  board.SetFromFen("8/8/8/3Bk3/8/5B2/2K5/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/4kb2/8/2K2B2/8/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
  board.SetFromFen("B7/1B3b2/2B3b1/4k2b/8/8/2K5/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
  board.SetFromFen("B7/1B6/2B5/4k3/8/8/2K5/8 w - - 0 1");
  EXPECT_FALSE(board.HasMatingMaterial());
}

TEST(ChessBoard, HasMatingMaterialMultipleBishopsNotSameColor) {
  ChessBoard board;
  board.SetFromFen("8/8/8/4k3/8/2K1bb2/8/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/4k3/8/2K1Bb2/8/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("8/8/8/4k3/8/2K2b2/5B2/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
  board.SetFromFen("B7/1B3b2/2B3b1/4k2b/7B/8/2K5/8 w - - 0 1");
  EXPECT_TRUE(board.HasMatingMaterial());
}

}  // namespace lczero

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  lczero::InitializeMagicBitboards();
  return RUN_ALL_TESTS();
}

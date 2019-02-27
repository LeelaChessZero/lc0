/*
  Originally from cfish's tbprobe.c
  Copyright (c) 2013-2018 Ronald de Man
  That file may be redistributed and/or modified without restrictions.

  This modified version is available under the GPL:

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

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>

#include "syzygy/syzygy.h"

#include "utils/exception.h"
#include "utils/logging.h"
#include "utils/mutex.h"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace lczero {

namespace {

#define TB_PIECES 7
#define TB_HASHBITS (TB_PIECES < 7 ? 11 : 12)
#define TB_MAX_PIECE (TB_PIECES < 7 ? 254 : 650)
#define TB_MAX_PAWN (TB_PIECES < 7 ? 256 : 861)

#ifdef _WIN32
typedef HANDLE map_t;
#define SEP_CHAR ';'
#else
typedef size_t map_t;
#define SEP_CHAR ':'
#endif

typedef uint64_t Key;

constexpr const char* kSuffix[] = {".rtbw", ".rtbm", ".rtbz"};
constexpr uint32_t kMagic[] = {0x5d23e871, 0x88ac504b, 0xa50c66d7};
enum { WDL, DTM, DTZ };

enum { PIECE_ENC, FILE_ENC, RANK_ENC };

enum PieceType {
  PAWN = 1,
  KNIGHT,
  BISHOP,
  ROOK,
  QUEEN,
  KING,
};
enum Piece {
  W_PAWN = 1,
  W_KNIGHT,
  W_BISHOP,
  W_ROOK,
  W_QUEEN,
  W_KING,
  B_PAWN = 9,
  B_KNIGHT,
  B_BISHOP,
  B_ROOK,
  B_QUEEN,
  B_KING,
};

struct PairsData {
  uint8_t* indexTable;
  uint16_t* sizeTable;
  uint8_t* data;
  uint16_t* offset;
  uint8_t* symLen;
  uint8_t* symPat;
  uint8_t blockSize;
  uint8_t idxBits;
  uint8_t minLen;
  uint8_t constValue[2];
  uint64_t base[1];  // must be base[1] in C++
};

struct EncInfo {
  PairsData* precomp;
  size_t factor[TB_PIECES];
  uint8_t pieces[TB_PIECES];
  uint8_t norm[TB_PIECES];
};

struct BaseEntry {
  Key key;
  uint8_t* data[3];
  map_t mapping[3];
  std::atomic<bool> ready[3];
  uint8_t num;
  bool symmetric;
  bool hasPawns;
  bool hasDtm;
  bool hasDtz;
  union {
    bool kk_enc;
    uint8_t pawns[2];
  };
  bool dtmLossOnly;
};

struct PieceEntry : BaseEntry {
  EncInfo ei[5];  // 2 + 2 + 1
  uint16_t* dtmMap;
  uint16_t dtmMapIdx[2][2];
  void* dtzMap;
  uint16_t dtzMapIdx[4];
  uint8_t dtzFlags;
};

struct PawnEntry : BaseEntry {
  EncInfo ei[24];  // 4 * 2 + 6 * 2 + 4
  uint16_t* dtmMap;
  uint16_t dtmMapIdx[6][2][2];
  void* dtzMap;
  uint16_t dtzMapIdx[4][4];
  uint8_t dtzFlags[4];
  bool dtmSwitched;
};

struct TbHashEntry {
  Key key;
  BaseEntry* ptr;
};

constexpr int kWdlToDtz[] = {-1, -101, 0, 101, 1};

// DTZ tables don't store valid scores for moves that reset the rule50 counter
// like captures and pawn moves but we can easily recover the correct dtz of the
// previous move if we know the position's WDL score.
int dtz_before_zeroing(WDLScore wdl) { return kWdlToDtz[wdl + 2]; }

// Return the sign of a number (-1, 0, 1)
template <typename T>
int sign_of(T val) {
  return (T(0) < val) - (val < T(0));
}

int count_pieces(const ChessBoard& pos, int type, bool theirs) {
  const BitBoard all = theirs ? pos.theirs() : pos.ours();
  switch (type) {
    case KING:
      return 1;
    case QUEEN:
      return (all & pos.queens()).count_few();
    case ROOK:
      return (all & pos.rooks()).count_few();
    case BISHOP:
      return (all & pos.bishops()).count_few();
    case KNIGHT:
      return (theirs ? pos.their_knights() : pos.our_knights()).count_few();
    case PAWN:
      return (all & pos.pawns()).count_few();
    default:
      assert(false);
  }
  return 0;
}

BitBoard pieces(const ChessBoard& pos, int type, bool theirs) {
  const BitBoard all = theirs ? pos.theirs() : pos.ours();
  switch (type) {
    case KING:
      return theirs ? pos.their_king() : pos.our_king();
    case QUEEN:
      return all & pos.queens();
    case ROOK:
      return all & pos.rooks();
    case BISHOP:
      return all & pos.bishops();
    case KNIGHT:
      return theirs ? pos.their_knights() : pos.our_knights();
    case PAWN:
      return all & pos.pawns();
    default:
      assert(false);
  }
  return BitBoard();
}

bool is_capture(const ChessBoard& pos, const Move& move) {
  // Simple capture.
  if (pos.theirs().get(move.to())) return true;
  // Enpassant capture. Pawn moves other than straight it must be a capture.
  if (pos.pawns().get(move.from()) && move.from().col() != move.to().col()) {
    return true;
  }
  return false;
}

constexpr char kPieceToChar[] = " PNBRQK  pnbrqk";

// Given a position, produce a text string of the form KQPvKRP, where
// "KQP" represents the white pieces if flip == false and the black pieces
// if flip == true.
void prt_str(const ChessBoard& pos, char* str, bool flip) {
  const bool first_theirs = flip ^ pos.flipped();

  for (int pt = KING; pt >= PAWN; pt--) {
    for (int i = count_pieces(pos, pt, first_theirs); i > 0; i--) {
      *str++ = kPieceToChar[pt];
    }
  }
  *str++ = 'v';
  for (int pt = KING; pt >= PAWN; pt--) {
    for (int i = count_pieces(pos, pt, !first_theirs); i > 0; i--) {
      *str++ = kPieceToChar[pt];
    }
  }
  *str++ = 0;
}

#define pchr(i) kPieceToChar[QUEEN - (i)]
#define Swap(a, b) \
  {                \
    int tmp = a;   \
    a = b;         \
    b = tmp;       \
  }

#define PIECE(x) (static_cast<PieceEntry*>(x))
#define PAWN(x) (static_cast<PawnEntry*>(x))

int num_tables(BaseEntry* be, const int type) {
  return be->hasPawns ? type == DTM ? 6 : 4 : 1;
}

EncInfo* first_ei(BaseEntry* be, const int type) {
  return be->hasPawns ? &PAWN(be)->ei[type == WDL ? 0 : type == DTM ? 8 : 20]
                      : &PIECE(be)->ei[type == WDL ? 0 : type == DTM ? 2 : 4];
}

constexpr int8_t kOffDiag[] = {
    0, -1, -1, -1, -1, -1, -1, -1, 1, 0, -1, -1, -1, -1, -1, -1,
    1, 1,  0,  -1, -1, -1, -1, -1, 1, 1, 1,  0,  -1, -1, -1, -1,
    1, 1,  1,  1,  0,  -1, -1, -1, 1, 1, 1,  1,  1,  0,  -1, -1,
    1, 1,  1,  1,  1,  1,  0,  -1, 1, 1, 1,  1,  1,  1,  1,  0};

constexpr uint8_t kTriangle[] = {
    6, 0, 1, 2, 2, 1, 0, 6, 0, 7, 3, 4, 4, 3, 7, 0, 1, 3, 8, 5, 5, 8,
    3, 1, 2, 4, 5, 9, 9, 5, 4, 2, 2, 4, 5, 9, 9, 5, 4, 2, 1, 3, 8, 5,
    5, 8, 3, 1, 0, 7, 3, 4, 4, 3, 7, 0, 6, 0, 1, 2, 2, 1, 0, 6};

constexpr uint8_t kFlipDiag[] = {
    0, 8,  16, 24, 32, 40, 48, 56, 1, 9,  17, 25, 33, 41, 49, 57,
    2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
    4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61,
    6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63};

constexpr uint8_t kLower[] = {
    28, 0,  1,  2,  3,  4,  5,  6,  0, 29, 7,  8,  9,  10, 11, 12,
    1,  7,  30, 13, 14, 15, 16, 17, 2, 8,  13, 31, 18, 19, 20, 21,
    3,  9,  14, 18, 32, 22, 23, 24, 4, 10, 15, 19, 22, 33, 25, 26,
    5,  11, 16, 20, 23, 25, 34, 27, 6, 12, 17, 21, 24, 26, 27, 35};

constexpr uint8_t kDiag[] = {
    0, 0, 0, 0, 0, 0,  0,  8, 0, 1, 0, 0, 0,  0,  9, 0, 0, 0, 2, 0, 0,  10,
    0, 0, 0, 0, 0, 3,  11, 0, 0, 0, 0, 0, 0,  12, 4, 0, 0, 0, 0, 0, 13, 0,
    0, 5, 0, 0, 0, 14, 0,  0, 0, 0, 6, 0, 15, 0,  0, 0, 0, 0, 0, 7};

constexpr uint8_t kFlap[2][64] = {
    {0, 0,  0,  0,  0,  0,  0,  0, 0, 6,  12, 18, 18, 12, 6,  0,
     1, 7,  13, 19, 19, 13, 7,  1, 2, 8,  14, 20, 20, 14, 8,  2,
     3, 9,  15, 21, 21, 15, 9,  3, 4, 10, 16, 22, 22, 16, 10, 4,
     5, 11, 17, 23, 23, 17, 11, 5, 0, 0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  3,  2,  1,  0,
     4,  5,  6,  7,  7,  6,  5,  4,  8,  9,  10, 11, 11, 10, 9,  8,
     12, 13, 14, 15, 15, 14, 13, 12, 16, 17, 18, 19, 19, 18, 17, 16,
     20, 21, 22, 23, 23, 22, 21, 20, 0,  0,  0,  0,  0,  0,  0,  0}};

constexpr uint8_t kPawnTwist[2][64] = {
    {0,  0,  0,  0, 0, 0,  0,  0,  47, 35, 23, 11, 10, 22, 34, 46,
     45, 33, 21, 9, 8, 20, 32, 44, 43, 31, 19, 7,  6,  18, 30, 42,
     41, 29, 17, 5, 4, 16, 28, 40, 39, 27, 15, 3,  2,  14, 26, 38,
     37, 25, 13, 1, 0, 12, 24, 36, 0,  0,  0,  0,  0,  0,  0,  0},
    {0,  0,  0,  0,  0,  0,  0,  0,  47, 45, 43, 41, 40, 42, 44, 46,
     39, 37, 35, 33, 32, 34, 36, 38, 31, 29, 27, 25, 24, 26, 28, 30,
     23, 21, 19, 17, 16, 18, 20, 22, 15, 13, 11, 9,  8,  10, 12, 14,
     7,  5,  3,  1,  0,  2,  4,  6,  0,  0,  0,  0,  0,  0,  0,  0}};

constexpr int16_t kKKIdx[10][64] = {
    {-1, -1, -1, 0,  1,  2,  3,  4,  -1, -1, -1, 5,  6,  7,  8,  9,
     10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
     26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
     42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57},
    {58,  -1,  -1,  -1,  59,  60,  61,  62,  63,  -1,  -1,  -1,  64,
     65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
     78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
     91,  92,  93,  94,  95,  96,  97,  98,  99,  100, 101, 102, 103,
     104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115},
    {116, 117, -1,  -1,  -1,  118, 119, 120, 121, 122, -1,  -1,  -1,
     123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
     136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
     149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
     162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173},
    {174, -1,  -1,  -1,  175, 176, 177, 178, 179, -1,  -1,  -1,  180,
     181, 182, 183, 184, -1,  -1,  -1,  185, 186, 187, 188, 189, 190,
     191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
     204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,
     217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228},
    {229, 230, -1,  -1,  -1,  231, 232, 233, 234, 235, -1,  -1,  -1,
     236, 237, 238, 239, 240, -1,  -1,  -1,  241, 242, 243, 244, 245,
     246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258,
     259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271,
     272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283},
    {284, 285, 286, 287, 288, 289, 290, 291, 292, 293, -1,  -1,  -1,
     294, 295, 296, 297, 298, -1,  -1,  -1,  299, 300, 301, 302, 303,
     -1,  -1,  -1,  304, 305, 306, 307, 308, 309, 310, 311, 312, 313,
     314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326,
     327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338},
    {-1,  -1,  339, 340, 341, 342, 343, 344, -1,  -1,  345, 346, 347,
     348, 349, 350, -1,  -1,  441, 351, 352, 353, 354, 355, -1,  -1,
     -1,  442, 356, 357, 358, 359, -1,  -1,  -1,  -1,  443, 360, 361,
     362, -1,  -1,  -1,  -1,  -1,  444, 363, 364, -1,  -1,  -1,  -1,
     -1,  -1,  445, 365, -1,  -1,  -1,  -1,  -1,  -1,  -1,  446},
    {-1, -1, -1, 366, 367, 368, 369, 370, -1, -1, -1, 371, 372, 373, 374, 375,
     -1, -1, -1, 376, 377, 378, 379, 380, -1, -1, -1, 447, 381, 382, 383, 384,
     -1, -1, -1, -1,  448, 385, 386, 387, -1, -1, -1, -1,  -1,  449, 388, 389,
     -1, -1, -1, -1,  -1,  -1,  450, 390, -1, -1, -1, -1,  -1,  -1,  -1,  451},
    {452, 391, 392, 393, 394, 395, 396, 397, -1,  -1,  -1,  -1,  398,
     399, 400, 401, -1,  -1,  -1,  -1,  402, 403, 404, 405, -1,  -1,
     -1,  -1,  406, 407, 408, 409, -1,  -1,  -1,  -1,  453, 410, 411,
     412, -1,  -1,  -1,  -1,  -1,  454, 413, 414, -1,  -1,  -1,  -1,
     -1,  -1,  455, 415, -1,  -1,  -1,  -1,  -1,  -1,  -1,  456},
    {457, 416, 417, 418, 419, 420, 421, 422, -1,  458, 423, 424, 425,
     426, 427, 428, -1,  -1,  -1,  -1,  -1,  429, 430, 431, -1,  -1,
     -1,  -1,  -1,  432, 433, 434, -1,  -1,  -1,  -1,  -1,  435, 436,
     437, -1,  -1,  -1,  -1,  -1,  459, 438, 439, -1,  -1,  -1,  -1,
     -1,  -1,  460, 440, -1,  -1,  -1,  -1,  -1,  -1,  -1,  461}};

constexpr uint8_t kFileToFile[] = {0, 1, 2, 3, 3, 2, 1, 0};
constexpr int kWdlToMap[5] = {1, 3, 0, 2, 0};
constexpr uint8_t kPAFlags[5] = {8, 0, 0, 0, 4};

size_t Binomial[7][64];
size_t PawnIdx[2][6][24];
size_t PawnFactorFile[6][4];
size_t PawnFactorRank[6][6];
Key MaterialHash[16][64];

void init_indices() {
  // Binomial[k][n] = Bin(n, k)
  for (int i = 0; i < 7; i++)
    for (int j = 0; j < 64; j++) {
      size_t f = 1;
      size_t l = 1;
      for (int k = 0; k < i; k++) {
        f *= (j - k);
        l *= (k + 1);
      }
      Binomial[i][j] = f / l;
    }

  for (int i = 0; i < 6; i++) {
    size_t s = 0;
    for (int j = 0; j < 24; j++) {
      PawnIdx[0][i][j] = s;
      s += Binomial[i][kPawnTwist[0][(1 + (j % 6)) * 8 + (j / 6)]];
      if ((j + 1) % 6 == 0) {
        PawnFactorFile[i][j / 6] = s;
        s = 0;
      }
    }
  }

  for (int i = 0; i < 6; i++) {
    size_t s = 0;
    for (int j = 0; j < 24; j++) {
      PawnIdx[1][i][j] = s;
      s += Binomial[i][kPawnTwist[1][(1 + (j / 4)) * 8 + (j % 4)]];
      if ((j + 1) % 4 == 0) {
        PawnFactorRank[i][j / 4] = s;
        s = 0;
      }
    }
  }
  // TODO: choose a good seed.
  std::mt19937 gen(123523465);
  std::uniform_int_distribution<Key> dist(std::numeric_limits<Key>::lowest(),
                                          std::numeric_limits<Key>::max());
  for (int i = 0; i < 16; i++) {
    // MaterialHash for 0 instances of a piece is 0 as an optimization so
    // calc_key_from_pieces doesn't have to add in all the missing pieces.
    MaterialHash[i][0] = 0;
    for (int j = 1; j < 64; j++) {
      MaterialHash[i][j] = dist(gen);
    }
  }
}

std::once_flag indicies_flag;

void initonce_indicies() { std::call_once(indicies_flag, init_indices); }

// Produce a 64-bit material key corresponding to the material combination
// defined by pcs[16], where pcs[1], ..., pcs[6] are the number of white
// pawns, ..., kings and pcs[9], ..., pcs[14] are the number of black
// pawns, ..., kings.
Key calc_key_from_pcs(int* pcs, bool flip) {
  Key key = 0;

  const int color = !flip ? 0 : 8;
  for (int i = W_PAWN; i <= B_KING; i++) key += MaterialHash[i][pcs[i ^ color]];

  return key;
}

// Produce a 64-bit material key corresponding to the material combination
// piece[0], ..., piece[num - 1], where each value corresponds to a piece
// (1-6 for white pawn-king, 9-14 for black pawn-king).
Key calc_key_from_pieces(uint8_t* piece, int num) {
  Key key = 0;

  for (int i = 0; i < num; i++) {
    if (piece[i]) key += MaterialHash[piece[i]][1];
  }

  return key;
}

Key calc_key_from_position(const ChessBoard& pos) {
  Key key = 0;
  const bool flipped = pos.flipped();
  for (int i = PAWN; i <= KING; i++) {
    // White pieces - ours if not flipped.
    key += MaterialHash[i][count_pieces(pos, i, flipped)];
    // Black pieces - ours if flipped.
    key += MaterialHash[i + 8][count_pieces(pos, i, !flipped)];
  }
  return key;
}

int leading_pawn(int* p, BaseEntry* be, const int enc) {
  for (int i = 1; i < be->pawns[0]; i++) {
    if (kFlap[enc - 1][p[0]] > kFlap[enc - 1][p[i]]) Swap(p[0], p[i]);
  }
  return enc == FILE_ENC ? kFileToFile[p[0] & 7] : (p[0] - 8) >> 3;
}

size_t encode(int* p, EncInfo* ei, BaseEntry* be, const int enc) {
  const int n = be->num;
  size_t idx;
  int k;

  if (p[0] & 0x04) {
    for (int i = 0; i < n; i++) p[i] ^= 0x07;
  }

  if (enc == PIECE_ENC) {
    if (p[0] & 0x20) {
      for (int i = 0; i < n; i++) p[i] ^= 0x38;
    }

    for (int i = 0; i < n; i++) {
      if (kOffDiag[p[i]]) {
        if (kOffDiag[p[i]] > 0 && i < (be->kk_enc ? 2 : 3)) {
          for (int j = 0; j < n; j++) p[j] = kFlipDiag[p[j]];
        }
        break;
      }
    }

    if (be->kk_enc) {
      idx = kKKIdx[kTriangle[p[0]]][p[1]];
      k = 2;
    } else {
      const int s1 = (p[1] > p[0]);
      const int s2 = (p[2] > p[0]) + (p[2] > p[1]);

      if (kOffDiag[p[0]]) {
        idx = kTriangle[p[0]] * 63 * 62 + (p[1] - s1) * 62 + (p[2] - s2);
      } else if (kOffDiag[p[1]]) {
        idx =
            6 * 63 * 62 + kDiag[p[0]] * 28 * 62 + kLower[p[1]] * 62 + p[2] - s2;
      } else if (kOffDiag[p[2]]) {
        idx = 6 * 63 * 62 + 4 * 28 * 62 + kDiag[p[0]] * 7 * 28 +
              (kDiag[p[1]] - s1) * 28 + kLower[p[2]];
      } else {
        idx = 6 * 63 * 62 + 4 * 28 * 62 + 4 * 7 * 28 + kDiag[p[0]] * 7 * 6 +
              (kDiag[p[1]] - s1) * 6 + (kDiag[p[2]] - s2);
      }
      k = 3;
    }
    idx *= ei->factor[0];
  } else {
    for (int i = 1; i < be->pawns[0]; i++) {
      for (int j = i + 1; j < be->pawns[0]; j++) {
        if (kPawnTwist[enc - 1][p[i]] < kPawnTwist[enc - 1][p[j]]) {
          Swap(p[i], p[j]);
        }
      }
    }
    k = be->pawns[0];
    idx = PawnIdx[enc - 1][k - 1][kFlap[enc - 1][p[0]]];
    for (int i = 1; i < k; i++) {
      idx += Binomial[k - i][kPawnTwist[enc - 1][p[i]]];
    }
    idx *= ei->factor[0];

    // Pawns of other color
    if (be->pawns[1]) {
      const int t = k + be->pawns[1];
      for (int i = k; i < t; i++) {
        for (int j = i + 1; j < t; j++) {
          if (p[i] > p[j]) Swap(p[i], p[j]);
        }
      }
      size_t s = 0;
      for (int i = k; i < t; i++) {
        const int sq = p[i];
        int skips = 0;
        for (int j = 0; j < k; j++) skips += (sq > p[j]);
        s += Binomial[i - k + 1][sq - skips - 8];
      }
      idx += s * ei->factor[k];
      k = t;
    }
  }

  for (; k < n;) {
    const int t = k + ei->norm[k];
    for (int i = k; i < t; i++) {
      for (int j = i + 1; j < t; j++) {
        if (p[i] > p[j]) Swap(p[i], p[j]);
      }
    }
    size_t s = 0;
    for (int i = k; i < t; i++) {
      const int sq = p[i];
      int skips = 0;
      for (int j = 0; j < k; j++) skips += (sq > p[j]);
      s += Binomial[i - k + 1][sq - skips];
    }
    idx += s * ei->factor[k];
    k = t;
  }

  return idx;
}

size_t encode_piece(int* p, EncInfo* ei, BaseEntry* be) {
  return encode(p, ei, be, PIECE_ENC);
}

size_t encode_pawn_f(int* p, EncInfo* ei, BaseEntry* be) {
  return encode(p, ei, be, FILE_ENC);
}

size_t encode_pawn_r(int* p, EncInfo* ei, BaseEntry* be) {
  return encode(p, ei, be, RANK_ENC);
}

// Count number of placements of k like pieces on n squares
size_t subfactor(size_t k, size_t n) {
  size_t f = n;
  size_t l = 1;
  for (size_t i = 1; i < k; i++) {
    f *= n - i;
    l *= i + 1;
  }

  return f / l;
}

size_t init_enc_info(EncInfo* ei, BaseEntry* be, uint8_t* tb, int shift, int t,
                     const int enc) {
  const bool more_pawns = enc != PIECE_ENC && be->pawns[1] > 0;

  for (int i = 0; i < be->num; i++) {
    ei->pieces[i] = (tb[i + 1 + more_pawns] >> shift) & 0x0f;
    ei->norm[i] = 0;
  }

  const int order = (tb[0] >> shift) & 0x0f;
  const int order2 = more_pawns ? (tb[1] >> shift) & 0x0f : 0x0f;

  int k = ei->norm[0] = enc != PIECE_ENC ? be->pawns[0] : be->kk_enc ? 2 : 3;

  if (more_pawns) {
    ei->norm[k] = be->pawns[1];
    k += ei->norm[k];
  }

  for (int i = k; i < be->num; i += ei->norm[i]) {
    for (int j = i; j < be->num && ei->pieces[j] == ei->pieces[i]; j++) {
      ei->norm[i]++;
    }
  }

  int n = 64 - k;
  size_t f = 1;

  for (int i = 0; k < be->num || i == order || i == order2; i++) {
    if (i == order) {
      ei->factor[0] = f;
      f *= enc == FILE_ENC
               ? PawnFactorFile[ei->norm[0] - 1][t]
               : enc == RANK_ENC ? PawnFactorRank[ei->norm[0] - 1][t]
                                 : be->kk_enc ? 462 : 31332;
    } else if (i == order2) {
      ei->factor[ei->norm[0]] = f;
      f *= subfactor(ei->norm[ei->norm[0]], 48 - ei->norm[0]);
    } else {
      ei->factor[k] = f;
      f *= subfactor(ei->norm[k], n);
      n -= ei->norm[k];
      k += ei->norm[k];
    }
  }

  return f;
}

void calc_symLen(PairsData* d, uint32_t s, char* tmp) {
  uint8_t* w = d->symPat + 3 * s;
  const uint32_t s2 = (w[2] << 4) | (w[1] >> 4);
  if (s2 == 0x0fff)
    d->symLen[s] = 0;
  else {
    const uint32_t s1 = ((w[1] & 0xf) << 8) | w[0];
    if (!tmp[s1]) calc_symLen(d, s1, tmp);
    if (!tmp[s2]) calc_symLen(d, s2, tmp);
    d->symLen[s] = d->symLen[s1] + d->symLen[s2] + 1;
  }
  tmp[s] = 1;
}

int is_little_endian() {
  union {
    uint32_t i;
    uint8_t byte[4];
  } num_union = {0x01020304};

  return num_union.byte[0] == 4;
}

template <typename T, int Half = sizeof(T) / 2, int End = sizeof(T) - 1>
T swap_endian(T val) {
  static_assert(std::is_unsigned<T>::value,
                "Argument of swap_endian not unsigned");
  T x = val;
  uint8_t tmp, *c = (uint8_t*)&x;
  for (int i = 0; i < Half; ++i) {
    tmp = c[i], c[i] = c[End - i], c[End - i] = tmp;
  }
  return x;
}

uint32_t from_le_u32(uint32_t v) {
  return is_little_endian() ? v : swap_endian(v);
}

uint16_t from_le_u16(uint16_t v) {
  return is_little_endian() ? v : swap_endian(v);
}

uint64_t from_be_u64(uint64_t v) {
  return is_little_endian() ? swap_endian(v) : v;
}

uint32_t from_be_u32(uint32_t v) {
  return is_little_endian() ? swap_endian(v) : v;
}

uint32_t read_le_u32(void* p) {
  return from_le_u32(*static_cast<uint32_t*>(p));
}

uint16_t read_le_u16(void* p) {
  return from_le_u16(*static_cast<uint16_t*>(p));
}

PairsData* setup_pairs(uint8_t** ptr, size_t tb_size, size_t* size,
                       uint8_t* flags, int type) {
  PairsData* d;
  uint8_t* data = *ptr;

  *flags = data[0];
  if (data[0] & 0x80) {
    d = static_cast<PairsData*>(malloc(sizeof(*d)));
    d->idxBits = 0;
    d->constValue[0] = type == WDL ? data[1] : 0;
    d->constValue[1] = 0;
    *ptr = data + 2;
    size[0] = size[1] = size[2] = 0;
    return d;
  }

  const uint8_t block_size = data[1];
  const uint8_t idx_bits = data[2];
  const uint32_t real_num_blocks = read_le_u32(&data[4]);
  const uint32_t num_blocks = real_num_blocks + data[3];
  const int max_len = data[8];
  const int min_len = data[9];
  const int h = max_len - min_len + 1;
  const uint32_t num_syms = read_le_u16(&data[10 + 2 * h]);
  d = static_cast<PairsData*>(
      malloc(sizeof(*d) + h * sizeof(uint64_t) + num_syms));
  d->blockSize = block_size;
  d->idxBits = idx_bits;
  d->offset = reinterpret_cast<uint16_t*>(&data[10]);
  d->symLen = reinterpret_cast<uint8_t*>(d) + sizeof(*d) + h * sizeof(uint64_t);
  d->symPat = &data[12 + 2 * h];
  d->minLen = min_len;
  *ptr = &data[12 + 2 * h + 3 * num_syms + (num_syms & 1)];

  const size_t num_indices = (tb_size + (1ULL << idx_bits) - 1) >> idx_bits;
  size[0] = 6ULL * num_indices;
  size[1] = 2ULL * num_blocks;
  size[2] = static_cast<size_t>(real_num_blocks) << block_size;

  std::vector<char> tmp;
  tmp.resize(num_syms);
  memset(tmp.data(), 0, num_syms);
  for (uint32_t s = 0; s < num_syms; s++) {
    if (!tmp[s]) calc_symLen(d, s, tmp.data());
  }

  d->base[h - 1] = 0;
  for (int i = h - 2; i >= 0; i--) {
    d->base[i] = (d->base[i + 1] +
                  read_le_u16(reinterpret_cast<uint8_t*>(d->offset + i)) -
                  read_le_u16(reinterpret_cast<uint8_t*>(d->offset + i + 1))) /
                 2;
  }
  for (int i = 0; i < h; i++) d->base[i] <<= 64 - (min_len + i);

  d->offset -= d->minLen;

  return d;
}

uint8_t* decompress_pairs(PairsData* d, size_t idx) {
  if (!d->idxBits) return d->constValue;

  const uint32_t main_idx = idx >> d->idxBits;
  int lit_idx = (idx & ((static_cast<size_t>(1) << d->idxBits) - 1)) -
                (static_cast<size_t>(1) << (d->idxBits - 1));
  uint32_t block;
  memcpy(&block, d->indexTable + 6 * main_idx, sizeof(block));
  block = from_le_u32(block);

  const uint16_t idx_offset =
      *reinterpret_cast<uint16_t*>(d->indexTable + 6 * main_idx + 4);
  lit_idx += from_le_u16(idx_offset);

  if (lit_idx < 0) {
    while (lit_idx < 0) lit_idx += d->sizeTable[--block] + 1;
  } else {
    while (lit_idx > d->sizeTable[block]) lit_idx -= d->sizeTable[block++] + 1;
  }

  uint32_t* ptr = reinterpret_cast<uint32_t*>(
      d->data + (static_cast<size_t>(block) << d->blockSize));

  const int m = d->minLen;
  uint16_t* offset = d->offset;
  uint64_t* base = d->base - m;
  uint8_t* sym_len = d->symLen;
  uint32_t sym;
  uint32_t bit_cnt = 0;  // number of "empty bits" in code

  uint64_t code = from_be_u64(*reinterpret_cast<uint64_t*>(ptr));

  ptr += 2;
  for (;;) {
    int l = m;
    while (code < base[l]) l++;
    sym = from_le_u16(offset[l]);
    sym += (code - base[l]) >> (64 - l);
    if (lit_idx < static_cast<int>(sym_len[sym]) + 1) break;
    lit_idx -= static_cast<int>(sym_len[sym]) + 1;
    code <<= l;
    bit_cnt += l;
    if (bit_cnt >= 32) {
      bit_cnt -= 32;
      const uint32_t tmp = from_be_u32(*ptr++);
      code |= static_cast<uint64_t>(tmp) << bit_cnt;
    }
  }

  uint8_t* symPat = d->symPat;
  while (sym_len[sym] != 0) {
    uint8_t* w = symPat + (3 * sym);
    const int s1 = ((w[1] & 0xf) << 8) | w[0];
    if (lit_idx < static_cast<int>(sym_len[s1]) + 1) {
      sym = s1;
    } else {
      lit_idx -= static_cast<int>(sym_len[s1]) + 1;
      sym = (w[2] << 4) | (w[1] >> 4);
    }
  }

  return &symPat[3 * sym];
}

// p[i] is to contain the square 0-63 (A1-H8) for a piece of type
// pc[i] ^ flip, where 1 = white pawn, ..., 14 = black king and pc ^ flip
// flips between white and black if flip == true.
// Pieces of the same type are guaranteed to be consecutive.
int fill_squares(const ChessBoard& pos, uint8_t* pc, bool flip, int mirror,
                 int* p, int i) {
  // if pos.flipped the board is already mirrored, so mirror it again.
  if (pos.flipped()) mirror ^= 0x38;
  BitBoard bb = pieces(pos, pc[i] & 7,
                       static_cast<bool>((pc[i] >> 3)) ^ flip ^ pos.flipped());
  for (auto sq : bb) {
    p[i++] = sq.as_int() ^ mirror;
  }
  return i;
}

}  // namespace

class SyzygyTablebaseImpl {
 public:
  SyzygyTablebaseImpl(const std::string& paths)
      : piece_entries_(TB_MAX_PIECE), pawn_entries_(TB_MAX_PAWN) {
    initonce_indicies();

    if (paths.size() == 0 || paths == "<empty>") return;
    paths_ = paths;

    tb_hash_.resize(1 << TB_HASHBITS);

    char str[33];

    for (int i = 0; i < 5; i++) {
      sprintf(str, "K%cvK", pchr(i));
      init_tb(str);
    }

    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        sprintf(str, "K%cvK%c", pchr(i), pchr(j));
        init_tb(str);
      }
    }

    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        sprintf(str, "K%c%cvK", pchr(i), pchr(j));
        init_tb(str);
      }
    }

    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = 0; k < 5; k++) {
          sprintf(str, "K%c%cvK%c", pchr(i), pchr(j), pchr(k));
          init_tb(str);
        }
      }
    }

    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = j; k < 5; k++) {
          sprintf(str, "K%c%c%cvK", pchr(i), pchr(j), pchr(k));
          init_tb(str);
        }
      }
    }

    // 6- and 7-piece TBs make sense only with a 64-bit address space
    if (sizeof(size_t) < 8 || TB_PIECES < 6) goto finished;

    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = i; k < 5; k++) {
          for (int l = (i == k) ? j : k; l < 5; l++) {
            sprintf(str, "K%c%cvK%c%c", pchr(i), pchr(j), pchr(k), pchr(l));
            init_tb(str);
          }
        }
      }
    }

    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = j; k < 5; k++) {
          for (int l = 0; l < 5; l++) {
            sprintf(str, "K%c%c%cvK%c", pchr(i), pchr(j), pchr(k), pchr(l));
            init_tb(str);
          }
        }
      }
    }

    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = j; k < 5; k++) {
          for (int l = k; l < 5; l++) {
            sprintf(str, "K%c%c%c%cvK", pchr(i), pchr(j), pchr(k), pchr(l));
            init_tb(str);
          }
        }
      }
    }

    if (TB_PIECES < 7) goto finished;

    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = j; k < 5; k++) {
          for (int l = k; l < 5; l++) {
            for (int m = l; m < 5; m++) {
              sprintf(str, "K%c%c%c%c%cvK", pchr(i), pchr(j), pchr(k), pchr(l),
                      pchr(m));
              init_tb(str);
            }
          }
        }
      }
    }

    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = j; k < 5; k++) {
          for (int l = k; l < 5; l++) {
            for (int m = 0; m < 5; m++) {
              sprintf(str, "K%c%c%c%cvK%c", pchr(i), pchr(j), pchr(k), pchr(l),
                      pchr(m));
              init_tb(str);
            }
          }
        }
      }
    }

    for (int i = 0; i < 5; i++) {
      for (int j = i; j < 5; j++) {
        for (int k = j; k < 5; k++) {
          for (int l = 0; l < 5; l++) {
            for (int m = l; m < 5; m++) {
              sprintf(str, "K%c%c%cvK%c%c", pchr(i), pchr(j), pchr(k), pchr(l),
                      pchr(m));
              init_tb(str);
            }
          }
        }
      }
    }

  finished:
    CERR << "Found " << num_wdl_ << "WDL, " << num_dtm_ << " DTM and "
         << num_dtz_ << " DTZ tablebase files.";
  }

  ~SyzygyTablebaseImpl() {
    // if pathString was set there may be entries in need of cleaning.
    if (!paths_.empty()) {
      for (int i = 0; i < num_piece_entries_; i++)
        free_tb_entry(&piece_entries_[i]);
      for (int i = 0; i < num_pawn_entries_; i++)
        free_tb_entry(&pawn_entries_[i]);
    }
  }

  int max_cardinality() const { return max_cardinality_; }

  int probe_wdl_table(const ChessBoard& pos, int* success) {
    return probe_table(pos, 0, success, WDL);
  }

  int probe_dtm_table(const ChessBoard& pos, int won, int* success) {
    return probe_table(pos, won, success, DTM);
  }

  int probe_dtz_table(const ChessBoard& pos, int wdl, int* success) {
    return probe_table(pos, wdl, success, DTZ);
  }

 private:
  std::string name_for_tb(const char* str, const char* suffix) {
    std::stringstream path_string_stream(paths_);
    std::string path;
    std::ifstream stream;
    while (std::getline(path_string_stream, path, SEP_CHAR)) {
      std::string fname = path + "/" + str + suffix;
      stream.open(fname);
      if (stream.is_open()) return fname;
    }
    return std::string();
  }

  bool test_tb(const char* str, const char* suffix) {
    return !name_for_tb(str, suffix).empty();
  }

  void* map_tb(const char* name, const char* suffix, map_t* mapping) {
    std::string fname = name_for_tb(name, suffix);
    void* base_address;
#ifndef _WIN32
    struct stat statbuf;
    int fd = ::open(fname.c_str(), O_RDONLY);
    if (fd == -1) return nullptr;
    fstat(fd, &statbuf);
    if (statbuf.st_size % 64 != 16) {
      throw Exception("Corrupt tablebase file " + fname);
    }
    *mapping = statbuf.st_size;
    base_address = mmap(nullptr, statbuf.st_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (base_address == MAP_FAILED) {
      CERR << "Could not mmap() " << fname;
      exit(1);
    }
#else
    const HANDLE fd =
        CreateFileA(fname.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fd == INVALID_HANDLE_VALUE) return nullptr;
    DWORD size_high;
    DWORD size_low = GetFileSize(fd, &size_high);
    if (size_low % 64 != 16) {
      throw Exception("Corrupt tablebase file " + fname);
    }
    HANDLE mmap = CreateFileMapping(fd, nullptr, PAGE_READONLY, size_high,
                                    size_low, nullptr);
    CloseHandle(fd);
    if (!mmap) {
      CERR << "CreateFileMapping() failed";
      exit(1);
    }
    *mapping = mmap;
    base_address = MapViewOfFile(mmap, FILE_MAP_READ, 0, 0, 0);
    if (!base_address) {
      CERR << "MapViewOfFile() failed, name = " << fname
           << ", error = " << GetLastError();
      exit(1);
    }
#endif
    return base_address;
  }

  void unmap_file(void* base_address, map_t mapping) {
#ifndef _WIN32
    munmap(base_address, mapping);
#else
    UnmapViewOfFile(base_address);
    CloseHandle(mapping);
#endif
  }

  void add_to_hash(BaseEntry* ptr, Key key) {
    int idx;

    idx = key >> (64 - TB_HASHBITS);

    while (tb_hash_[idx].ptr) idx = (idx + 1) & ((1 << TB_HASHBITS) - 1);

    tb_hash_[idx].key = key;
    tb_hash_[idx].ptr = ptr;
  }

  void init_tb(char* str) {
    if (!test_tb(str, kSuffix[WDL])) return;

    int pcs[16];
    for (int i = 0; i < 16; i++) pcs[i] = 0;
    int color = 0;
    for (char* s = str; *s; s++) {
      if (*s == 'v') {
        color = 8;
      } else {
        for (int i = PAWN; i <= KING; i++) {
          if (*s == kPieceToChar[i]) {
            pcs[i | color]++;
            break;
          }
        }
      }
    }

    const Key key = calc_key_from_pcs(pcs, false);
    const Key key2 = calc_key_from_pcs(pcs, true);

    const bool has_pawns = pcs[W_PAWN] || pcs[B_PAWN];

    BaseEntry* be =
        has_pawns
            ? static_cast<BaseEntry*>(&pawn_entries_[num_pawn_entries_++])
            : static_cast<BaseEntry*>(&piece_entries_[num_piece_entries_++]);
    be->hasPawns = has_pawns;
    be->key = key;
    be->symmetric = key == key2;
    be->num = 0;
    for (int i = 0; i < 16; i++) be->num += pcs[i];

    num_wdl_++;
    num_dtm_ += be->hasDtm = test_tb(str, kSuffix[DTM]);
    num_dtz_ += be->hasDtz = test_tb(str, kSuffix[DTZ]);

    max_cardinality_ = std::max(max_cardinality_, static_cast<int>(be->num));
    if (be->hasDtm)
      max_cardinality_dtm_ =
          std::max(max_cardinality_dtm_, static_cast<int>(be->num));

    for (int type = 0; type < 3; type++) be->ready[type] = 0;

    if (!be->hasPawns) {
      int j = 0;
      for (int i = 0; i < 16; i++) {
        if (pcs[i] == 1) j++;
      }
      be->kk_enc = j == 2;
    } else {
      be->pawns[0] = pcs[W_PAWN];
      be->pawns[1] = pcs[B_PAWN];
      if (pcs[B_PAWN] && (!pcs[W_PAWN] || pcs[W_PAWN] > pcs[B_PAWN])) {
        Swap(be->pawns[0], be->pawns[1]);
      }
    }

    add_to_hash(be, key);
    if (key != key2) add_to_hash(be, key2);
  }

  void free_tb_entry(BaseEntry* be) {
    for (int type = 0; type < 3; type++) {
      if (atomic_load_explicit(&be->ready[type], std::memory_order_relaxed)) {
        unmap_file(be->data[type], be->mapping[type]);
        const int num = num_tables(be, type);
        EncInfo* ei = first_ei(be, type);
        for (int t = 0; t < num; t++) {
          free(ei[t].precomp);
          if (type != DTZ) free(ei[num + t].precomp);
        }
        atomic_store_explicit(&be->ready[type], false,
                              std::memory_order_relaxed);
      }
    }
  }

  bool init_table(BaseEntry* be, const char* str, int type) {
    uint8_t* data =
        static_cast<uint8_t*>(map_tb(str, kSuffix[type], &be->mapping[type]));
    if (!data) return false;

    if (read_le_u32(data) != kMagic[type]) {
      fprintf(stderr, "Corrupted table.\n");
      unmap_file(data, be->mapping[type]);
      return false;
    }

    be->data[type] = data;

    const bool split = type != DTZ && (data[4] & 0x01);
    if (type == DTM) be->dtmLossOnly = data[4] & 0x04;

    data += 5;

    size_t tb_size[6][2];
    const int num = num_tables(be, type);
    EncInfo* ei = first_ei(be, type);
    const int enc = !be->hasPawns ? PIECE_ENC : type != DTM ? FILE_ENC : RANK_ENC;

    for (int t = 0; t < num; t++) {
      tb_size[t][0] = init_enc_info(&ei[t], be, data, 0, t, enc);
      if (split) {
        tb_size[t][1] = init_enc_info(&ei[num + t], be, data, 4, t, enc);
      }
      data += be->num + 1 + (be->hasPawns && be->pawns[1]);
    }
    data += (uintptr_t)data & 1;

    size_t size[6][2][3];
    for (int t = 0; t < num; t++) {
      uint8_t flags;
      ei[t].precomp =
          setup_pairs(&data, tb_size[t][0], size[t][0], &flags, type);
      if (type == DTZ) {
        if (!be->hasPawns) {
          PIECE(be)->dtzFlags = flags;
        } else {
          PAWN(be)->dtzFlags[t] = flags;
        }
      }
      if (split) {
        ei[num + t].precomp =
            setup_pairs(&data, tb_size[t][1], size[t][1], &flags, type);
      } else if (type != DTZ) {
        ei[num + t].precomp = NULL;
      }
    }

    if (type == DTM && !be->dtmLossOnly) {
      uint16_t* map = reinterpret_cast<uint16_t*>(data);
      *(be->hasPawns ? &PAWN(be)->dtmMap : &PIECE(be)->dtmMap) = map;
      uint16_t(*mapIdx)[2][2] =
          be->hasPawns ? &PAWN(be)->dtmMapIdx[0] : &PIECE(be)->dtmMapIdx;
      for (int t = 0; t < num; t++) {
        for (int i = 0; i < 2; i++) {
          mapIdx[t][0][i] = reinterpret_cast<uint16_t*>(data) + 1 - map;
          data += 2 + 2 * read_le_u16(data);
        }
        if (split) {
          for (int i = 0; i < 2; i++) {
            mapIdx[t][1][i] = reinterpret_cast<uint16_t*>(data) + 1 - map;
            data += 2 + 2 * read_le_u16(data);
          }
        }
      }
    }

    if (type == DTZ) {
      void* map = data;
      *(be->hasPawns ? &PAWN(be)->dtzMap : &PIECE(be)->dtzMap) = map;
      uint16_t(*mapIdx)[4] =
          be->hasPawns ? &PAWN(be)->dtzMapIdx[0] : &PIECE(be)->dtzMapIdx;
      uint8_t* flags =
          be->hasPawns ? &PAWN(be)->dtzFlags[0] : &PIECE(be)->dtzFlags;
      for (int t = 0; t < num; t++) {
        if (flags[t] & 2) {
          if (!(flags[t] & 16)) {
            for (int i = 0; i < 4; i++) {
              mapIdx[t][i] = data + 1 - static_cast<uint8_t*>(map);
              data += 1 + data[0];
            }
          } else {
            data += reinterpret_cast<uintptr_t>(data) & 0x01;
            for (int i = 0; i < 4; i++) {
              mapIdx[t][i] = reinterpret_cast<uint16_t*>(data) + 1 -
                             static_cast<uint16_t*>(map);
              data += 2 + 2 * read_le_u16(data);
            }
          }
        }
      }
      data += reinterpret_cast<uintptr_t>(data) & 0x01;
    }

    for (int t = 0; t < num; t++) {
      ei[t].precomp->indexTable = data;
      data += size[t][0][0];
      if (split) {
        ei[num + t].precomp->indexTable = data;
        data += size[t][1][0];
      }
    }

    for (int t = 0; t < num; t++) {
      ei[t].precomp->sizeTable = reinterpret_cast<uint16_t*>(data);
      data += size[t][0][1];
      if (split) {
        ei[num + t].precomp->sizeTable = reinterpret_cast<uint16_t*>(data);
        data += size[t][1][1];
      }
    }

    for (int t = 0; t < num; t++) {
      data = reinterpret_cast<uint8_t*>(
          (reinterpret_cast<uintptr_t>(data) + 0x3f) & ~0x3f);
      ei[t].precomp->data = data;
      data += size[t][0][2];
      if (split) {
        data = reinterpret_cast<uint8_t*>(
            (reinterpret_cast<uintptr_t>(data) + 0x3f) & ~0x3f);
        ei[num + t].precomp->data = data;
        data += size[t][1][2];
      }
    }

    if (type == DTM && be->hasPawns) {
      PAWN(be)->dtmSwitched =
          calc_key_from_pieces(ei[0].pieces, be->num) != be->key;
    }

    return true;
  }

  int probe_table(const ChessBoard& pos, int s, int* success, const int type) {
    // Obtain the position's material-signature key
    const Key key = calc_key_from_position(pos);

    // Test for KvK
    if (type == WDL && pos.ours() == pos.our_king() &&
        pos.theirs() == pos.their_king()) {
      return 0;
    }

    int hash_idx = key >> (64 - TB_HASHBITS);
    while (tb_hash_[hash_idx].key && tb_hash_[hash_idx].key != key) {
      hash_idx = (hash_idx + 1) & ((1 << TB_HASHBITS) - 1);
    }
    if (!tb_hash_[hash_idx].ptr) {
      *success = 0;
      return 0;
    }

    BaseEntry* be = tb_hash_[hash_idx].ptr;
    if ((type == DTM && !be->hasDtm) || (type == DTZ && !be->hasDtz)) {
      *success = 0;
      return 0;
    }

    // Use double-checked locking to reduce locking overhead
    if (!atomic_load_explicit(&be->ready[type], std::memory_order_acquire)) {
      Mutex::Lock lock(ready_mutex_);
      if (!atomic_load_explicit(&be->ready[type], std::memory_order_relaxed)) {
        char str[16];
        prt_str(pos, str, be->key != key);
        if (!init_table(be, str, type)) {
          tb_hash_[hash_idx].ptr = nullptr;  // mark as deleted
          *success = 0;
          return 0;
        }
        atomic_store_explicit(&be->ready[type], true,
                              std::memory_order_release);
      }
    }

    bool bside, flip;
    if (!be->symmetric) {
      flip = key != be->key;
      bside = (!pos.flipped()) == flip;
      if (type == DTM && be->hasPawns && PAWN(be)->dtmSwitched) {
        flip = !flip;
        bside = !bside;
      }
    } else {
      flip = pos.flipped();
      bside = false;
    }

    EncInfo* ei = first_ei(be, type);
    int p[TB_PIECES];
    size_t idx;
    int t = 0;
    uint8_t flags = 0;

    if (!be->hasPawns) {
      if (type == DTZ) {
        flags = PIECE(be)->dtzFlags;
        if ((flags & 1) != bside && !be->symmetric) {
          *success = -1;
          return 0;
        }
      }
      ei = type != DTZ ? &ei[bside] : ei;
      for (int i = 0; i < be->num;) {
        i = fill_squares(pos, ei->pieces, flip, 0, p, i);
      }
      idx = encode_piece(p, ei, be);
    } else {
      int i = fill_squares(pos, ei->pieces, flip, flip ? 0x38 : 0, p, 0);
      t = leading_pawn(p, be, type != DTM ? FILE_ENC : RANK_ENC);
      if (type == DTZ) {
        flags = PAWN(be)->dtzFlags[t];
        if ((flags & 1) != bside && !be->symmetric) {
          *success = -1;
          return 0;
        }
      }
      ei = type == WDL ? &ei[t + 4 * bside]
                       : type == DTM ? &ei[t + 6 * bside] : &ei[t];
      while (i < be->num) {
        i = fill_squares(pos, ei->pieces, flip, flip ? 0x38 : 0, p, i);
      }
      idx = type != DTM ? encode_pawn_f(p, ei, be) : encode_pawn_r(p, ei, be);
    }

    uint8_t* w = decompress_pairs(ei->precomp, idx);

    if (type == WDL) return static_cast<int>(w[0]) - 2;

    int v = w[0] + ((w[1] & 0x0f) << 8);

    if (type == DTM) {
      if (!be->dtmLossOnly) {
        v = from_le_u16(
            be->hasPawns
                ? PAWN(be)->dtmMap[PAWN(be)->dtmMapIdx[t][bside][s] + v]
                : PIECE(be)->dtmMap[PIECE(be)->dtmMapIdx[bside][s] + v]);
      }
    } else {
      if (flags & 2) {
        const int m = kWdlToMap[s + 2];
        if (!(flags & 16)) {
          v = be->hasPawns
                  ? static_cast<uint8_t*>(
                        PAWN(be)->dtzMap)[PAWN(be)->dtzMapIdx[t][m] + v]
                  : static_cast<uint8_t*>(
                        PIECE(be)->dtzMap)[PIECE(be)->dtzMapIdx[m] + v];
        } else {
          v = from_le_u16(
              be->hasPawns
                  ? static_cast<uint16_t*>(
                        PAWN(be)->dtzMap)[PAWN(be)->dtzMapIdx[t][m] + v]
                  : static_cast<uint16_t*>(
                        PIECE(be)->dtzMap)[PIECE(be)->dtzMapIdx[m] + v]);
        }
      }
      if (!(flags & kPAFlags[s + 2]) || (s & 1)) v *= 2;
    }

    return v;
  }

  int max_cardinality_ = 0;
  int max_cardinality_dtm_ = 0;

  Mutex ready_mutex_;
  std::string paths_;

  int num_piece_entries_ = 0;
  int num_pawn_entries_ = 0;
  int num_wdl_ = 0;
  int num_dtm_ = 0;
  int num_dtz_ = 0;

  std::vector<PieceEntry> piece_entries_;
  std::vector<PawnEntry> pawn_entries_;
  std::vector<TbHashEntry> tb_hash_;
};

SyzygyTablebase::SyzygyTablebase() : max_cardinality_(0) {}

SyzygyTablebase::~SyzygyTablebase() = default;

bool SyzygyTablebase::init(const std::string& paths) {
  paths_ = paths;
  impl_.reset(new SyzygyTablebaseImpl(paths_));
  max_cardinality_ = impl_->max_cardinality();
  if (max_cardinality_ <= 2) {
    impl_ = nullptr;
    return false;
  }
  return true;
}

// For a position where the side to move has a winning capture it is not
// necessary to store a winning value so the generator treats such positions as
// "don't cares" and tries to assign to it a value that improves the compression
// ratio. Similarly, if the side to move has a drawing capture, then the
// position is at least drawn. If the position is won, then the TB needs to
// store a win value. But if the position is drawn, the TB may store a loss
// value if that is better for compression. All of this means that during
// probing, the engine must look at captures and probe their results and must
// probe the position itself. The "best" result of these probes is the correct
// result for the position. DTZ table don't store values when a following move
// is a zeroing winning move (winning capture or winning pawn move). Also DTZ
// store wrong values for positions where the best move is an ep-move (even if
// losing). So in all these cases set the state to ZEROING_BEST_MOVE.
template <bool CheckZeroingMoves>
WDLScore SyzygyTablebase::search(const Position& pos, ProbeState* result) {
  WDLScore value;
  WDLScore best_value = WDL_LOSS;
  auto move_list = pos.GetBoard().GenerateLegalMoves();
  const size_t total_count = move_list.size();
  size_t move_count = 0;
  for (const Move& move : move_list) {
    if (!is_capture(pos.GetBoard(), move) &&
        (!CheckZeroingMoves || !pos.GetBoard().pawns().get(move.from()))) {
      continue;
    }
    move_count++;
    auto new_pos = Position(pos, move);
    value = static_cast<WDLScore>(-search(new_pos, result));
    if (*result == FAIL) return WDL_DRAW;
    if (value > best_value) {
      best_value = value;
      if (value >= WDL_WIN) {
        *result = ZEROING_BEST_MOVE;  // Winning DTZ-zeroing move
        return value;
      }
    }
  }
  // In case we have already searched all the legal moves we don't have to probe
  // the TB because the stored score could be wrong. For instance TB tables do
  // not contain information on position with ep rights, so in this case the
  // result of probe_wdl_table is wrong. Also in case of only capture moves, for
  // instance here 4K3/4q3/6p1/2k5/6p1/8/8/8 w - - 0 7, we have to return with
  // ZEROING_BEST_MOVE set.
  const bool no_more_moves = (move_count && move_count == total_count);
  if (no_more_moves) {
    value = best_value;
  } else {
    int raw_result = static_cast<int>(ProbeState::OK);
    value = static_cast<WDLScore>(
        impl_->probe_wdl_table(pos.GetBoard(), &raw_result));
    *result = static_cast<ProbeState>(raw_result);
    if (*result == FAIL) return WDL_DRAW;
  }
  // DTZ stores a "don't care" value if bestValue is a win
  if (best_value >= value) {
    *result = (best_value > WDL_DRAW || no_more_moves ? ZEROING_BEST_MOVE : OK);
    return best_value;
  }
  *result = OK;
  return value;
}

// Probe the WDL table for a particular position.
// If *result != FAIL, the probe was successful.
// The return value is from the point of view of the side to move:
// -2 : loss
// -1 : loss, but draw under 50-move rule
//  0 : draw
//  1 : win, but draw under 50-move rule
//  2 : win
WDLScore SyzygyTablebase::probe_wdl(const Position& pos, ProbeState* result) {
  *result = OK;
  return search(pos, result);
}

// Probe the DTZ table for a particular position.
// If *result != FAIL, the probe was successful.
// The return value is from the point of view of the side to move:
//         n < -100 : loss, but draw under 50-move rule
// -100 <= n < -1   : loss in n ply (assuming 50-move counter == 0)
//        -1        : loss, the side to move is mated
//         0        : draw
//     1 < n <= 100 : win in n ply (assuming 50-move counter == 0)
//   100 < n        : win, but draw under 50-move rule
//
// The return value n can be off by 1: a return value -n can mean a loss  in n+1
// ply and a return value +n can mean a win in n+1 ply. This cannot happen for
// tables with positions exactly on the "edge" of the 50-move rule.
//
// This implies that if dtz > 0 is returned, the position is certainly a win if
// dtz + 50-move-counter <= 99. Care must be taken that the engine picks moves
// that preserve dtz + 50-move-counter <= 99.
//
// If n = 100 immediately after a capture or pawn move, then the position is
// also certainly a win, and during the whole phase until the next capture or
// pawn move, the inequality to be preserved is dtz
// + 50-movecounter <= 100.
//
// In short, if a move is available resulting in dtz + 50-move-counter <= 99,
// then do not accept moves leading to dtz + 50-move-counter == 100.
int SyzygyTablebase::probe_dtz(const Position& pos, ProbeState* result) {
  *result = OK;
  const WDLScore wdl = search<true>(pos, result);
  if (*result == FAIL || wdl == WDL_DRAW) {  // DTZ tables don't store draws
    return 0;
  }
  // DTZ stores a 'don't care' value in this case, or even a plain wrong one as
  // in case the best move is a losing ep, so it cannot be probed.
  if (*result == ZEROING_BEST_MOVE) return dtz_before_zeroing(wdl);
  int raw_result = 1;
  int dtz = impl_->probe_dtz_table(pos.GetBoard(), wdl, &raw_result);
  *result = static_cast<ProbeState>(raw_result);
  if (*result == FAIL) return 0;
  if (*result != CHANGE_STM) {
    return (dtz + 1 +
            100 * (wdl == WDL_BLESSED_LOSS || wdl == WDL_CURSED_WIN)) *
           sign_of(wdl);
  }
  // DTZ stores results for the other side, so we need to do a 1-ply search and
  // find the winning move that minimizes DTZ.
  int min_DTZ = 0xFFFF;
  for (const Move& move : pos.GetBoard().GenerateLegalMoves()) {
    Position next_pos = Position(pos, move);
    const bool zeroing = next_pos.GetNoCaptureNoPawnPly() == 0;
    // For zeroing moves we want the dtz of the move _before_ doing it,
    // otherwise we will get the dtz of the next move sequence. Search the
    // position after the move to get the score sign (because even in a winning
    // position we could make a losing capture or going for a draw).
    dtz = zeroing ? -dtz_before_zeroing(search(next_pos, result))
                  : -probe_dtz(next_pos, result);
    // If the move mates, force minDTZ to 1
    if (dtz == 1 && next_pos.GetBoard().IsUnderCheck() &&
        next_pos.GetBoard().GenerateLegalMoves().empty()) {
      min_DTZ = 1;
    }
    // Convert result from 1-ply search. Zeroing moves are already accounted by
    // dtz_before_zeroing() that returns the DTZ of the previous move.
    if (!zeroing) dtz += sign_of(dtz);
    // Skip the draws and if we are winning only pick positive dtz
    if (dtz < min_DTZ && sign_of(dtz) == sign_of(wdl)) min_DTZ = dtz;
    if (*result == FAIL) return 0;
  }
  // When there are no legal moves, the position is mate: we return -1
  return min_DTZ == 0xFFFF ? -1 : min_DTZ;
}

// Use the DTZ tables to rank root moves.
//
// A return value false indicates that not all probes were successful.
bool SyzygyTablebase::root_probe(const Position& pos, bool has_repeated,
                                 std::vector<Move>* safe_moves) {
  ProbeState result;
  auto root_moves = pos.GetBoard().GenerateLegalMoves();
  // Obtain 50-move counter for the root position
  const int cnt50 = pos.GetNoCaptureNoPawnPly();
  // Check whether a position was repeated since the last zeroing move.
  const bool rep = has_repeated;
  int dtz;
  std::vector<int> ranks;
  ranks.reserve(root_moves.size());
  int best_rank = -1000;
  // Probe and rank each move
  for (auto& m : root_moves) {
    Position next_pos = Position(pos, m);
    // Calculate dtz for the current move counting from the root position
    if (next_pos.GetNoCaptureNoPawnPly() == 0) {
      // In case of a zeroing move, dtz is one of -101/-1/0/1/101
      const WDLScore wdl = static_cast<WDLScore>(-probe_wdl(next_pos, &result));
      dtz = dtz_before_zeroing(wdl);
    } else {
      // Otherwise, take dtz for the new position and correct by 1 ply
      dtz = -probe_dtz(next_pos, &result);
      dtz = dtz > 0 ? dtz + 1 : dtz < 0 ? dtz - 1 : dtz;
    }
    // Make sure that a mating move is assigned a dtz value of 1
    if (next_pos.GetBoard().IsUnderCheck() && dtz == 2 &&
        next_pos.GetBoard().GenerateLegalMoves().size() == 0) {
      dtz = 1;
    }
    if (result == FAIL) return false;
    // Better moves are ranked higher. Certain wins are ranked equally.
    // Losing moves are ranked equally unless a 50-move draw is in sight.
    int r = dtz > 0
                ? (dtz + cnt50 <= 99 && !rep ? 1000 : 1000 - (dtz + cnt50))
                : dtz < 0 ? (-dtz * 2 + cnt50 < 100 ? -1000
                                                    : -1000 + (-dtz + cnt50))
                          : 0;
    if (r > best_rank) best_rank = r;
    ranks.push_back(r);
  }
  // Disable all but the equal best moves.
  int counter = 0;
  for (auto& m : root_moves) {
    if (ranks[counter] == best_rank) {
      safe_moves->push_back(m);
    }
    counter++;
  }
  return true;
}

// Use the WDL tables to rank root moves.
// This is a fallback for the case that some or all DTZ tables are missing.
//
// A return value false indicates that not all probes were successful.
bool SyzygyTablebase::root_probe_wdl(const Position& pos,
                                     std::vector<Move>* safe_moves) {
  static const int WDL_to_rank[] = {-1000, -899, 0, 899, 1000};
  auto root_moves = pos.GetBoard().GenerateLegalMoves();
  ProbeState result;
  std::vector<int> ranks;
  ranks.reserve(root_moves.size());
  int best_rank = -1000;
  // Probe and rank each move
  for (auto& m : root_moves) {
    Position nextPos = Position(pos, m);
    const WDLScore wdl = static_cast<WDLScore>(-probe_wdl(nextPos, &result));
    if (result == FAIL) return false;
    ranks.push_back(WDL_to_rank[wdl + 2]);
    if (ranks.back() > best_rank) best_rank = ranks.back();
  }
  // Disable all but the equal best moves.
  int counter = 0;
  for (auto& m : root_moves) {
    if (ranks[counter] == best_rank) {
      safe_moves->push_back(m);
    }
    counter++;
  }
  return true;
}
}  // namespace lczero

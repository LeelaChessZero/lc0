/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021-2026 The LCZero Authors

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

// The trivial backend: a very fast CPU-only handcrafted evaluator. It is meant
// to never limit search nps while still producing reasonably realistic trees.
//
// Value: tapered PeSTO material+PST, mobility, simple pawn structure and king
// safety, plus an estimate of pending captures and threats (a cheap substitute
// for a quiescence search). Exchanges are estimated from attack maps, with
// static exchange evaluation (SEE) only for squares attacked more than once.
//
// Policy: per legal move, the SEE material outcome of the move, the PST
// improvement, escaping threats, checks, new threats and castling, turned into
// probabilities with a softmax.
//
// Weights were tuned in self-play at 1000 nodes; they can be overridden with
// --backend-opts (see Params) for further tuning.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

#include "chess/board.h"
#include "neural/backend.h"
#include "neural/register.h"
#include "neural/shared_params.h"
#include "utils/bititer.h"
#include "utils/fastmath.h"

namespace lczero {
namespace {

using u64 = uint64_t;

// Own piece indexing (differs from PieceType).
enum : int { P = 0, N = 1, B = 2, R = 3, Q = 4, K = 5, kNone = 7 };

constexpr u64 kFileAMask = 0x0101010101010101ULL;
constexpr u64 kFileHMask = 0x8080808080808080ULL;

constexpr int kSeeVal[8] = {100, 320, 330, 500, 950, 20000, 0, 0};
constexpr int kMgVal[6] = {82, 337, 365, 477, 1025, 0};
constexpr int kEgVal[6] = {94, 281, 297, 512, 936, 0};

// PeSTO piece-square tables (Rofchade), from white's point of view, a8 first.
// clang-format off
constexpr int kPestoMg[6][64] = {
  {   0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0},
  {-167, -89, -34, -49,  61, -97, -15, -107,
    -73, -41,  72,  36,  23,  62,   7,  -17,
    -47,  60,  37,  65,  84, 129,  73,   44,
     -9,  17,  19,  53,  37,  69,  18,   22,
    -13,   4,  16,  13,  28,  19,  21,   -8,
    -23,  -9,  12,  10,  19,  17,  25,  -16,
    -29, -53, -12,  -3,  -1,  18, -14,  -19,
   -105, -21, -58, -33, -17, -28, -19,  -23},
  { -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21},
  {  32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26},
  { -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50},
  { -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14},
};
constexpr int kPestoEg[6][64] = {
  {   0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0},
  { -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64},
  { -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17},
  {  13, 10, 18, 15, 12,  12,   8,   5,
     11, 13, 13, 11, -3,   3,   8,   3,
      7,  7,  7,  5,  4,  -3,  -5,  -3,
      4,  3, 13,  1,  2,   1,  -1,   2,
      3,  5,  8,  4, -5,  -6,  -8, -11,
     -4,  0, -5, -1, -7, -12,  -8, -16,
     -6, -6,  0,  2, -9,  -9, -11,  -3,
     -9,  2,  3, -1, -5, -13,   4, -20},
  {  -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41},
  { -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43},
};
// clang-format on

struct Tables {
  // Material + PST, indexed by square from the side-to-move point of view
  // (a1 = 0). For the opponent's pieces, index with sq ^ 56.
  int16_t mg[6][64];
  int16_t eg[6][64];
  u64 knight[64];
  u64 king[64];
  u64 pawn_up[64];    // Squares attacked by "our" pawn standing on the square.
  u64 pawn_down[64];  // Squares attacked by "their" pawn standing on it.
  u64 king_zone[64];
};

constexpr Tables MakeTables() {
  Tables t{};
  for (int pt = 0; pt < 6; ++pt) {
    for (int sq = 0; sq < 64; ++sq) {
      t.mg[pt][sq] = kMgVal[pt] + kPestoMg[pt][sq ^ 56];
      t.eg[pt][sq] = kEgVal[pt] + kPestoEg[pt][sq ^ 56];
    }
  }
  auto bb = [](int f, int r) -> u64 {
    if (f < 0 || f > 7 || r < 0 || r > 7) return 0;
    return 1ULL << (r * 8 + f);
  };
  for (int sq = 0; sq < 64; ++sq) {
    const int f = sq % 8, r = sq / 8;
    t.knight[sq] = bb(f + 1, r + 2) | bb(f - 1, r + 2) | bb(f + 2, r + 1) |
                   bb(f - 2, r + 1) | bb(f + 1, r - 2) | bb(f - 1, r - 2) |
                   bb(f + 2, r - 1) | bb(f - 2, r - 1);
    t.king[sq] = bb(f + 1, r) | bb(f - 1, r) | bb(f, r + 1) | bb(f, r - 1) |
                 bb(f + 1, r + 1) | bb(f - 1, r + 1) | bb(f + 1, r - 1) |
                 bb(f - 1, r - 1);
    t.pawn_up[sq] = bb(f + 1, r + 1) | bb(f - 1, r + 1);
    t.pawn_down[sq] = bb(f + 1, r - 1) | bb(f - 1, r - 1);
    t.king_zone[sq] = t.king[sq] | (1ULL << sq);
  }
  return t;
}

constexpr Tables kT = MakeTables();

// Material+PST tapered for each game phase (0..24), used by the policy.
struct TaperedPst {
  int16_t v[25][6][64];
};
constexpr TaperedPst MakeTaperedPst() {
  TaperedPst t{};
  for (int ph = 0; ph <= 24; ++ph) {
    for (int pt = 0; pt < 6; ++pt) {
      for (int sq = 0; sq < 64; ++sq) {
        t.v[ph][pt][sq] = (kT.mg[pt][sq] * ph + kT.eg[pt][sq] * (24 - ph)) / 24;
      }
    }
  }
  return t;
}
constexpr TaperedPst kTaperedPst = MakeTaperedPst();

inline u64 Bit(int sq) { return 1ULL << sq; }
inline int Lsb(u64 x) { return std::countr_zero(x); }
inline int PopCount(u64 x) { return std::popcount(x); }
inline u64 FillUp(u64 b) {
  b |= b << 8;
  b |= b << 16;
  return b | (b << 32);
}
inline u64 FillDown(u64 b) {
  b |= b >> 8;
  b |= b >> 16;
  return b | (b >> 32);
}
inline u64 ShiftSides(u64 b) {
  return ((b & ~kFileHMask) << 1) | ((b & ~kFileAMask) >> 1);
}
inline u64 RookAtt(int sq, u64 occ) {
  return GetRookAttacks(Square::FromIdx(sq), BitBoard(occ)).as_int();
}
inline u64 BishopAtt(int sq, u64 occ) {
  return GetBishopAttacks(Square::FromIdx(sq), BitBoard(occ)).as_int();
}
inline u64 OurPawnAttacks(u64 p) {
  return ((p & ~kFileAMask) << 7) | ((p & ~kFileHMask) << 9);
}
inline u64 TheirPawnAttacks(u64 p) {
  return ((p & ~kFileAMask) >> 9) | ((p & ~kFileHMask) >> 7);
}

// Tunable parameters. Can be overridden through --backend-opts for tuning.
struct Params {
  // Value.
  float cp_scale = 250.0f;  // q = atan(cp / cp_scale) / 1.5637...
  int tempo = 15;
  int capture_pct = 90;          // Share of our best capture counted.
  int threat_pct = 110;          // Share of the second biggest threat counted.
  int mob_mg[4] = {4, 5, 2, 1};  // N, B, R, Q per attacked square.
  int mob_eg[4] = {4, 5, 4, 2};
  int king_attack = 7;  // Per attacker weight unit on king zone.
  int passed_mg = 10;
  int passed_eg = 20;
  int isolated = 10;
  int doubled = 10;
  int bishop_pair = 30;
  // Policy.
  float policy_temp = 140.0f;  // cp per logit unit for material outcomes.
  float pst_weight = 2.0f;     // Multiplier for PST delta (positional).
  int check_bonus = 250;
  int castle_bonus = 80;
  int escape_pct = 35;
  int threat_bonus = 120;  // Creating a threat on a more valuable piece.
};

struct Pos {
  u64 us, them, occ;
  u64 pc[6];  // Both colors.
  int our_king, their_king;
  int8_t type[64];
};

void Fill(const ChessBoard& cb, Pos* p) {
  p->us = cb.ours().as_int();
  p->them = cb.theirs().as_int();
  p->occ = p->us | p->them;
  p->pc[P] = cb.pawns().as_int();
  p->pc[N] = cb.knights().as_int();
  p->pc[B] = cb.bishops().as_int();
  p->pc[R] = cb.rooks().as_int();
  p->pc[Q] = cb.queens().as_int();
  p->pc[K] = cb.kings().as_int();
  p->our_king = Lsb(p->pc[K] & p->us);
  p->their_king = Lsb(p->pc[K] & p->them);
  std::memset(p->type, kNone, sizeof(p->type));
  for (int t = 0; t < 6; ++t) {
    for (u64 x = p->pc[t]; x; x &= x - 1) p->type[Lsb(x)] = t;
  }
}

inline u64 AttackersTo(const Pos& p, int sq, u64 occ) {
  return (kT.pawn_down[sq] & p.pc[P] & p.us) |
         (kT.pawn_up[sq] & p.pc[P] & p.them) | (kT.knight[sq] & p.pc[N]) |
         (kT.king[sq] & p.pc[K]) | (BishopAtt(sq, occ) & (p.pc[B] | p.pc[Q])) |
         (RookAtt(sq, occ) & (p.pc[R] | p.pc[Q]));
}

// Static exchange evaluation of a piece of type `attacker` moving from `from`
// to `to` (capturing `captured_val`, may be 0). `mover` is the bitboard of the
// moving side, `other` of the opponent.
// `att` are all attackers of `to` with the current occupancy.
int See(const Pos& p, int from, int to, int attacker, int captured_val,
        u64 mover, u64 other, u64 att) {
  int gain[34];
  int d = 0;
  u64 occ = p.occ ^ Bit(from);
  gain[0] = captured_val;
  int cur_val = kSeeVal[attacker];
  const u64 diag = p.pc[B] | p.pc[Q];
  const u64 orth = p.pc[R] | p.pc[Q];
  // X-rays through the vacated square.
  if (attacker == P || attacker == B || attacker == Q) {
    att |= BishopAtt(to, occ) & diag;
  }
  if (attacker == R || attacker == Q) att |= RookAtt(to, occ) & orth;
  att &= occ;
  const u64 sides[2] = {other, mover};
  int s = 0;
  while (d < 32) {
    const u64 mine = att & sides[s];
    if (!mine) break;
    int t = P;
    u64 bb = mine & p.pc[P];
    while (!bb) bb = mine & p.pc[++t];
    ++d;
    gain[d] = cur_val - gain[d - 1];
    if (std::max(-gain[d - 1], gain[d]) < 0) break;
    cur_val = kSeeVal[t];
    occ ^= bb & (0 - bb);
    if (t == P || t == B || t == Q) att |= BishopAtt(to, occ) & diag;
    if (t == R || t == Q) att |= RookAtt(to, occ) & orth;
    att &= occ;
    s ^= 1;
  }
  for (int i = d; i > 0; --i) gain[i - 1] = -std::max(-gain[i - 1], gain[i]);
  return gain[0];
}

// Best gain for `mover` capturing on `sq` with its least valuable attacker.
inline int ExchangeValue(const Pos& p, int sq, int victim, u64 mover,
                         u64 other) {
  const u64 all = AttackersTo(p, sq, p.occ);
  const u64 attackers = all & mover;
  for (int t = P; t <= K; ++t) {
    if (u64 bb = attackers & p.pc[t]) {
      return See(p, Lsb(bb), sq, t, victim, mover, other, all);
    }
  }
  return 0;
}

// Information computed during evaluation that is reused for the policy.
struct EvalInfo {
  int phase;
  u64 att_us;    // Squares attacked by us.
  u64 att_them;  // Squares attacked by them.
  u64 att2_us;   // Squares attacked by us at least twice.
  u64 att2_them;
  u64 by_us[6];  // Squares attacked by our pieces of a given type.
  u64 by_them[6];
  int threat[64];  // Estimated gain of the opponent capturing our piece there.
};

// Value of the least valuable attacker of `sq` (0 if none).
inline int LvaValue(const u64 by[6], int sq) {
  for (int t = P; t <= K; ++t) {
    if (by[t] & Bit(sq)) return kSeeVal[t];
  }
  return 0;
}

inline int Taper(int mg, int eg, int phase) {
  return (mg * phase + eg * (24 - phase)) / 24;
}

int EvaluatePosition(const Pos& p, const Params& w, EvalInfo* info) {
  // Pawns and kings; other pieces' PST is added in the mobility loop.
  int mg = kT.mg[K][p.our_king] - kT.mg[K][p.their_king ^ 56];
  int eg = kT.eg[K][p.our_king] - kT.eg[K][p.their_king ^ 56];
  for (u64 x = p.pc[P] & p.us; x; x &= x - 1) {
    const int sq = Lsb(x);
    mg += kT.mg[P][sq];
    eg += kT.eg[P][sq];
  }
  for (u64 x = p.pc[P] & p.them; x; x &= x - 1) {
    const int sq = Lsb(x) ^ 56;
    mg -= kT.mg[P][sq];
    eg -= kT.eg[P][sq];
  }
  const int phase =
      std::min(24, PopCount(p.pc[N] | p.pc[B]) + 2 * PopCount(p.pc[R]) +
                       4 * PopCount(p.pc[Q]));
  info->phase = phase;

  const u64 our_pawns = p.pc[P] & p.us;
  const u64 their_pawns = p.pc[P] & p.them;
  const u64 pawn_att_us = OurPawnAttacks(our_pawns);
  const u64 pawn_att_them = TheirPawnAttacks(their_pawns);
  info->by_us[P] = pawn_att_us;
  info->by_them[P] = pawn_att_them;
  info->by_us[K] = kT.king[p.our_king];
  info->by_them[K] = kT.king[p.their_king];
  for (int t = N; t <= Q; ++t) info->by_us[t] = info->by_them[t] = 0;
  u64 att_us = pawn_att_us | kT.king[p.our_king];
  u64 att_them = pawn_att_them | kT.king[p.their_king];
  // Squares attacked at least twice.
  u64 att2_us =
      (((our_pawns & ~kFileAMask) << 7) & ((our_pawns & ~kFileHMask) << 9)) |
      (pawn_att_us & kT.king[p.our_king]);
  u64 att2_them = (((their_pawns & ~kFileAMask) >> 9) &
                   ((their_pawns & ~kFileHMask) >> 7)) |
                  (pawn_att_them & kT.king[p.their_king]);

  // Mobility and king attacks.
  const u64 zone_them = kT.king_zone[p.their_king];
  const u64 zone_us = kT.king_zone[p.our_king];
  int king_att_us = 0, king_att_them = 0;  // Attack units on enemy king.
  int nattackers_us = 0, nattackers_them = 0;
  {
    const u64 safe = ~p.us & ~pawn_att_them;
    for (int t = N; t <= Q; ++t) {
      for (u64 x = p.pc[t] & p.us; x; x &= x - 1) {
        const int sq = Lsb(x);
        mg += kT.mg[t][sq];
        eg += kT.eg[t][sq];
        u64 a;
        if (t == N) {
          a = kT.knight[sq];
        } else if (t == B) {
          a = BishopAtt(sq, p.occ);
        } else if (t == R) {
          a = RookAtt(sq, p.occ);
        } else {
          a = BishopAtt(sq, p.occ) | RookAtt(sq, p.occ);
        }
        att2_us |= att_us & a;
        att_us |= a;
        info->by_us[t] |= a;
        const int m = PopCount(a & safe);
        mg += w.mob_mg[t - N] * (m - 4);
        eg += w.mob_eg[t - N] * (m - 4);
        if (a & zone_them) {
          ++nattackers_us;
          king_att_us += PopCount(a & zone_them) * (t == Q ? 2 : 1);
        }
      }
    }
  }
  {
    const u64 safe = ~p.them & ~pawn_att_us;
    for (int t = N; t <= Q; ++t) {
      for (u64 x = p.pc[t] & p.them; x; x &= x - 1) {
        const int sq = Lsb(x);
        mg -= kT.mg[t][sq ^ 56];
        eg -= kT.eg[t][sq ^ 56];
        u64 a;
        if (t == N) {
          a = kT.knight[sq];
        } else if (t == B) {
          a = BishopAtt(sq, p.occ);
        } else if (t == R) {
          a = RookAtt(sq, p.occ);
        } else {
          a = BishopAtt(sq, p.occ) | RookAtt(sq, p.occ);
        }
        att2_them |= att_them & a;
        att_them |= a;
        info->by_them[t] |= a;
        const int m = PopCount(a & safe);
        mg -= w.mob_mg[t - N] * (m - 4);
        eg -= w.mob_eg[t - N] * (m - 4);
        if (a & zone_us) {
          ++nattackers_them;
          king_att_them += PopCount(a & zone_us) * (t == Q ? 2 : 1);
        }
      }
    }
  }
  info->att_us = att_us;
  info->att_them = att_them;
  info->att2_us = att2_us;
  info->att2_them = att2_them;
  // King safety: grows quadratically with the number of attackers.
  if (nattackers_us >= 2) {
    mg += w.king_attack * king_att_us * nattackers_us / 2;
  }
  if (nattackers_them >= 2) {
    mg -= w.king_attack * king_att_them * nattackers_them / 2;
  }

  // Pawn structure.
  {
    const u64 our_span = FillUp(our_pawns << 8);
    const u64 their_span = FillDown(their_pawns >> 8);
    const u64 our_block = our_span | ShiftSides(our_span);
    const u64 their_block = their_span | ShiftSides(their_span);
    int passed_us = 0, passed_them = 0;
    for (u64 x = our_pawns & ~their_block; x; x &= x - 1) {
      passed_us += Lsb(x) >> 3;
    }
    for (u64 x = their_pawns & ~our_block; x; x &= x - 1) {
      passed_them += 7 - (Lsb(x) >> 3);
    }
    mg += w.passed_mg * (passed_us - passed_them) / 2;
    eg += w.passed_eg * (passed_us - passed_them) / 2;
    const u64 files_us = FillDown(FillUp(our_pawns)) & 0xFF;
    const u64 files_them = FillDown(FillUp(their_pawns)) & 0xFF;
    const u64 iso_us = files_us & ~((files_us << 1) | (files_us >> 1));
    const u64 iso_them = files_them & ~((files_them << 1) | (files_them >> 1));
    const int isolated = PopCount(our_pawns & FillUp(iso_us)) -
                         PopCount(their_pawns & FillUp(iso_them));
    const int doubled = (PopCount(our_pawns) - PopCount(files_us)) -
                        (PopCount(their_pawns) - PopCount(files_them));
    const int pawn_score = -w.isolated * isolated - w.doubled * doubled;
    mg += pawn_score;
    eg += pawn_score;
  }
  if (PopCount(p.pc[B] & p.us) >= 2) {
    mg += w.bishop_pair;
    eg += w.bishop_pair;
  }
  if (PopCount(p.pc[B] & p.them) >= 2) {
    mg -= w.bishop_pair;
    eg -= w.bishop_pair;
  }

  int cp = Taper(mg, eg, phase) + w.tempo;

  // Pending captures and threats. With a single attacker (or no defender),
  // the exchange value follows from the attack maps; otherwise run SEE.
  const bool in_check = att_them & Bit(p.our_king);
  int best_capture = 0;
  for (int vt = Q; vt >= P; --vt) {
    const int victim = kSeeVal[vt];
    if (victim <= best_capture) break;
    for (u64 x = p.them & att_us & p.pc[vt]; x; x &= x - 1) {
      const int sq = Lsb(x);
      int gain = victim;
      if (att_them & Bit(sq)) {
        gain = victim - LvaValue(info->by_us, sq);
        if (gain < victim && (att2_us & Bit(sq))) {
          gain = ExchangeValue(p, sq, victim, p.us, p.them);
        }
      }
      best_capture = std::max(best_capture, gain);
    }
  }
  int t1 = 0, t2 = 0;
  std::memset(info->threat, 0, sizeof(info->threat));
  for (int vt = Q; vt >= P; --vt) {
    const int victim = kSeeVal[vt];
    for (u64 x = p.us & att_them & p.pc[vt]; x; x &= x - 1) {
      const int sq = Lsb(x);
      int gain = victim;
      if (att_us & Bit(sq)) {
        gain = victim - LvaValue(info->by_them, sq);
        // Exact value only matters for the two biggest threats.
        if (victim > t2 && (att2_them & Bit(sq))) {
          gain = ExchangeValue(p, sq, victim, p.them, p.us);
        }
      }
      if (gain <= 0) continue;
      info->threat[sq] = gain;
      if (gain > t1) {
        t2 = t1;
        t1 = gain;
      } else if (gain > t2) {
        t2 = gain;
      }
    }
  }
  if (in_check) best_capture /= 2;
  const int with_capture = best_capture * w.capture_pct / 100 - t1;
  const int without = -t2 * w.threat_pct / 100;
  cp += std::max(with_capture, without);

  // Drawish material: no pawns for the stronger side and little extra.
  if ((cp > 0 && !our_pawns) || (cp < 0 && !their_pawns)) {
    auto npm = [&](u64 side) {
      return kSeeVal[N] * PopCount(p.pc[N] & side) +
             kSeeVal[B] * PopCount(p.pc[B] & side) +
             kSeeVal[R] * PopCount(p.pc[R] & side) +
             kSeeVal[Q] * PopCount(p.pc[Q] & side);
    };
    const int diff = npm(p.us) - npm(p.them);
    if (cp > 0 ? diff < 400 : -diff < 400) cp /= 4;
  }
  return cp;
}

inline int ToOwnType(PieceType pt) {
  // PieceType: n q r b p k.
  constexpr int kMap[6] = {N, Q, R, B, P, K};
  return kMap[pt.idx];
}

// Rough piece classes for "attacking a more valuable piece".
constexpr int kClass[8] = {0, 1, 1, 2, 3, 4, 0, 0};

void ComputePolicy(const Pos& p, const EvalInfo& info, const Params& w,
                   std::span<const Move> moves, std::span<float> out,
                   float inv_softmax_temp) {
  const auto& pst = kTaperedPst.v[info.phase];
  const float material_scale = inv_softmax_temp / w.policy_temp;
  const float pst_scale = material_scale * w.pst_weight;

  // Squares from which a piece of a given type gives check.
  const int ksq = p.their_king;
  const u64 king_diag = BishopAtt(ksq, p.occ);
  const u64 king_orth = RookAtt(ksq, p.occ);
  const u64 check_sq[6] = {kT.pawn_down[ksq], kT.knight[ksq],        king_diag,
                           king_orth,         king_diag | king_orth, 0};
  // Squares from which a piece of a given type attacks a more valuable or
  // undefended enemy piece.
  u64 threat_sq[6] = {0, 0, 0, 0, 0, 0};
  for (u64 x = p.them & ~p.pc[K]; x; x &= x - 1) {
    const int sq = Lsb(x);
    const int cls = kClass[p.type[sq]];
    const bool undefended = !(info.att_them & Bit(sq));
    // Pawns and knights.
    if (cls > 0 || undefended) threat_sq[P] |= kT.pawn_down[sq];
    if (cls > 1 || undefended) threat_sq[N] |= kT.knight[sq];
    const bool for_bishop = cls > 1 || undefended;
    const bool for_rook = cls > 2 || undefended;
    const bool for_queen = undefended;
    if (for_bishop || for_queen) {
      const u64 d = BishopAtt(sq, p.occ);
      if (for_bishop) threat_sq[B] |= d;
      if (for_queen) threat_sq[Q] |= d;
    }
    if (for_rook || for_queen) {
      const u64 o = RookAtt(sq, p.occ);
      if (for_rook) threat_sq[R] |= o;
      if (for_queen) threat_sq[Q] |= o;
    }
  }

  float max_logit = -1e9f;
  for (size_t i = 0; i < moves.size(); ++i) {
    const Move m = moves[i];
    const int from = m.from().as_idx();
    const int to = m.to().as_idx();
    const int t = p.type[from];
    float logit;
    if (m.is_castling()) {
      logit = w.castle_bonus * material_scale;
    } else {
      const int new_t = m.is_promotion() ? ToOwnType(m.promotion()) : t;
      int victim = kNone;
      if (m.is_en_passant()) {
        victim = P;
      } else if (p.them & Bit(to)) {
        victim = p.type[to];
      }
      // Positional delta (material change of promotion included).
      const int pos_delta = pst[new_t][to] - pst[t][from];
      int material = kSeeVal[victim];
      if (info.att_them & Bit(to)) {
        // With a single attacker worth at least the moving piece and the
        // square defended, they won't gain by capturing: skip SEE.
        // (Pawn pushes don't attack the destination themselves.)
        const bool self_attacks = t != P || victim != kNone;
        const bool defended =
            ((self_attacks ? info.att2_us : info.att_us) & Bit(to)) != 0;
        if (!defended || (info.att2_them & Bit(to)) ||
            LvaValue(info.by_them, to) < kSeeVal[new_t]) {
          const u64 att = AttackersTo(p, to, p.occ) & ~Bit(from);
          material = See(p, from, to, new_t, material, p.us, p.them, att);
        }
      }
      if (material >= 0 && info.threat[from] > 0) {
        material += info.threat[from] * w.escape_pct / 100;
      }
      int bonus = 0;
      if (check_sq[new_t] & Bit(to)) bonus += w.check_bonus;
      if (material >= 0 && (threat_sq[new_t] & Bit(to))) {
        bonus += w.threat_bonus;
      }
      logit = (material + bonus) * material_scale + pos_delta * pst_scale;
    }
    out[i] = logit;
    max_logit = std::max(max_logit, logit);
  }
  float total = 0.0f;
  for (float& v : out) total += (v = FastExp(v - max_logit));
  const float scale = 1.0f / total;
  for (float& v : out) v *= scale;
}

class TrivialBackend;

class TrivialComputation : public BackendComputation {
 public:
  explicit TrivialComputation(const TrivialBackend* backend)
      : backend_(backend) {}
  size_t UsedBatchSize() const override { return batch_size_; }
  AddInputResult AddInput(const EvalPosition& pos,
                          EvalResultPtr result) override;
  void ComputeBlocking() override {}

 private:
  const TrivialBackend* backend_;
  size_t batch_size_ = 0;
};

class TrivialBackend : public Backend {
 public:
  explicit TrivialBackend(const OptionsDict& options)
      : backend_opts_(
            options.Get<std::string>(SharedBackendParams::kBackendOptionsId)) {
    OptionsDict opts;
    opts.AddSubdictFromString(backend_opts_);
    auto get_int = [&](const char* key, int* val) {
      *val = opts.GetOrDefault<int>(key, *val);
    };
    auto get_float = [&](const char* key, float* val) {
      if (opts.Exists<int>(key)) {
        *val = opts.Get<int>(key);
      } else {
        *val = opts.GetOrDefault<float>(key, *val);
      }
    };
    get_float("cp_scale", &params_.cp_scale);
    get_int("tempo", &params_.tempo);
    get_int("capture_pct", &params_.capture_pct);
    get_int("threat_pct", &params_.threat_pct);
    get_int("king_attack", &params_.king_attack);
    get_int("passed_mg", &params_.passed_mg);
    get_int("passed_eg", &params_.passed_eg);
    get_int("isolated", &params_.isolated);
    get_int("doubled", &params_.doubled);
    get_int("bishop_pair", &params_.bishop_pair);
    get_float("policy_temp", &params_.policy_temp);
    get_float("pst_weight", &params_.pst_weight);
    get_int("check_bonus", &params_.check_bonus);
    get_int("castle_bonus", &params_.castle_bonus);
    get_int("escape_pct", &params_.escape_pct);
    get_int("threat_bonus", &params_.threat_bonus);
    int mob = 100;
    get_int("mobility_pct", &mob);
    for (int i = 0; i < 4; ++i) {
      params_.mob_mg[i] = params_.mob_mg[i] * mob / 100;
      params_.mob_eg[i] = params_.mob_eg[i] * mob / 100;
    }
    opts.CheckAllOptionsRead("");
    UpdateConfiguration(options);
  }

  BackendAttributes GetAttributes() const override {
    return BackendAttributes{.has_mlh = false,
                             .has_wdl = false,
                             .runs_on_cpu = false,
                             .suggested_num_search_threads = 1,
                             .recommended_batch_size = 256,
                             .maximum_batch_size = 1024};
  }

  std::unique_ptr<BackendComputation> CreateComputation() override {
    return std::make_unique<TrivialComputation>(this);
  }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    Backend::UpdateConfiguration(options);
    if (backend_opts_ !=
        options.Get<std::string>(SharedBackendParams::kBackendOptionsId)) {
      return NEED_RESTART;
    }
    inv_softmax_temp_ =
        1.0f / options.Get<float>(SharedBackendParams::kPolicySoftmaxTemp);
    return UPDATE_OK;
  }

  void Evaluate(const EvalPosition& pos, EvalResultPtr result) const {
    Pos p;
    Fill(pos.pos.back().GetBoard(), &p);
    EvalInfo info;
    const int cp = EvaluatePosition(p, params_, &info);
    if (result.q) {
      const float q = std::atan(cp / params_.cp_scale) * (1.0f / 1.5637541897f);
      *result.q = std::clamp(q, -1.0f, 1.0f);
    }
    if (result.d) *result.d = 0.0f;
    if (result.m) *result.m = 0.0f;
    if (!result.p.empty()) {
      ComputePolicy(p, info, params_, pos.legal_moves, result.p,
                    inv_softmax_temp_);
    }
  }

 private:
  const std::string backend_opts_;
  Params params_;
  float inv_softmax_temp_ = 1.0f;
};

BackendComputation::AddInputResult TrivialComputation::AddInput(
    const EvalPosition& pos, EvalResultPtr result) {
  // The evaluation is so cheap that it is done right away; the legal moves
  // span is not guaranteed to outlive this call anyway.
  backend_->Evaluate(pos, result);
  ++batch_size_;
  return ENQUEUED_FOR_EVAL;
}

class TrivialBackendFactory : public BackendFactory {
 public:
  int GetPriority() const override { return 4; }
  std::string_view GetName() const override { return "trivial"; }
  std::unique_ptr<Backend> Create(const OptionsDict& options) override {
    return std::make_unique<TrivialBackend>(options);
  }
};

REGISTER_BACKEND(TrivialBackendFactory)

}  // namespace
}  // namespace lczero

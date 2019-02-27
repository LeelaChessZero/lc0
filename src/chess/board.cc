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

#include "chess/board.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include "utils/exception.h"

#if not defined(NO_PEXT)
// Include header for pext instruction.
#include <immintrin.h>
#endif

namespace lczero {

using std::string;

const char* ChessBoard::kStartposFen =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const ChessBoard ChessBoard::kStartposBoard(ChessBoard::kStartposFen);

void ChessBoard::Clear() {
  std::memset(reinterpret_cast<void*>(this), 0, sizeof(ChessBoard));
}

void ChessBoard::Mirror() {
  our_pieces_.Mirror();
  their_pieces_.Mirror();
  std::swap(our_pieces_, their_pieces_);
  rooks_.Mirror();
  bishops_.Mirror();
  pawns_.Mirror();
  our_king_.Mirror();
  their_king_.Mirror();
  std::swap(our_king_, their_king_);
  castlings_.Mirror();
  flipped_ = !flipped_;
}

namespace {
static const BitBoard kPawnMask = 0x00FFFFFFFFFFFF00ULL;

static const std::pair<int, int> kKingMoves[] = {
    {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

static const std::pair<int, int> kRookDirections[] = {
    {1, 0}, {-1, 0}, {0, 1}, {0, -1}};

static const std::pair<int, int> kBishopDirections[] = {
    {1, 1}, {-1, 1}, {1, -1}, {-1, -1}};

// If those squares are attacked, king cannot castle.
static const int k00Attackers[] = {4, 5, 6};
static const int k000Attackers[] = {2, 3, 4};

// Which squares can rook attack from every of squares.
static const BitBoard kRookAttacks[] = {
    0x01010101010101FEULL, 0x02020202020202FDULL, 0x04040404040404FBULL,
    0x08080808080808F7ULL, 0x10101010101010EFULL, 0x20202020202020DFULL,
    0x40404040404040BFULL, 0x808080808080807FULL, 0x010101010101FE01ULL,
    0x020202020202FD02ULL, 0x040404040404FB04ULL, 0x080808080808F708ULL,
    0x101010101010EF10ULL, 0x202020202020DF20ULL, 0x404040404040BF40ULL,
    0x8080808080807F80ULL, 0x0101010101FE0101ULL, 0x0202020202FD0202ULL,
    0x0404040404FB0404ULL, 0x0808080808F70808ULL, 0x1010101010EF1010ULL,
    0x2020202020DF2020ULL, 0x4040404040BF4040ULL, 0x80808080807F8080ULL,
    0x01010101FE010101ULL, 0x02020202FD020202ULL, 0x04040404FB040404ULL,
    0x08080808F7080808ULL, 0x10101010EF101010ULL, 0x20202020DF202020ULL,
    0x40404040BF404040ULL, 0x808080807F808080ULL, 0x010101FE01010101ULL,
    0x020202FD02020202ULL, 0x040404FB04040404ULL, 0x080808F708080808ULL,
    0x101010EF10101010ULL, 0x202020DF20202020ULL, 0x404040BF40404040ULL,
    0x8080807F80808080ULL, 0x0101FE0101010101ULL, 0x0202FD0202020202ULL,
    0x0404FB0404040404ULL, 0x0808F70808080808ULL, 0x1010EF1010101010ULL,
    0x2020DF2020202020ULL, 0x4040BF4040404040ULL, 0x80807F8080808080ULL,
    0x01FE010101010101ULL, 0x02FD020202020202ULL, 0x04FB040404040404ULL,
    0x08F7080808080808ULL, 0x10EF101010101010ULL, 0x20DF202020202020ULL,
    0x40BF404040404040ULL, 0x807F808080808080ULL, 0xFE01010101010101ULL,
    0xFD02020202020202ULL, 0xFB04040404040404ULL, 0xF708080808080808ULL,
    0xEF10101010101010ULL, 0xDF20202020202020ULL, 0xBF40404040404040ULL,
    0x7F80808080808080ULL};
// Which squares can bishop attack.
static const BitBoard kBishopAttacks[] = {
    0x8040201008040200ULL, 0x0080402010080500ULL, 0x0000804020110A00ULL,
    0x0000008041221400ULL, 0x0000000182442800ULL, 0x0000010204885000ULL,
    0x000102040810A000ULL, 0x0102040810204000ULL, 0x4020100804020002ULL,
    0x8040201008050005ULL, 0x00804020110A000AULL, 0x0000804122140014ULL,
    0x0000018244280028ULL, 0x0001020488500050ULL, 0x0102040810A000A0ULL,
    0x0204081020400040ULL, 0x2010080402000204ULL, 0x4020100805000508ULL,
    0x804020110A000A11ULL, 0x0080412214001422ULL, 0x0001824428002844ULL,
    0x0102048850005088ULL, 0x02040810A000A010ULL, 0x0408102040004020ULL,
    0x1008040200020408ULL, 0x2010080500050810ULL, 0x4020110A000A1120ULL,
    0x8041221400142241ULL, 0x0182442800284482ULL, 0x0204885000508804ULL,
    0x040810A000A01008ULL, 0x0810204000402010ULL, 0x0804020002040810ULL,
    0x1008050005081020ULL, 0x20110A000A112040ULL, 0x4122140014224180ULL,
    0x8244280028448201ULL, 0x0488500050880402ULL, 0x0810A000A0100804ULL,
    0x1020400040201008ULL, 0x0402000204081020ULL, 0x0805000508102040ULL,
    0x110A000A11204080ULL, 0x2214001422418000ULL, 0x4428002844820100ULL,
    0x8850005088040201ULL, 0x10A000A010080402ULL, 0x2040004020100804ULL,
    0x0200020408102040ULL, 0x0500050810204080ULL, 0x0A000A1120408000ULL,
    0x1400142241800000ULL, 0x2800284482010000ULL, 0x5000508804020100ULL,
    0xA000A01008040201ULL, 0x4000402010080402ULL, 0x0002040810204080ULL,
    0x0005081020408000ULL, 0x000A112040800000ULL, 0x0014224180000000ULL,
    0x0028448201000000ULL, 0x0050880402010000ULL, 0x00A0100804020100ULL,
    0x0040201008040201ULL};
// Which squares can knight attack.
static const BitBoard kKnightAttacks[] = {
    0x0000000000020400ULL, 0x0000000000050800ULL, 0x00000000000A1100ULL,
    0x0000000000142200ULL, 0x0000000000284400ULL, 0x0000000000508800ULL,
    0x0000000000A01000ULL, 0x0000000000402000ULL, 0x0000000002040004ULL,
    0x0000000005080008ULL, 0x000000000A110011ULL, 0x0000000014220022ULL,
    0x0000000028440044ULL, 0x0000000050880088ULL, 0x00000000A0100010ULL,
    0x0000000040200020ULL, 0x0000000204000402ULL, 0x0000000508000805ULL,
    0x0000000A1100110AULL, 0x0000001422002214ULL, 0x0000002844004428ULL,
    0x0000005088008850ULL, 0x000000A0100010A0ULL, 0x0000004020002040ULL,
    0x0000020400040200ULL, 0x0000050800080500ULL, 0x00000A1100110A00ULL,
    0x0000142200221400ULL, 0x0000284400442800ULL, 0x0000508800885000ULL,
    0x0000A0100010A000ULL, 0x0000402000204000ULL, 0x0002040004020000ULL,
    0x0005080008050000ULL, 0x000A1100110A0000ULL, 0x0014220022140000ULL,
    0x0028440044280000ULL, 0x0050880088500000ULL, 0x00A0100010A00000ULL,
    0x0040200020400000ULL, 0x0204000402000000ULL, 0x0508000805000000ULL,
    0x0A1100110A000000ULL, 0x1422002214000000ULL, 0x2844004428000000ULL,
    0x5088008850000000ULL, 0xA0100010A0000000ULL, 0x4020002040000000ULL,
    0x0400040200000000ULL, 0x0800080500000000ULL, 0x1100110A00000000ULL,
    0x2200221400000000ULL, 0x4400442800000000ULL, 0x8800885000000000ULL,
    0x100010A000000000ULL, 0x2000204000000000ULL, 0x0004020000000000ULL,
    0x0008050000000000ULL, 0x00110A0000000000ULL, 0x0022140000000000ULL,
    0x0044280000000000ULL, 0x0088500000000000ULL, 0x0010A00000000000ULL,
    0x0020400000000000ULL};
// Opponent pawn attacks
static const BitBoard kPawnAttacks[] = {
    0x0000000000000200ULL, 0x0000000000000500ULL, 0x0000000000000A00ULL,
    0x0000000000001400ULL, 0x0000000000002800ULL, 0x0000000000005000ULL,
    0x000000000000A000ULL, 0x0000000000004000ULL, 0x0000000000020000ULL,
    0x0000000000050000ULL, 0x00000000000A0000ULL, 0x0000000000140000ULL,
    0x0000000000280000ULL, 0x0000000000500000ULL, 0x0000000000A00000ULL,
    0x0000000000400000ULL, 0x0000000002000000ULL, 0x0000000005000000ULL,
    0x000000000A000000ULL, 0x0000000014000000ULL, 0x0000000028000000ULL,
    0x0000000050000000ULL, 0x00000000A0000000ULL, 0x0000000040000000ULL,
    0x0000000200000000ULL, 0x0000000500000000ULL, 0x0000000A00000000ULL,
    0x0000001400000000ULL, 0x0000002800000000ULL, 0x0000005000000000ULL,
    0x000000A000000000ULL, 0x0000004000000000ULL, 0x0000020000000000ULL,
    0x0000050000000000ULL, 0x00000A0000000000ULL, 0x0000140000000000ULL,
    0x0000280000000000ULL, 0x0000500000000000ULL, 0x0000A00000000000ULL,
    0x0000400000000000ULL, 0x0002000000000000ULL, 0x0005000000000000ULL,
    0x000A000000000000ULL, 0x0014000000000000ULL, 0x0028000000000000ULL,
    0x0050000000000000ULL, 0x00A0000000000000ULL, 0x0040000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL, 0x0000000000000000ULL, 0x0000000000000000ULL,
    0x0000000000000000ULL};

static const Move::Promotion kPromotions[] = {
    Move::Promotion::Queen,
    Move::Promotion::Rook,
    Move::Promotion::Bishop,
    Move::Promotion::Knight,
};

// Magic bitboard routines and structures.
// We use so-called "fancy" magic bitboards.

// Structure holding all relevant magic parameters per square.
struct MagicParams {
  // Relevant occupancy mask.
  uint64_t mask_;
  // Pointer to lookup table.
  BitBoard* attacks_table_;
#if defined(NO_PEXT)
  // Magic number.
  uint64_t magic_number_;
  // Number of bits to shift.
  uint8_t shift_bits_;
#endif
};

#if defined(NO_PEXT)
// Magic numbers determined via trial and error with random number generator
// such that the number of relevant occupancy bits suffice to index the attacks
// tables with only constructive collisions.
static const BitBoard kRookMagicNumbers[] = {
    0x088000102088C001ULL, 0x10C0200040001000ULL, 0x83001041000B2000ULL,
    0x0680280080041000ULL, 0x488004000A080080ULL, 0x0100180400010002ULL,
    0x040001C401021008ULL, 0x02000C04A980C302ULL, 0x0000800040082084ULL,
    0x5020C00820025000ULL, 0x0001002001044012ULL, 0x0402001020400A00ULL,
    0x00C0800800040080ULL, 0x4028800200040080ULL, 0x00A0804200802500ULL,
    0x8004800040802100ULL, 0x0080004000200040ULL, 0x1082810020400100ULL,
    0x0020004010080040ULL, 0x2004818010042800ULL, 0x0601010008005004ULL,
    0x4600808002001400ULL, 0x0010040009180210ULL, 0x020412000406C091ULL,
    0x040084228000C000ULL, 0x8000810100204000ULL, 0x0084110100402000ULL,
    0x0046001A00204210ULL, 0x2001040080080081ULL, 0x0144020080800400ULL,
    0x0840108400080229ULL, 0x0480308A0000410CULL, 0x0460324002800081ULL,
    0x620080A001804000ULL, 0x2800802000801006ULL, 0x0002809000800800ULL,
    0x4C09040080802800ULL, 0x4808800C00800200ULL, 0x0200311004001802ULL,
    0x0400008402002141ULL, 0x0410800140008020ULL, 0x000080C001050020ULL,
    0x004080204A020010ULL, 0x0224201001010038ULL, 0x0109001108010004ULL,
    0x0282004844020010ULL, 0x8228180110040082ULL, 0x0001000080C10002ULL,
    0x024000C120801080ULL, 0x0001406481060200ULL, 0x0101243200418600ULL,
    0x0108800800100080ULL, 0x4022080100100D00ULL, 0x0000843040600801ULL,
    0x8301000200CC0500ULL, 0x1000004500840200ULL, 0x1100104100800069ULL,
    0x2001008440001021ULL, 0x2002008830204082ULL, 0x0010145000082101ULL,
    0x01A2001004200842ULL, 0x1007000608040041ULL, 0x000A08100203028CULL,
    0x02D4048040290402ULL};
static const BitBoard kBishopMagicNumbers[] = {
    0x0008201802242020ULL, 0x0021040424806220ULL, 0x4006360602013080ULL,
    0x0004410020408002ULL, 0x2102021009001140ULL, 0x08C2021004000001ULL,
    0x6001031120200820ULL, 0x1018310402201410ULL, 0x401CE00210820484ULL,
    0x001029D001004100ULL, 0x2C00101080810032ULL, 0x0000082581000010ULL,
    0x10000A0210110020ULL, 0x200002016C202000ULL, 0x0201018821901000ULL,
    0x006A0300420A2100ULL, 0x0010014005450400ULL, 0x1008C12008028280ULL,
    0x00010010004A0040ULL, 0x3000820802044020ULL, 0x0000800405A02820ULL,
    0x8042004300420240ULL, 0x10060801210D2000ULL, 0x0210840500511061ULL,
    0x0008142118509020ULL, 0x0021109460040104ULL, 0x00A1480090019030ULL,
    0x0102008808008020ULL, 0x884084000880E001ULL, 0x040041020A030100ULL,
    0x3000810104110805ULL, 0x04040A2006808440ULL, 0x0044040404C01100ULL,
    0x4122B80800245004ULL, 0x0044020502380046ULL, 0x0100400888020200ULL,
    0x01C0002060020080ULL, 0x4008811100021001ULL, 0x8208450441040609ULL,
    0x0408004900008088ULL, 0x0294212051220882ULL, 0x000041080810E062ULL,
    0x10480A018E005000ULL, 0x80400A0204201600ULL, 0x2800200204100682ULL,
    0x0020200400204441ULL, 0x0A500600A5002400ULL, 0x801602004A010100ULL,
    0x0801841008040880ULL, 0x10010880C4200028ULL, 0x0400004424040000ULL,
    0x0401000142022100ULL, 0x00A00010020A0002ULL, 0x1010400204010810ULL,
    0x0829910400840000ULL, 0x0004235204010080ULL, 0x1002008143082000ULL,
    0x11840044440C2080ULL, 0x2802A02104030440ULL, 0x6100000900840401ULL,
    0x1C20A15A90420200ULL, 0x0088414004480280ULL, 0x0000204242881100ULL,
    0x0240080802809010ULL};
#endif

// Magic parameters for rooks/bishops.
static MagicParams rook_magic_params[64];
static MagicParams bishop_magic_params[64];

// Precomputed attacks bitboard tables.
static BitBoard rook_attacks_table[102400];
static BitBoard bishop_attacks_table[5248];

// Builds rook or bishop attacks table.
static void BuildAttacksTable(MagicParams* magic_params,
                              BitBoard* attacks_table,
                              const std::pair<int, int>* directions) {
  // Offset into lookup table.
  uint32_t table_offset = 0;

  // Initialize for all board squares.
  for (unsigned square = 0; square < 64; square++) {
    const BoardSquare b_sq(square);

    // Calculate relevant occupancy masks.
    BitBoard mask = {0};

    for (int j = 0; j < 4; j++) {
      auto direction = directions[j];
      auto dst_row = b_sq.row();
      auto dst_col = b_sq.col();
      while (true) {
        dst_row += direction.first;
        dst_col += direction.second;
        // If the next square in this direction is invalid, the current square
        // is at the board's edge and should not be added.
        if (!BoardSquare::IsValid(dst_row + direction.first,
                                  dst_col + direction.second))
          break;
        const BoardSquare destination(dst_row, dst_col);
        mask.set(destination);
      }
    }

    // Set mask.
    magic_params[square].mask_ = mask.as_int();

    // Cache relevant occupancy board squares.
    std::vector<BoardSquare> occupancy_squares;

    for (auto occ_sq : BitBoard(magic_params[square].mask_)) {
      occupancy_squares.emplace_back(occ_sq);
    }

#if defined(NO_PEXT)
    // Set number of shifted bits. The magic numbers have been chosen such that
    // the number of relevant occupancy bits suffice to index the attacks table.
    magic_params[square].shift_bits_ = 64 - occupancy_squares.size();
#endif

    // Set pointer to lookup table.
    magic_params[square].attacks_table_ = &attacks_table[table_offset];

    // Clear attacks table (used for sanity check later on).
    for (int i = 0; i < (1 << occupancy_squares.size()); i++) {
      attacks_table[table_offset + i] = 0;
    }

    // Build square attacks table for every possible relevant occupancy
    // bitboard.
    for (int i = 0; i < (1 << occupancy_squares.size()); i++) {
      BitBoard occupancy(0);

      for (size_t bit = 0; bit < occupancy_squares.size(); bit++) {
        occupancy.set_if(occupancy_squares[bit], (1 << bit) & i);
      }

      // Calculate attacks bitboard corresponding to this occupancy bitboard.
      BitBoard attacks(0);

      for (int j = 0; j < 4; j++) {
        auto direction = directions[j];
        auto dst_row = b_sq.row();
        auto dst_col = b_sq.col();
        while (true) {
          dst_row += direction.first;
          dst_col += direction.second;
          if (!BoardSquare::IsValid(dst_row, dst_col)) break;
          const BoardSquare destination(dst_row, dst_col);
          attacks.set(destination);
          if (occupancy.get(destination)) break;
        }
      }

#if defined(NO_PEXT)
      // Calculate magic index.
      uint64_t index = occupancy.as_int();
      index *= magic_params[square].magic_number_;
      index >>= magic_params[square].shift_bits_;

      // Sanity check. The magic numbers have been chosen such that
      // the number of relevant occupancy bits suffice to index the attacks
      // table. If the table already contains an attacks bitboard, possible
      // collisions should be constructive.
      if (attacks_table[table_offset + index] != 0 &&
          attacks_table[table_offset + index] != attacks) {
        throw Exception("Invalid magic number!");
      }
#else
      uint64_t index =
          _pext_u64(occupancy.as_int(), magic_params[square].mask_);
#endif

      // Update table.
      attacks_table[table_offset + index] = attacks;
    }

    // Update table offset.
    table_offset += (1 << occupancy_squares.size());
  }
}

// Returns the rook attacks bitboard for the given rook board square and the
// given occupied piece bitboard.
static inline BitBoard GetRookAttacks(const BoardSquare rook_square,
                                      const BitBoard pieces) {
  // Calculate magic index.
  const uint8_t square = rook_square.as_int();

#if defined(NO_PEXT)
  uint64_t index = pieces.as_int() & rook_magic_params[square].mask_;
  index *= rook_magic_params[square].magic_number_;
  index >>= rook_magic_params[square].shift_bits_;
#else
  uint64_t index = _pext_u64(pieces.as_int(), rook_magic_params[square].mask_);
#endif

  // Return attacks bitboard.
  return rook_magic_params[square].attacks_table_[index];
}

// Returns the bishop attacks bitboard for the given bishop board square and
// the given occupied piece bitboard.
static inline BitBoard GetBishopAttacks(const BoardSquare bishop_square,
                                        const BitBoard pieces) {
  // Calculate magic index.
  const uint8_t square = bishop_square.as_int();

#if defined(NO_PEXT)
  uint64_t index = pieces.as_int() & bishop_magic_params[square].mask_;
  index *= bishop_magic_params[square].magic_number_;
  index >>= bishop_magic_params[square].shift_bits_;
#else
  uint64_t index =
      _pext_u64(pieces.as_int(), bishop_magic_params[square].mask_);
#endif

  // Return attacks bitboard.
  return bishop_magic_params[square].attacks_table_[index];
}

}  // namespace

void InitializeMagicBitboards() {
#if defined(NO_PEXT)
  // Set magic numbers for all board squares.
  for (unsigned square = 0; square < 64; square++) {
    rook_magic_params[square].magic_number_ =
        kRookMagicNumbers[square].as_int();
    bishop_magic_params[square].magic_number_ =
        kBishopMagicNumbers[square].as_int();
  }
#endif

  // Build attacks tables.
  BuildAttacksTable(rook_magic_params, rook_attacks_table, kRookDirections);
  BuildAttacksTable(bishop_magic_params, bishop_attacks_table,
                    kBishopDirections);
}

BitBoard ChessBoard::pawns() const { return pawns_ & kPawnMask; }

BitBoard ChessBoard::en_passant() const { return pawns_ - pawns(); }

MoveList ChessBoard::GeneratePseudolegalMoves() const {
  MoveList result;
  result.reserve(60);
  for (auto source : our_pieces_) {
    // King
    if (source == our_king_) {
      for (const auto& delta : kKingMoves) {
        const auto dst_row = source.row() + delta.first;
        const auto dst_col = source.col() + delta.second;
        if (!BoardSquare::IsValid(dst_row, dst_col)) continue;
        const BoardSquare destination(dst_row, dst_col);
        if (our_pieces_.get(destination)) continue;
        if (IsUnderAttack(destination)) continue;
        result.emplace_back(source, destination);
      }
      // Castlings.
      if (castlings_.we_can_00()) {
        bool can_castle = true;
        for (int i = 5; i < 7; ++i) {
          if (our_pieces_.get(i) || their_pieces_.get(i)) {
            can_castle = false;
            break;
          }
        }
        if (can_castle) {
          for (auto x : k00Attackers) {
            if (IsUnderAttack(x)) {
              can_castle = false;
              break;
            }
          }
        }
        if (can_castle) {
          result.emplace_back(source, BoardSquare(0, 6));
          result.back().SetCastling();
        }
      }
      if (castlings_.we_can_000()) {
        bool can_castle = true;
        for (int i = 1; i < 4; ++i) {
          if (our_pieces_.get(i) || their_pieces_.get(i)) {
            can_castle = false;
            break;
          }
        }
        if (can_castle) {
          for (auto x : k000Attackers) {
            if (IsUnderAttack(x)) {
              can_castle = false;
              break;
            }
          }
        }
        if (can_castle) {
          result.emplace_back(source, BoardSquare(0, 2));
          result.back().SetCastling();
        }
      }
      continue;
    }
    bool processed_piece = false;
    // Rook (and queen)
    if (rooks_.get(source)) {
      processed_piece = true;
      BitBoard attacked =
          GetRookAttacks(source, our_pieces_ | their_pieces_) - our_pieces_;

      for (const auto& destination : attacked) {
        result.emplace_back(source, destination);
      }
    }
    // Bishop (and queen)
    if (bishops_.get(source)) {
      processed_piece = true;
      BitBoard attacked =
          GetBishopAttacks(source, our_pieces_ | their_pieces_) - our_pieces_;

      for (const auto& destination : attacked) {
        result.emplace_back(source, destination);
      }
    }
    if (processed_piece) continue;
    // Pawns.
    if ((pawns_ & kPawnMask).get(source)) {
      // Moves forward.
      {
        const auto dst_row = source.row() + 1;
        const auto dst_col = source.col();
        const BoardSquare destination(dst_row, dst_col);

        if (!our_pieces_.get(destination) && !their_pieces_.get(destination)) {
          if (dst_row != 7) {
            result.emplace_back(source, destination);
            if (dst_row == 2) {
              // Maybe it'll be possible to move two squares.
              if (!our_pieces_.get(3, dst_col) &&
                  !their_pieces_.get(3, dst_col)) {
                result.emplace_back(source, BoardSquare(3, dst_col));
              }
            }
          } else {
            // Promotions
            for (auto promotion : kPromotions) {
              result.emplace_back(source, destination, promotion);
            }
          }
        }
      }
      // Captures.
      {
        for (auto direction : {-1, 1}) {
          const auto dst_row = source.row() + 1;
          const auto dst_col = source.col() + direction;
          if (dst_col < 0 || dst_col >= 8) continue;
          const BoardSquare destination(dst_row, dst_col);
          if (their_pieces_.get(destination)) {
            if (dst_row == 7) {
              // Promotion.
              for (auto promotion : kPromotions) {
                result.emplace_back(source, destination, promotion);
              }
            } else {
              // Ordinary capture.
              result.emplace_back(source, destination);
            }
          } else if (dst_row == 5 && pawns_.get(7, dst_col)) {
            // En passant.
            // "Pawn" on opponent's file 8 means that en passant is possible.
            // Those fake pawns are reset in ApplyMove.
            result.emplace_back(source, destination);
          }
        }
      }
      continue;
    }
    // Knight.
    {
      for (const auto destination :
           kKnightAttacks[source.as_int()] - our_pieces_) {
        result.emplace_back(source, destination);
      }
    }
  }
  return result;
}

bool ChessBoard::ApplyMove(Move move) {
  const auto& from = move.from();
  const auto& to = move.to();
  const auto from_row = from.row();
  const auto from_col = from.col();
  const auto to_row = to.row();
  const auto to_col = to.col();

  // Move in our pieces.
  our_pieces_.reset(from);
  our_pieces_.set(to);

  // Remove captured piece
  bool reset_50_moves = their_pieces_.get(to);
  their_pieces_.reset(to);
  rooks_.reset(to);
  bishops_.reset(to);
  pawns_.reset(to);
  if (to.as_int() == 63) {
    castlings_.reset_they_can_00();
  }
  if (to.as_int() == 56) {
    castlings_.reset_they_can_000();
  }

  // En passant
  if (from_row == 4 && pawns_.get(from) && from_col != to_col &&
      pawns_.get(7, to_col)) {
    pawns_.reset(4, to_col);
    their_pieces_.reset(4, to_col);
  }

  // Remove en passant flags.
  pawns_ &= kPawnMask;

  // If pawn was moved, reset 50 move draw counter.
  reset_50_moves |= pawns_.get(from);

  // King
  if (from == our_king_) {
    castlings_.reset_we_can_00();
    castlings_.reset_we_can_000();
    our_king_ = to;
    // Castling
    if (to_col - from_col > 1) {
      // 0-0
      our_pieces_.reset(7);
      rooks_.reset(7);
      our_pieces_.set(5);
      rooks_.set(5);
    } else if (from_col - to_col > 1) {
      // 0-0-0
      our_pieces_.reset(0);
      rooks_.reset(0);
      our_pieces_.set(3);
      rooks_.set(3);
    }
    return reset_50_moves;
  }

  // Promotion
  if (move.promotion() != Move::Promotion::None) {
    switch (move.promotion()) {
      case Move::Promotion::Rook:
        rooks_.set(to);
        break;
      case Move::Promotion::Bishop:
        bishops_.set(to);
        break;
      case Move::Promotion::Queen:
        rooks_.set(to);
        bishops_.set(to);
        break;
      default:;
    }
    pawns_.reset(from);
    return true;
  }

  // Reset castling rights.
  if (from.as_int() == 0) {
    castlings_.reset_we_can_000();
  }
  if (from.as_int() == 7) {
    castlings_.reset_we_can_00();
  }

  // Ordinary move.
  rooks_.set_if(to, rooks_.get(from));
  bishops_.set_if(to, bishops_.get(from));
  pawns_.set_if(to, pawns_.get(from));
  rooks_.reset(from);
  bishops_.reset(from);
  pawns_.reset(from);

  // Set en passant flag.
  if (to_row - from_row == 2 && pawns_.get(to)) {
    BoardSquare ep_sq(to_row - 1, to_col);
    if (kPawnAttacks[ep_sq.as_int()].intersects(their_pieces_ & pawns_)) {
      pawns_.set(0, to_col);
    }
  }
  return reset_50_moves;
}

bool ChessBoard::IsUnderAttack(BoardSquare square) const {
  const int row = square.row();
  const int col = square.col();
  // Check king.
  {
    const int krow = their_king_.row();
    const int kcol = their_king_.col();
    if (std::abs(krow - row) <= 1 && std::abs(kcol - col) <= 1) return true;
  }
  // Check rooks (and queens).
  if (GetRookAttacks(square, our_pieces_ | their_pieces_)
          .intersects(their_pieces_ & rooks_)) {
    return true;
  }
  // Check bishops.
  if (GetBishopAttacks(square, our_pieces_ | their_pieces_)
          .intersects(their_pieces_ & bishops_)) {
    return true;
  }
  // Check pawns.
  if (kPawnAttacks[square.as_int()].intersects(their_pieces_ & pawns_)) {
    return true;
  }
  // Check knights.
  {
    if (kKnightAttacks[square.as_int()].intersects(their_pieces_ - their_king_ -
                                                   rooks_ - bishops_ -
                                                   (pawns_ & kPawnMask))) {
      return true;
    }
  }
  return false;
}

KingAttackInfo ChessBoard::GenerateKingAttackInfo() const {
  KingAttackInfo king_attack_info;

  // Number of attackers that give check (used for double check detection).
  unsigned num_king_attackers = 0;

  const int row = our_king_.row();
  const int col = our_king_.col();
  // King checks are unnecessary, as kings cannot give check.
  // Check rooks (and queens).
  if (kRookAttacks[our_king_.as_int()].intersects(their_pieces_ & rooks_)) {
    for (const auto& direction : kRookDirections) {
      auto dst_row = row;
      auto dst_col = col;
      BitBoard attack_line(0);
      bool possible_pinned_piece_found = false;
      BoardSquare possible_pinned_piece;
      while (true) {
        dst_row += direction.first;
        dst_col += direction.second;
        if (!BoardSquare::IsValid(dst_row, dst_col)) break;
        const BoardSquare destination(dst_row, dst_col);
        if (our_pieces_.get(destination)) {
          if (possible_pinned_piece_found) {
            // No pieces pinned.
            break;
          } else {
            // This is a possible pinned piece.
            possible_pinned_piece_found = true;
            possible_pinned_piece = destination;
          }
        }
        if (!possible_pinned_piece_found) {
          attack_line.set(destination);
        }
        if (their_pieces_.get(destination)) {
          if (rooks_.get(destination)) {
            if (possible_pinned_piece_found) {
              // Store the pinned piece.
              king_attack_info.pinned_pieces_.set(possible_pinned_piece);
            } else {
              // Update attack lines.
              king_attack_info.attack_lines_ =
                  king_attack_info.attack_lines_ | attack_line;
              num_king_attackers++;
            }
          }
          break;
        }
      }
    }
  }
  // Check bishops.
  if (kBishopAttacks[our_king_.as_int()].intersects(their_pieces_ & bishops_)) {
    for (const auto& direction : kBishopDirections) {
      auto dst_row = row;
      auto dst_col = col;
      BitBoard attack_line(0);
      bool possible_pinned_piece_found = false;
      BoardSquare possible_pinned_piece;
      while (true) {
        dst_row += direction.first;
        dst_col += direction.second;
        if (!BoardSquare::IsValid(dst_row, dst_col)) break;
        const BoardSquare destination(dst_row, dst_col);
        if (our_pieces_.get(destination)) {
          if (possible_pinned_piece_found) {
            // No pieces pinned.
            break;
          } else {
            // This is a possible pinned piece.
            possible_pinned_piece_found = true;
            possible_pinned_piece = destination;
          }
        }
        if (!possible_pinned_piece_found) {
          attack_line.set(destination);
        }
        if (their_pieces_.get(destination)) {
          if (bishops_.get(destination)) {
            if (possible_pinned_piece_found) {
              // Store the pinned piece.
              king_attack_info.pinned_pieces_.set(possible_pinned_piece);
            } else {
              // Update attack lines.
              king_attack_info.attack_lines_ =
                  king_attack_info.attack_lines_ | attack_line;
              num_king_attackers++;
            }
          }
          break;
        }
      }
    }
  }
  // Check pawns.
  const BitBoard attacking_pawns =
      kPawnAttacks[our_king_.as_int()] & their_pieces_ & pawns_;
  king_attack_info.attack_lines_ =
      king_attack_info.attack_lines_ | attacking_pawns;

  if (attacking_pawns.as_int()) {
    // No more than one pawn can give check.
    num_king_attackers++;
  }

  // Check knights.
  const BitBoard attacking_knights =
      kKnightAttacks[our_king_.as_int()] &
      (their_pieces_ - their_king_ - rooks_ - bishops_ - (pawns_ & kPawnMask));
  king_attack_info.attack_lines_ =
      king_attack_info.attack_lines_ | attacking_knights;

  if (attacking_knights.as_int()) {
    // No more than one knight can give check.
    num_king_attackers++;
  }

  assert(num_king_attackers <= 2);
  king_attack_info.double_check_ = (num_king_attackers == 2);

  return king_attack_info;
}

bool ChessBoard::IsLegalMove(Move move,
                             const KingAttackInfo& king_attack_info) const {
  const auto& from = move.from();
  const auto& to = move.to();

  // En passant. Complex but rare. Just apply
  // and check that we are not under check.
  if (from.row() == 4 && pawns_.get(from) && from.col() != to.col() &&
      pawns_.get(7, to.col())) {
    ChessBoard board(*this);
    board.ApplyMove(move);
    return !board.IsUnderCheck();
  }

  // Check if we are already under check.
  if (king_attack_info.in_check()) {
    // King move.
    if (from == our_king_) {
      // Just apply and check that we are not under check.
      ChessBoard board(*this);
      board.ApplyMove(move);
      return !board.IsUnderCheck();
    }

    // Pinned pieces can never resolve a check.
    if (king_attack_info.is_pinned(from)) {
      return false;
    }

    // The piece to move is no king and is not pinned.
    if (king_attack_info.in_double_check()) {
      // Only a king move can resolve the double check.
      return false;
    } else {
      // Only one attacking piece gives check.
      // Our piece is free to move (not pinned). Check if the attacker is
      // captured or interposed after the piece has moved to its destination
      // square.
      return king_attack_info.is_on_attack_line(to);
    }
  }

  // Castlings were checked earlier.
  // Moreover, no pseudolegal king moves to an attacked square are generated.
  // If it's king's move at this moment, its certainly legal.
  if (from == our_king_) {
    return true;
  }

  // If we get here, we are not under check.
  // If the piece is not pinned, it is free to move anywhere.
  if (!king_attack_info.is_pinned(from)) return true;

  // The piece is pinned. Now check that it stays on the same line w.r.t. the
  // king.
  const int dx_from = from.col() - our_king_.col();
  const int dy_from = from.row() - our_king_.row();
  const int dx_to = to.col() - our_king_.col();
  const int dy_to = to.row() - our_king_.row();

  if (dx_from == 0 || dx_to == 0) {
    return (dx_from == dx_to);
  } else {
    return (dx_from * dy_to == dx_to * dy_from);
  }
}

MoveList ChessBoard::GenerateLegalMoves() const {
  const KingAttackInfo king_attack_info = GenerateKingAttackInfo();
  MoveList move_list = GeneratePseudolegalMoves();
  MoveList result;
  result.reserve(move_list.size());

  for (Move m : move_list) {
    if (IsLegalMove(m, king_attack_info)) result.emplace_back(m);
  }

  return result;
}

void ChessBoard::SetFromFen(const std::string& fen, int* no_capture_ply,
                            int* moves) {
  Clear();
  int row = 7;
  int col = 0;

  std::istringstream fen_str(fen);
  string board;
  string who_to_move;
  string castlings;
  string en_passant;
  int no_capture_halfmoves;
  int total_moves;
  fen_str >> board >> who_to_move >> castlings >> en_passant >>
      no_capture_halfmoves >> total_moves;

  if (!fen_str) throw Exception("Bad fen string: " + fen);

  for (char c : board) {
    if (c == '/') {
      --row;
      col = 0;
      continue;
    }
    if (std::isdigit(c)) {
      col += c - '0';
      continue;
    }

    if (std::isupper(c)) {
      // White piece.
      our_pieces_.set(row, col);
    } else {
      // Black piece.
      their_pieces_.set(row, col);
    }

    if (c == 'K') {
      our_king_.set(row, col);
    } else if (c == 'k') {
      their_king_.set(row, col);
    } else if (c == 'R' || c == 'r') {
      rooks_.set(row, col);
    } else if (c == 'B' || c == 'b') {
      bishops_.set(row, col);
    } else if (c == 'Q' || c == 'q') {
      rooks_.set(row, col);
      bishops_.set(row, col);
    } else if (c == 'P' || c == 'p') {
      pawns_.set(row, col);
    } else if (c == 'N' || c == 'n') {
      // Do nothing
    } else {
      throw Exception("Bad fen string: " + fen);
    }
    ++col;
  }

  if (castlings != "-") {
    for (char c : castlings) {
      switch (c) {
        case 'K':
          castlings_.set_we_can_00();
          break;
        case 'k':
          castlings_.set_they_can_00();
          break;
        case 'Q':
          castlings_.set_we_can_000();
          break;
        case 'q':
          castlings_.set_they_can_000();
          break;
        default:
          throw Exception("Bad fen string: " + fen);
      }
    }
  }

  if (en_passant != "-") {
    auto square = BoardSquare(en_passant);
    if (square.row() != 2 && square.row() != 5)
      throw Exception("Bad fen string: " + fen + " wrong en passant rank");
    pawns_.set((square.row() == 2) ? 0 : 7, square.col());
  }

  if (who_to_move == "b" || who_to_move == "B") {
    Mirror();
  }
  if (no_capture_ply) *no_capture_ply = no_capture_halfmoves;
  if (moves) *moves = total_moves;
}

bool ChessBoard::HasMatingMaterial() const {
  if (!rooks_.empty() || !pawns_.empty()) {
    return true;
  }

  if ((our_pieces_ | their_pieces_).count() < 4) {
    // K v K, K+B v K, K+N v K.
    return false;
  }
  if (!our_knights().empty() || !their_knights().empty()) {
    return true;
  }

  // Only kings and bishops remain.

  constexpr BitBoard kLightSquares(0x55AA55AA55AA55AAULL);
  constexpr BitBoard kDarkSquares(0xAA55AA55AA55AA55ULL);

  const bool light_bishop = bishops_.intersects(kLightSquares);
  const bool dark_bishop = bishops_.intersects(kDarkSquares);
  return light_bishop && dark_bishop;
}

string ChessBoard::DebugString() const {
  string result;
  for (int i = 7; i >= 0; --i) {
    for (int j = 0; j < 8; ++j) {
      if (!our_pieces_.get(i, j) && !their_pieces_.get(i, j)) {
        if (i == 2 && pawns_.get(0, j))
          result += '*';
        else if (i == 5 && pawns_.get(7, j))
          result += '*';
        else
          result += '.';
        continue;
      }
      if (our_king_ == i * 8 + j) {
        result += 'K';
        continue;
      }
      if (their_king_ == i * 8 + j) {
        result += 'k';
        continue;
      }
      char c = '?';
      if ((pawns_ & kPawnMask).get(i, j)) {
        c = 'p';
      } else if (bishops_.get(i, j)) {
        if (rooks_.get(i, j))
          c = 'q';
        else
          c = 'b';
      } else if (rooks_.get(i, j)) {
        c = 'r';
      } else {
        c = 'n';
      }
      if (our_pieces_.get(i, j)) c = std::toupper(c);
      result += c;
    }
    if (i == 0) {
      result += " " + castlings_.as_string();
      result += flipped_ ? " (from black's eyes)" : " (from white's eyes)";
      result += " Hash: " + std::to_string(Hash());
    }
    result += '\n';
  }
  return result;
}

}  // namespace lczero

// Originally taken from stockfish syzygy/tbprobe.h.
/*
Stockfish, a UCI chess playing engine derived from Glaurung 2.1
Copyright (c) 2013 Ronald de Man
Copyright (C) 2016-2018 Marco Costalba, Lucas Braesch

Stockfish is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Stockfish is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <algorithm>
#include <atomic>
#include <deque>
#include <tuple>
#include <vector>
#include "chess/position.h"

namespace lczero {
enum WDLScore {
  WDL_LOSS = -2,  // Loss
  WDL_BLESSED_LOSS = -1,  // Loss, but draw under 50-move rule
  WDL_DRAW = 0,  // Draw
  WDL_CURSED_WIN = 1,  // Win, but draw under 50-move rule
  WDL_WIN = 2,  // Win

  WDL_SCORE_NONE = -1000
};

// Possible states after a probing operation
enum ProbeState {
  FAIL = 0,  // Probe failed (missing file table)
  OK = 1,  // Probe succesful
  CHANGE_STM = -1,  // DTZ should check the other side
  ZEROING_BEST_MOVE = 2  // Best move zeroes DTZ (capture or pawn move)
};

enum PieceType {
  NO_PIECE_TYPE, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING,
  ALL_PIECES = 0,
  QUEEN_DIAGONAL = 7,
  PIECE_TYPE_NB = 8
};

enum Piece {
  NO_PIECE,
  W_PAWN = 1, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, W_KING,
  B_PAWN = 9, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING,
  PIECE_NB = 16
};

enum TBType { KEY, WDL, DTZ }; // Used as template parameter

constexpr int TBPIECES = 6;  // Max number of supported pieces

constexpr int NUM_SQUARES = 64;  // Num squares on a chess board.

typedef uint64_t Key;

// Numbers in little endian used by sparseIndex[] to point into blockLength[]
struct SparseEntry {
  char block[4];   // Number of block
  char offset[2];  // Offset within the block
};

static_assert(sizeof(SparseEntry) == 6, "SparseEntry must be 6 bytes");

typedef uint16_t Sym; // Huffman symbol

struct LR {
  enum Side { Left, Right, Value };

  uint8_t lr[3]; // The first 12 bits is the left-hand symbol, the second 12
                 // bits is the right-hand symbol. If symbol has length 1,
                 // then the first byte is the stored value.
  template<Side S>
  Sym get() {
    return S == Left ? ((lr[1] & 0xF) << 8) | lr[0] :
      S == Right ? (lr[2] << 4) | (lr[1] >> 4) :
      S == Value ? lr[0] : (assert(false), Sym(-1));
  }
};

static_assert(sizeof(LR) == 3, "LR tree entry must be 3 bytes");

// struct PairsData contains low level indexing information to access TB data.
// There are 8, 4 or 2 PairsData records for each TBTable, according to type of
// table and if positions have pawns or not. It is populated at first access.
struct PairsData {
  uint8_t flags;                 // Table flags, see enum TBFlag
  uint8_t maxSymLen;             // Maximum length in bits of the Huffman symbols
  uint8_t minSymLen;             // Minimum length in bits of the Huffman symbols
  uint32_t blocksNum;            // Number of blocks in the TB file
  size_t sizeofBlock;            // Block size in bytes
  size_t span;                   // About every span values there is a SparseIndex[] entry
  Sym* lowestSym;                // lowestSym[l] is the symbol of length l with the lowest value
  LR* btree;                     // btree[sym] stores the left and right symbols that expand sym
  uint16_t* blockLength;         // Number of stored positions (minus one) for each block: 1..65536
  uint32_t blockLengthSize;      // Size of blockLength[] table: padded so it's bigger than blocksNum
  SparseEntry* sparseIndex;      // Partial indices into blockLength[]
  size_t sparseIndexSize;        // Size of SparseIndex[] table
  uint8_t* data;                 // Start of Huffman compressed data
  std::vector<uint64_t> base64;  // base64[l - min_sym_len] is the 64bit-padded lowest symbol of length l
  std::vector<uint8_t> symlen;   // Number of values (-1) represented by a given Huffman symbol: 1..256
  Piece pieces[TBPIECES];        // Position pieces: the order of pieces defines the groups
  uint64_t groupIdx[TBPIECES + 1]; // Start index used for the encoding of the group's pieces
  int groupLen[TBPIECES + 1];      // Number of pieces in a given group: KRKN -> (3, 1)
  uint16_t map_idx[4];           // WDLWin, WDLLoss, WDLCursedWin, WDLBlessedLoss (used in DTZ)
};

// TBTable contains indexing information to access the corresponding TBFile.
// There are 2 types of TBTable, corresponding to a WDL or a DTZ file. TBTable
// is populated at init time but the nested PairsData records are populated at
// first access, when the corresponding file is memory mapped.
template<TBType Type>
struct TBTable {
  typedef typename std::conditional<Type == WDL, WDLScore, int>::type Ret;

  static constexpr int Sides = Type == WDL ? 2 : 1;

  std::atomic_bool ready;
  void* baseAddress;
  uint8_t* map;
  uint64_t mapping;
  Key key;
  Key key2;
  int pieceCount;
  bool hasPawns;
  bool hasUniquePieces;
  uint8_t pawnCount[2]; // [Lead color / other color]
  PairsData items[Sides][4]; // [wtm / btm][FILE_A..FILE_D or 0]

  PairsData* get(int stm, int f) {
    return &items[stm % Sides][hasPawns ? f : 0];
  }

  TBTable() : ready(false), baseAddress(nullptr) {}
  explicit TBTable(const std::string& code);
  explicit TBTable(const TBTable<WDL>& wdl);

  ~TBTable();
};

// TBTables creates and keeps ownership of the TBTable objects, one for
// each TB file found. It supports a fast, hash based, table lookup. Populated
// at init time, accessed at probe time.  Its thread safety comes from the
// guarantee that init is strictly ordered with respect to any probes.
class TBTables {

  typedef std::tuple<Key, TBTable<WDL>*, TBTable<DTZ>*> Entry;

  static const int Size = 1 << 13; // 8K table, indexed by key's 13 lsb

  Entry hashTable[Size];

  std::deque<TBTable<WDL>> wdlTable;
  std::deque<TBTable<DTZ>> dtzTable;
  size_t m_dtzsize{ 0 };
  int max_cardinality_{ 0 };

  void insert(Key key, TBTable<WDL>* wdl, TBTable<DTZ>* dtz) {
    Entry* entry = &hashTable[(uint32_t)key & (Size - 1)];

    // Ensure last element is empty to avoid overflow when looking up
    for (; entry - hashTable < Size - 1; ++entry)
      if (std::get<KEY>(*entry) == key || !std::get<WDL>(*entry)) {
        *entry = std::make_tuple(key, wdl, dtz);
        return;
      }
    std::cerr << "TB hash table size too low!" << std::endl;
    exit(1);
  }

 public:
  template<TBType Type>
  TBTable<Type>* get(Key key) {
    for (const Entry* entry = &hashTable[(uint32_t)key & (Size - 1)]; ; ++entry) {
      if (std::get<KEY>(*entry) == key || !std::get<Type>(*entry))
        return std::get<Type>(*entry);
    }
  }

  void clear() {
    memset(hashTable, 0, sizeof(hashTable));
    wdlTable.clear();
    dtzTable.clear();
    m_dtzsize = size_t{ 0 };
    max_cardinality_ = 0;
  }
  size_t size() const { return wdlTable.size(); }
  size_t dtz_size() const { return m_dtzsize; }
  int max_cardinality() const { return max_cardinality_; }
  void add(const std::vector<PieceType>& pieces, const std::string& paths);
};

// Provides methods to load and probe syzygy tablebases.
// Thread safe methods are thread safe subject to the non-thread sfaety
// conditions of the init method.
class SyzygyTablebase {
 public:

  // Current maximum number of pieces on board that can be probed for. Will
  // be 0 unless initialized with tablebase paths.
  // Thread safe.
  int max_cardinality() { return tables_.max_cardinality(); }

  // Allows for the tablebases being used to be changed. This method is not
  // thread safe, there must be no concurrent usage while this method is
  // running. All other thread safe method calls must be strictly ordered with
  // respect to this method.
  void init(const std::string& paths);

  // Probes WDL tables for the given position to determine a WDLScore.
  // Thread safe.
  // Result is only strictly valid for positions with 0 ply 50 move counter.
  // Probe state will return FAIL if the position is not in the tablebase.
  WDLScore probe_wdl(Position& pos, ProbeState* result);

  // Probes DTZ tables for the given position to determine the number of ply
  // before a zeroing move under optimal play.
  // Thread safe.
  // Probe state will return FAIL if the position is not in the tablebase.
  int probe_dtz(Position& pos, ProbeState* result);

  // Probes DTZ tables to determine which moves are on the optimal play path.
  // Assumes the position is one reached such that the side to move has been
  // performing optimal play moves since the last 50 move counter reset.
  // Thread safe.
  // Returns false if the position is not in the tablebase.
  // Safe moves are added to the safe_moves output paramater.
  bool root_probe(Position& pos, std::vector<Move>* safe_moves);

  // Probes WDL tables to determine which moves might be on the optimal play
  // path. If 50 move ply counter is non-zero some (or maybe even all) of the
  // returned safe moves in a 'winning' position, may actually be draws.
  // Returns false if the position is not in the tablebase.
  // Safe moves are added to the safe_moves output paramater.
  bool root_probe_wdl(Position& pos, std::vector<Move>* safe_moves);

 private:
  template<TBType Type, typename Ret = typename TBTable<Type>::Ret>
  Ret probe_table(const Position& pos, ProbeState* result, WDLScore wdl = WDL_DRAW);

  template<bool CheckZeroingMoves = false>
  WDLScore search(Position& pos, ProbeState* result);

  template<TBType Type>
  void* mapped(TBTable<Type>& e, const Position& pos);


  TBTables tables_;
  std::string paths_;

};

}

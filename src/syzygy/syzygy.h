#pragma once

#include <algorithm>
#include <atomic>
#include <deque>
#include <memory>
#include <tuple>
#include <vector>
#include "chess/position.h"

namespace lczero {

enum WDLScore {
  WDL_LOSS = -2,          // Loss
  WDL_BLESSED_LOSS = -1,  // Loss, but draw under 50-move rule
  WDL_DRAW = 0,           // Draw
  WDL_CURSED_WIN = 1,     // Win, but draw under 50-move rule
  WDL_WIN = 2,            // Win
  WDL_SCORE_NONE = -1000
};

// Possible states after a probing operation
enum ProbeState {
  FAIL = 0,              // Probe failed (missing file table)
  OK = 1,                // Probe succesful
  CHANGE_STM = -1,       // DTZ should check the other side
  ZEROING_BEST_MOVE = 2  // Best move zeroes DTZ (capture or pawn move)
};

class SyzygyTablebaseImpl;

// Provides methods to load and probe syzygy tablebases.
// Thread safe methods are thread safe subject to the non-thread sfaety
// conditions of the init method.
class SyzygyTablebase {
 public:
  SyzygyTablebase();
  virtual ~SyzygyTablebase();
  // Current maximum number of pieces on board that can be probed for. Will
  // be 0 unless initialized with tablebase paths.
  // Thread safe.
  int max_cardinality() { return max_cardinality_; }
  // Allows for the tablebases being used to be changed. This method is not
  // thread safe, there must be no concurrent usage while this method is
  // running. All other thread safe method calls must be strictly ordered with
  // respect to this method.
  void init(const std::string& paths);
  // Probes WDL tables for the given position to determine a WDLScore.
  // Thread safe.
  // Result is only strictly valid for positions with 0 ply 50 move counter.
  // Probe state will return FAIL if the position is not in the tablebase.
  WDLScore probe_wdl(const Position& pos, ProbeState* result);
  // Probes DTZ tables for the given position to determine the number of ply
  // before a zeroing move under optimal play.
  // Thread safe.
  // Probe state will return FAIL if the position is not in the tablebase.
  int probe_dtz(const Position& pos, ProbeState* result);
  // Probes DTZ tables to determine which moves are on the optimal play path.
  // Assumes the position is one reached such that the side to move has been
  // performing optimal play moves since the last 50 move counter reset.
  // Thread safe.
  // Returns false if the position is not in the tablebase.
  // Safe moves are added to the safe_moves output paramater.
  bool root_probe(const Position& pos, std::vector<Move>* safe_moves);
  // Probes WDL tables to determine which moves might be on the optimal play
  // path. If 50 move ply counter is non-zero some (or maybe even all) of the
  // returned safe moves in a 'winning' position, may actually be draws.
  // Returns false if the position is not in the tablebase.
  // Safe moves are added to the safe_moves output paramater.
  bool root_probe_wdl(const Position& pos, std::vector<Move>* safe_moves);

 private:
  template <bool CheckZeroingMoves = false>
  WDLScore search(const Position& pos, ProbeState* result);

  std::string paths_;
  // Caches the max_cardinality from the impl, as max_cardinality may be a hot
  // path.
  int max_cardinality_;
  std::unique_ptr<SyzygyTablebaseImpl> impl_;
};

}  // namespace lczero

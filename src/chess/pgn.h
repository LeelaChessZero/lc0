/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

#pragma once

#include <zlib.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <fstream>

#include "chess/bitboard.h"
#include "chess/board.h"
#include "utils/exception.h"
#include "utils/logging.h"

namespace lczero {

struct Opening {
  std::string start_fen = ChessBoard::kStartposFen;
  MoveList moves;
};

inline bool GzGetLine(gzFile file, std::string& line) {
  bool flag = false;
  char s[2000];
  line.clear();
  while (gzgets(file, s, sizeof(s))) {
    flag = true;
    line += s;
    auto r = line.find_last_of('\n');
    if (r != std::string::npos) {
      line.erase(r);
      break;
    }
  }
  return flag;
}

class PgnReader {
 public:
  void AddPgnFile(const std::string& filepath) {
    const gzFile file = gzopen(filepath.c_str(), "r");
    if (!file) {
      throw Exception(errno == ENOENT ? "Opening book file not found."
                                      : "Error opening opening book file.");
    }

    std::string line;
    bool in_comment = false;
    bool started = false;
    while (GzGetLine(file, line)) {
      // Check if we have a UTF-8 BOM. If so, just ignore it.
      // Only supposed to exist in the first line, but should not matter.
      if (line.substr(0,3) == "\xEF\xBB\xBF") line = line.substr(3);
      if (!line.empty() && line.back() == '\r') line.pop_back();
      // TODO: support line breaks in tags to ensure they are properly ignored.
      if (line.empty() || line[0] == '[') {
        if (started) {
          Flush();
          started = false;
        }
        auto uc_line = line;
        std::transform(
            uc_line.begin(), uc_line.end(), uc_line.begin(),
            [](unsigned char c) { return std::toupper(c); }  // correct
        );
        if (uc_line.find("[FEN \"", 0) == 0) {
          auto start_trimmed = line.substr(6);
          cur_startpos_ = start_trimmed.substr(0, start_trimmed.find('"'));
          cur_board_.SetFromFen(cur_startpos_);
        }
        continue;
      }
      // Must have at least one non-tag non-empty line in order to be considered
      // a game.
      started = true;
      // Handle braced comments.
      int cur_offset = 0;
      while ((in_comment && line.find('}', cur_offset) != std::string::npos) ||
             (!in_comment && line.find('{', cur_offset) != std::string::npos)) {
        if (in_comment && line.find('}', cur_offset) != std::string::npos) {
          line = line.substr(0, cur_offset) +
                 line.substr(line.find('}', cur_offset) + 1);
          in_comment = false;
        } else {
          cur_offset = line.find('{', cur_offset);
          in_comment = true;
        }
      }
      if (in_comment) {
        line = line.substr(0, cur_offset);
      }
      // Trim trailing comment.
      if (line.find(';') != std::string::npos) {
        line = line.substr(0, line.find(';'));
      }
      if (line.empty()) continue;
      std::istringstream iss(line);
      std::string word;
      while (!iss.eof()) {
        word.clear();
        iss >> word;
        if (word.size() < 2) continue;
        // Trim move numbers from front.
        const auto idx = word.find('.');
        if (idx != std::string::npos) {
          bool all_nums = true;
          for (size_t i = 0; i < idx; i++) {
            if (word[i] < '0' || word[i] > '9') {
              all_nums = false;
              break;
            }
          }
          if (all_nums) {
            word = word.substr(idx + 1);
          }
        }
        // Pure move numbers can be skipped.
        if (word.size() < 2) continue;
        // Ignore score line.
        if (word == "1/2-1/2" || word == "1-0" || word == "0-1" || word == "*")
          continue;
        cur_game_.push_back(SanToMove(word, cur_board_));
        cur_board_.ApplyMove(cur_game_.back());
        // Board ApplyMove wants mirrored for black, but outside code wants
        // normal, so mirror it back again.
        // Check equal to 0 since we've already added the position.
        if ((cur_game_.size() % 2) == 0) {
          cur_game_.back().Mirror();
        }
        cur_board_.Mirror();
      }
    }
    if (started) {
      Flush();
    }
    gzclose(file);
  }
  std::vector<Opening> GetGames() const { return games_; }
  std::vector<Opening>&& ReleaseGames() { return std::move(games_); }

 private:
  void Flush() {
    games_.push_back({cur_startpos_, cur_game_});
    cur_game_.clear();
    cur_board_.SetFromFen(ChessBoard::kStartposFen);
    cur_startpos_ = ChessBoard::kStartposFen;
  }

  Move::Promotion PieceToPromotion(int p) {
    switch (p) {
      case -1:
        return Move::Promotion::None;
      case 2:
        return Move::Promotion::Queen;
      case 3:
        return Move::Promotion::Bishop;
      case 4:
        return Move::Promotion::Knight;
      case 5:
        return Move::Promotion::Rook;
      default:
        // 0 and 1 are pawn and king, which are not legal promotions, other
        // numbers don't correspond to a known piece type.
        CERR << "Unexpected promotion!!";
        throw Exception("Trying to create a move with illegal promotion.");
    }
  }

  Move SanToMove(const std::string& san, const ChessBoard& board) {
    int p = 0;
    size_t idx = 0;
    if (san[0] == 'K') {
      p = 1;
    } else if (san[0] == 'Q') {
      p = 2;
    } else if (san[0] == 'B') {
      p = 3;
    } else if (san[0] == 'N') {
      p = 4;
    } else if (san[0] == 'R') {
      p = 5;
    } else if (san[0] == 'O' && san.size() > 2 && san[1] == '-' &&
               san[2] == 'O') {
      Move m;
      auto king_board = board.kings() & board.ours();
      BoardSquare king_sq(GetLowestBit(king_board.as_int()));
      if (san.size() > 4 && san[3] == '-' && san[4] == 'O') {
        m = Move(BoardSquare(0, king_sq.col()),
                 BoardSquare(0, board.castlings().our_queenside_rook()));
      } else {
        m = Move(BoardSquare(0, king_sq.col()),
                 BoardSquare(0, board.castlings().our_kingside_rook()));
      }
      return m;
    }
    if (p != 0) idx++;
    // Formats e4 1e5 de5 d1e5 - with optional x's - followed by =Q for
    // promotions, and even more characters after that also optional.
    int r1 = -1;
    int c1 = -1;
    int r2 = -1;
    int c2 = -1;
    int p2 = -1;
    bool pPending = false;
    for (; idx < san.size(); idx++) {
      if (san[idx] == 'x') continue;
      if (san[idx] == '=') {
        pPending = true;
        continue;
      }
      if (san[idx] >= '1' && san[idx] <= '8') {
        r1 = r2;
        r2 = san[idx] - '1';
        continue;
      }
      if (san[idx] >= 'a' && san[idx] <= 'h') {
        c1 = c2;
        c2 = san[idx] - 'a';
        continue;
      }
      if (pPending) {
        if (san[idx] == 'Q') {
          p2 = 2;
        } else if (san[idx] == 'B') {
          p2 = 3;
        } else if (san[idx] == 'N') {
          p2 = 4;
        } else if (san[idx] == 'R') {
          p2 = 5;
        }
        pPending = false;
        break;
      }
      break;
    }
    if (r1 == -1 || c1 == -1) {
      // Need to find the from cell based on piece.
      int sr1 = r1;
      int sr2 = r2;
      if (board.flipped()) {
        if (sr1 != -1) sr1 = 7 - sr1;
        sr2 = 7 - sr2;
      }
      BitBoard searchBits;
      if (p == 0) {
        searchBits = (board.pawns() & board.ours());
      } else if (p == 1) {
        searchBits = (board.kings() & board.ours());
      } else if (p == 2) {
        searchBits = (board.queens() & board.ours());
      } else if (p == 3) {
        searchBits = (board.bishops() & board.ours());
      } else if (p == 4) {
        searchBits = (board.knights() & board.ours());
      } else if (p == 5) {
        searchBits = (board.rooks() & board.ours());
      }
      auto plm = board.GenerateLegalMoves();
      int pr1 = -1;
      int pc1 = -1;
      for (BoardSquare sq : searchBits) {
        if (sr1 != -1 && sq.row() != sr1) continue;
        if (c1 != -1 && sq.col() != c1) continue;
        if (std::find(plm.begin(), plm.end(),
                      Move(sq, BoardSquare(sr2, c2), PieceToPromotion(p2))) ==
            plm.end()) {
          continue;
        }
        if (pc1 != -1) {
          CERR << "Ambiguous!!";
          throw Exception("Opening book move seems ambiguous.");
        }
        pr1 = sq.row();
        pc1 = sq.col();
      }
      if (pc1 == -1) {
        CERR << "No Match!!";
        throw Exception("Opening book move seems illegal.");
      }
      r1 = pr1;
      c1 = pc1;
      if (board.flipped()) {
        r1 = 7 - r1;
      }
    }
    Move m(BoardSquare(r1, c1), BoardSquare(r2, c2), PieceToPromotion(p2));
    if (board.flipped()) m.Mirror();
    return m;
  }

  ChessBoard cur_board_{ChessBoard::kStartposFen};
  MoveList cur_game_;
  std::string cur_startpos_ = ChessBoard::kStartposFen;
  std::vector<Opening> games_;
};

}  // namespace lczero

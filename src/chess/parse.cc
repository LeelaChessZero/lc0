/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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

#include "chess/parse.h"

#include "utils/exception.h"

namespace lczero {

/*
    Move ParseMove(const ChessBoard& board, std::string_view move_str,
               bool flip_if_black) {
  if (move_str.size() < 4) {
    throw Exception("Invalid move string: " + std::string(move_str));
  }
  File from_file = File::Parse(move_str[0]);
  Rank from_rank = Rank::Parse(move_str[1]);
  File to_file = File::Parse(move_str[2]);
  Rank to_rank = Rank::Parse(move_str[3]);
  if (!from_file.on_board() || !from_rank.on_board() || !to_file.on_board() ||
      !to_rank.on_board()) {
    throw Exception("Invalid move string: " + std::string(move_str));
  }
}
  */

}  // namespace lczero
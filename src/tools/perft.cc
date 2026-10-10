/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#include "tools/perft.h"

#include <cmath>
#include <iostream>

#include "chess/board.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace {
const OptionId kFen{"fen", "", "Perft initial position FEN.", 'f'};
const OptionId kDepth{"depth", "", "Depth to reach.", 'd'};

std::uint64_t DoPerft(const ChessBoard& board, int max_depth, bool dump = false,
                      int depth = 0) {
  if (depth == max_depth) return 1;
  std::uint64_t total_count = 0;

  auto moves = board.GenerateLegalMoves();
  if (depth == max_depth - 1) return moves.size();

  for (const auto& move : moves) {
    auto new_board = board;
    new_board.ApplyMove(move);

    new_board.Mirror();
    auto count = DoPerft(new_board, max_depth, dump, depth + 1);
    if (dump && depth == 0) {
      Move m = move;
      if (board.flipped()) m.Flip();
      std::cout << m.ToString(true) << ": " << count << std::endl;
    }
    total_count += count;
  }

  return total_count;
}

}  // namespace

void Perft::Run() {
  OptionsParser options;
  options.Add<StringOption>(kFen) = ChessBoard::kStartposFen;
  options.Add<IntOption>(kDepth, 1, 15) = 6;

  if (!options.ProcessAllFlags()) return;
  auto option_dict = options.GetOptionsDict();
  const auto start = std::chrono::steady_clock::now();
  const auto perft = DoPerft(ChessBoard(option_dict.Get<std::string>(kFen)),
                             option_dict.Get<int>(kDepth), true);
  const auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  std::cout << std::endl;
  std::cout << "Positions searched: " << perft << std::endl;
  std::cout << "Positions / second: "
            << std::lround(1000.0 * perft / (time.count() + 1)) << std::endl;
  return;
}
}  // namespace lczero

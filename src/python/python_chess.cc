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

#include <pybind11/pybind11.h>
#include <string>
#include <vector>

#include "python/python_chess.h"
#include "python/weights.h"

namespace py = pybind11;

namespace lczero {
namespace python {
namespace python_chess {

BoardData GetBoardData(const py::handle& board) {
    BoardData board_data;
    
    py::object board_copy = board.attr("copy")();
    auto move_stack = board.attr("move_stack");
    for (auto _ : move_stack) {
        board_copy.attr("pop")();
    }
    
    board_data.fen = board_copy.attr("fen")().cast<std::string>();
  
    board_data.is_c960 = py::hasattr(board, "chess960") ? 
                         board.attr("chess960").cast<bool>() : false;
    
    for (auto move : move_stack) {
        board_data.moves.push_back(move.attr("uci")().cast<std::string>());
    }
    
    return board_data;
}

py::list UciMovesToChessMoves(const std::vector<std::string>& uci_moves) {
    py::object chess_module = py::module::import("chess");
    py::object Move = chess_module.attr("Move");

    py::list result;
    for (const auto& uci : uci_moves) {
        result.append(Move.attr("from_uci")(uci));
    }
    return result;
}

py::object ToBoard(const GameState& gs) {
  py::object chess = py::module::import("chess");

  py::object board = chess.attr("Board")(
      gs.startpos().value_or(ChessBoard::kStartposFen));

  for (const auto& move : gs.move_history()) {
    board.attr("push_uci")(move);
  }

  return board;
}

void PushMove(GameState& gs, const py::handle& move) {
  gs.push_uci(move.attr("uci")().cast<std::string>());
}

} // namespace python_chess
} // namespace python
} // namespace lczero

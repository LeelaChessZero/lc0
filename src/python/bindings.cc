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
#include <pybind11/stl.h>
 
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
 
#include "chess/bitboard.h"
#include "python/weights.h"
#include "python/python_chess.h"

#include "utils/exception.h"
 
namespace py = pybind11;
 
PYBIND11_MODULE(backends, m) {
  lczero::InitializeMagicBitboards();
  static py::exception<lczero::Exception> py_lczero_exception(
      m, "LczeroException");
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const lczero::Exception& e) {
      py_lczero_exception(e.what());
    }
  });

  // Weights
  py::class_<lczero::python::Weights> weights(m, "Weights");
  weights.def(py::init<const std::optional<std::string>&>(),
              py::arg("filename") = std::nullopt);
  weights.def("filename", &lczero::python::Weights::filename);
  weights.def("license", &lczero::python::Weights::license);
  weights.def("min_version", &lczero::python::Weights::min_version);
  weights.def("input_format", &lczero::python::Weights::input_format);
  weights.def("policy_format", &lczero::python::Weights::policy_format);
  weights.def("value_format", &lczero::python::Weights::value_format);
  weights.def("moves_left_format",
              &lczero::python::Weights::moves_left_format);
  weights.def("blocks", &lczero::python::Weights::blocks);
  weights.def("filters", &lczero::python::Weights::filters);

  // Input
  py::class_<lczero::python::Input> input(m, "Input");
  input.def(py::init<>());
  input.def("set_mask", &lczero::python::Input::set_mask, py::arg("plane"),
            py::arg("mask"));
  input.def("set_val", &lczero::python::Input::set_val, py::arg("plane"),
            py::arg("value"));
  input.def("mask", &lczero::python::Input::mask, py::arg("plane"));
  input.def("val", &lczero::python::Input::val, py::arg("plane"));
  input.def("clone", &lczero::python::Input::clone);

  // Output
  py::class_<lczero::python::Output> output(m, "Output");
  output.def("q", &lczero::python::Output::q);
  output.def("d", &lczero::python::Output::d);
  output.def("m", &lczero::python::Output::m);
  output.def("p_raw", &lczero::python::Output::p_raw, py::arg("samples"));
  output.def("p_softmax", &lczero::python::Output::p_softmax,
              py::arg("samples"));

  // BackendCapabilities
  py::class_<lczero::python::BackendCapabilities> backend_caps(
      m, "BackendCapabilities");
  backend_caps.def("input_format",
                    &lczero::python::BackendCapabilities::input_format);
  backend_caps.def("moves_left_format",
                    &lczero::python::BackendCapabilities::moves_left_format);

  // Backend
  py::class_<lczero::python::Backend> backend(m, "Backend");
  backend.def_static("available_backends",
                      &lczero::python::Backend::available_backends);
  backend.def(py::init<const lczero::python::Weights*,
                        const std::optional<std::string>&,
                        const std::optional<std::string>&>(),
              py::arg("weights") = nullptr, py::arg("backend") = std::nullopt,
              py::arg("options") = std::nullopt);
  backend.def("evaluate", &lczero::python::Backend::evaluate,
              py::arg("inputs"));
  backend.def("capabilities", &lczero::python::Backend::capabilities);

  // GameState
  py::class_<lczero::python::GameState> game_state(m, "GameState");
  game_state.def(py::init<const std::optional<std::string>,
                           const std::vector<std::string>&,
                           const bool>(),
                  py::arg("fen") = std::nullopt,
                  py::arg("moves") = std::vector<std::string>{},
                  py::arg("is_c960") = false);
  game_state.def(py::init([](py::object board) {
      const auto boardData = 
        lczero::python::python_chess::GetBoardData(board);
      return std::make_unique<lczero::python::GameState>(
        boardData.fen, boardData.moves, boardData.is_c960);
    }), py::arg("board"));                  
  game_state.def("as_input", &lczero::python::GameState::as_input,
                  py::arg("backend"));
  game_state.def("moves", &lczero::python::GameState::moves);
  game_state.def("policy_indices",
                  &lczero::python::GameState::policy_indices);
  game_state.def("as_string", &lczero::python::GameState::as_string);
}

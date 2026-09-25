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
#include <sstream>
 
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

  // Py Enum bindings so that the user does not get an int output
  py::enum_<pblczero::NetworkFormat::InputFormat>(m, "InputFormat")
      .value("INPUT_UNKNOWN", pblczero::NetworkFormat::INPUT_UNKNOWN)
      .value("INPUT_CLASSICAL_112_PLANE", 
             pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE)
      .value("INPUT_112_WITH_CASTLING_PLANE",
             pblczero::NetworkFormat::INPUT_112_WITH_CASTLING_PLANE)
      .value("INPUT_112_WITH_CANONICALIZATION",
             pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION)
      .value("INPUT_112_WITH_CANONICALIZATION_HECTOPLIES",
             pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_HECTOPLIES)
      .value("INPUT_112_WITH_CANONICALIZATION_V2",
             pblczero::NetworkFormat::INPUT_112_WITH_CANONICALIZATION_V2)
      .export_values();

  py::enum_<pblczero::NetworkFormat::PolicyFormat>(m, "PolicyFormat")
      .value("POLICY_UNKNOWN", pblczero::NetworkFormat::POLICY_UNKNOWN)
      .value("POLICY_CLASSICAL", pblczero::NetworkFormat::POLICY_CLASSICAL)
      .value("POLICY_CONVOLUTION", pblczero::NetworkFormat::POLICY_CONVOLUTION)
      .value("POLICY_ATTENTION", pblczero::NetworkFormat::POLICY_ATTENTION)
      .export_values();

  py::enum_<pblczero::NetworkFormat::ValueFormat>(m, "ValueFormat")
      .value("VALUE_UNKNOWN", pblczero::NetworkFormat::VALUE_UNKNOWN)
      .value("VALUE_CLASSICAL", pblczero::NetworkFormat::VALUE_CLASSICAL)
      .value("VALUE_WDL", pblczero::NetworkFormat::VALUE_WDL)
      .value("VALUE_PARAM", pblczero::NetworkFormat::VALUE_PARAM)
      .export_values();

  py::enum_<pblczero::NetworkFormat::MovesLeftFormat>(m, "MovesLeftFormat")
      .value("MOVES_LEFT_NONE", pblczero::NetworkFormat::MOVES_LEFT_NONE)
      .value("MOVES_LEFT_V1", pblczero::NetworkFormat::MOVES_LEFT_V1)
      .export_values();

  py::enum_<pblczero::NetworkFormat::OutputFormat>(m, "OutputFormat")
      .value("OUTPUT_UNKNOWN", pblczero::NetworkFormat::OUTPUT_UNKNOWN)
      .value("OUTPUT_CLASSICAL", pblczero::NetworkFormat::OUTPUT_CLASSICAL)
      .value("OUTPUT_WDL", pblczero::NetworkFormat::OUTPUT_WDL)
      .export_values();

  py::enum_<pblczero::NetworkFormat::NetworkStructure>(m, "NetworkStructure")
      .value("NETWORK_UNKNOWN", pblczero::NetworkFormat::NETWORK_UNKNOWN)
      .value("NETWORK_CLASSICAL", pblczero::NetworkFormat::NETWORK_CLASSICAL)
      .value("NETWORK_SE", pblczero::NetworkFormat::NETWORK_SE)
      .value("NETWORK_CLASSICAL_WITH_HEADFORMAT", 
             pblczero::NetworkFormat::NETWORK_CLASSICAL_WITH_HEADFORMAT)
      .value("NETWORK_SE_WITH_HEADFORMAT", 
             pblczero::NetworkFormat::NETWORK_SE_WITH_HEADFORMAT)
      .value("NETWORK_ONNX", pblczero::NetworkFormat::NETWORK_ONNX)
      .value("NETWORK_ATTENTIONBODY_WITH_HEADFORMAT",
             pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT)
      .value("NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT",
             pblczero::NetworkFormat::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT)
      .export_values();

  py::enum_<pblczero::Format::Encoding>(m, "WeightsEncoding")
      .value("UNKNOWN", pblczero::Format::UNKNOWN)
      .value("LINEAR16", pblczero::Format::LINEAR16)
      .export_values();

  // Weights
  py::class_<lczero::python::Weights> weights(m, "Weights");
  weights.def(py::init<const std::optional<std::string>&>(),
              py::arg("filename") = std::nullopt);
  weights.def("filename", &lczero::python::Weights::filename);
  weights.def("license", &lczero::python::Weights::license);
  weights.def("min_version", &lczero::python::Weights::min_version);
  weights.def("input_format", [](const lczero::python::Weights& w) {
    return static_cast<pblczero::NetworkFormat::InputFormat>(
        w.input_format());
  });
  weights.def("policy_format", [](const lczero::python::Weights& w) {
    return static_cast<pblczero::NetworkFormat::PolicyFormat>(
        w.policy_format());
  });
  weights.def("value_format", [](const lczero::python::Weights& w) {
    return static_cast<pblczero::NetworkFormat::ValueFormat>(
        w.value_format());
  });
  weights.def("moves_left_format", [](const lczero::python::Weights& w) {
    return static_cast<pblczero::NetworkFormat::MovesLeftFormat>(
        w.moves_left_format());
  });
  
  weights.def("blocks", &lczero::python::Weights::blocks);
  weights.def("filters", &lczero::python::Weights::filters);
  weights.def("is_transformer", &lczero::python::Weights::is_transformer);
  weights.def("attention_heads", &lczero::python::Weights::attention_heads);
  weights.def("embedding_size", &lczero::python::Weights::embedding_size);
  weights.def("__repr__", [](const lczero::python::Weights& w) {
    std::ostringstream oss;
    oss << "Weights(filename='" << w.filename() 
        << "', input_format=" << w.input_format()
        << ", policy_format=" << w.policy_format()
        << ", value_format=" << w.value_format()
        << ", moves_left_format=" << w.moves_left_format()
        << ", blocks=" << w.blocks()
        << ", filters=" << w.filters()
        << ", blocks=" << w.is_transformer()
        << ", blocks=" << w.attention_heads()
        << ", blocks=" << w.embedding_size() << ")";
    return oss.str();
  });
  weights.def("__eq__", [](const lczero::python::Weights& w1, const py::object& obj) {
    if (!py::isinstance<lczero::python::Weights>(obj)) {
      return false;
    }
    const auto& w2 = obj.cast<const lczero::python::Weights&>();
    return w1.filename() == w2.filename() &&
           w1.input_format() == w2.input_format() &&
           w1.policy_format() == w2.policy_format() &&
           w1.value_format() == w2.value_format() &&
           w1.moves_left_format() == w2.moves_left_format() &&
           w1.blocks() == w2.blocks() &&
           w1.filters() == w2.filters() &&
           w1.is_transformer() == w2.is_transformer() &&
           w1.attention_heads() == w2.attention_heads() &&
           w1.embedding_size() == w2.embedding_size();
  });

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
  output.def("__repr__", [](const lczero::python::Output& o) {
    std::ostringstream oss;
    oss << "Output(q=" << o.q() << ", d=" << o.d() << ", m=" << o.m() << ")";
    return oss.str();
  });
  output.def("__eq__", [](const lczero::python::Output& o1, const py::object& obj) {
    if (!py::isinstance<lczero::python::Output>(obj)) {
      return false;
    }
    const auto& o2 = obj.cast<const lczero::python::Output&>();
    const float epsilon = 1e-6f;
    return std::abs(o1.q() - o2.q()) < epsilon &&
           std::abs(o1.d() - o2.d()) < epsilon &&
           std::abs(o1.m() - o2.m()) < epsilon;
  });

  // BackendCapabilities
  py::class_<lczero::python::BackendCapabilities> backend_caps(
      m, "BackendCapabilities");
  backend_caps.def("input_format",
                    [](const lczero::python::BackendCapabilities& bc) {
                      return static_cast<pblczero::NetworkFormat::InputFormat>(
                          bc.input_format());
                    });
  backend_caps.def("moves_left_format",
                    [](const lczero::python::BackendCapabilities& bc) {
                      return static_cast<pblczero::NetworkFormat::MovesLeftFormat>(
                          bc.moves_left_format());
                    });
  backend_caps.def("__repr__", [](const lczero::python::BackendCapabilities& bc) {
    std::ostringstream oss;
    oss << "BackendCapabilities(input_format=" << bc.input_format()
        << ", moves_left_format=" << bc.moves_left_format() << ")";
    return oss.str();
  });
  backend_caps.def("__eq__", [](const lczero::python::BackendCapabilities& bc1, const py::object& obj) {
    if (!py::isinstance<lczero::python::BackendCapabilities>(obj)) {
      return false;
    }
    const auto& bc2 = obj.cast<const lczero::python::BackendCapabilities&>();
    return bc1.input_format() == bc2.input_format() &&
           bc1.moves_left_format() == bc2.moves_left_format();
  });

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
  game_state.def("moves", [](const lczero::python::GameState& gs) {
    return lczero::python::python_chess::UciMovesToChessMoves(gs.moves());
  });
  game_state.def("policy_indices",
                  &lczero::python::GameState::policy_indices);
  game_state.def("as_string", &lczero::python::GameState::as_string);
  game_state.def("push_uci", &lczero::python::GameState::push_uci, py::arg("move"));
  game_state.def("to_board", &lczero::python::python_chess::ToBoard);
}

/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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

#include "neural/xla/print_hlo.h"

namespace lczero {
namespace {

std::string CEscape(std::string_view str) {
  std::string result = "\"";
  for (char c : str) {
    switch (c) {
      case '\n':
        result += "\\n";
        break;
      case '\t':
        result += "\\t";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\v':
        result += "\\v";
        break;
      case '\f':
        result += "\\f";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\"':
        result += "\\\"";
        break;
      case '\'':
        result += "\\\'";
        break;
      default:
        result += c;
    }
  }
  return result + "\"";
}

class HloPrettyPrinter {
 public:
  HloPrettyPrinter(PrettyPrintHloOptions options, std::ostream& stream)
      : options_(options), s_(stream) {}

  void PrintModule(const pblczero::HloModuleProto& module) {
    current_module_ = &module;
    s_ << "HloModule " << module.name();
    if (module.has_host_program_shape()) {
      s_ << ", entry_computation_layout=";
      PrintProgramShape(module.host_program_shape());
    }
    s_ << "\n";

    for (const auto& computation : module.computations()) {
      s_ << "\n";
      if (module.entry_computation_id() == computation.id()) s_ << "ENTRY ";
      PrintComputation(computation);
    }
    current_module_ = nullptr;
  }

 private:
  template <typename T, typename F>
  void PrintDelimeted(const T& vec, F print_fn, std::string_view delim,
                      std::string_view prefix = "",
                      std::string_view suffix = "") {
    s_ << prefix;
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) s_ << delim;
      print_fn(vec[i]);
    }
    s_ << suffix;
  }

  std::string GetTypeLiteral(pblczero::XlaShapeProto::Type type) {
    std::string name = pblczero::XlaShapeProto::Type_Name(type);
    for (char& c : name) c = std::tolower(c);
    return name;
  }

  void PrintLayout(const pblczero::XlaLayoutProto& layout) {
    if (!options_.print_layout) return;
    PrintDelimeted(
        layout.minor_to_major(), [&](const auto& dim) { s_ << dim; }, ",", " {",
        "}");
  }

  void PrintShape(const pblczero::XlaShapeProto& shape) {
    if (shape.element_type() == pblczero::XlaShapeProto::TUPLE) {
      PrintDelimeted(
          shape.tuple_shapes(), [&](const auto& s) { PrintShape(s); }, ", ",
          "(", ")");
      return;
    }
    s_ << GetTypeLiteral(shape.element_type());
    PrintDelimeted(
        shape.dimensions(), [&](int64_t dim) { s_ << dim; }, ",", "[", "]");
    if (shape.has_layout()) PrintLayout(shape.layout());
  }

  void PrintProgramShape(const pblczero::XlaProgramShapeProto& shape) {
    s_ << "{(";
    for (size_t i = 0; i < shape.parameters_size(); ++i) {
      if (i > 0) s_ << ", ";
      if (shape.parameter_names_size() > i &&
          !shape.parameter_names(i).empty()) {
        s_ << shape.parameter_names(i) << ": ";
      }
      PrintShape(shape.parameters(i));
    }
    s_ << ") -> ";
    PrintShape(shape.result());
    s_ << "}";
  }

  void PrintLiteral(const pblczero::XlaLiteralProto& literal) {
    // For now just print as a flat array with sometimes wrong encoding (i.e. in
    // bf16 case).
    auto print_array = [&](const auto& array) {
      PrintDelimeted(
          array,
          [&](const auto& x) {
            if constexpr (std::is_same_v<std::decay_t<decltype(x)>, char> ||
                          std::is_same_v<std::decay_t<decltype(x)>, bool>) {
              s_ << static_cast<int>(x);
            } else {
              s_ << x;
            }
          },
          ",");
    };
    switch (literal.shape().element_type()) {
      case pblczero::XlaShapeProto::TUPLE:
        PrintDelimeted(
            literal.tuple_literals(), [&](const auto& l) { PrintLiteral(l); },
            ", ", "(", ")");
        break;
      case pblczero::XlaShapeProto::TOKEN:
        s_ << "token";
        break;
      case pblczero::XlaShapeProto::PRED:
        return print_array(literal.preds());
      case pblczero::XlaShapeProto::S4:
        return print_array(literal.s4s());
      case pblczero::XlaShapeProto::U4:
        return print_array(literal.u4s());
      case pblczero::XlaShapeProto::S8:
        return print_array(literal.s8s());
      case pblczero::XlaShapeProto::U8:
        return print_array(literal.u8s());
      case pblczero::XlaShapeProto::S32:
        return print_array(literal.s32s());
      case pblczero::XlaShapeProto::S64:
        return print_array(literal.s64s());
      case pblczero::XlaShapeProto::U32:
        return print_array(literal.u32s());
      case pblczero::XlaShapeProto::U64:
        return print_array(literal.u64s());
      case pblczero::XlaShapeProto::F32:
        return print_array(literal.f32s());
      case pblczero::XlaShapeProto::F64:
        return print_array(literal.f64s());
      case pblczero::XlaShapeProto::C64:
        return print_array(literal.c64s());
      case pblczero::XlaShapeProto::C128:
        return print_array(literal.c128s());
      case pblczero::XlaShapeProto::F16:
        return print_array(literal.f16s());
      case pblczero::XlaShapeProto::BF16:
        return print_array(literal.bf16s());
      case pblczero::XlaShapeProto::U16:
        return print_array(literal.u16s());
      case pblczero::XlaShapeProto::S16:
        return print_array(literal.s16s());
      case pblczero::XlaShapeProto::F8E5M2:
        return print_array(literal.f8e5m2s());
      case pblczero::XlaShapeProto::F8E4M3FN:
        return print_array(literal.f8e4m3fns());
      case pblczero::XlaShapeProto::F8E4M3B11FNUZ:
        return print_array(literal.f8e4m3b11fnuzs());
      case pblczero::XlaShapeProto::F8E5M2FNUZ:
        return print_array(literal.f8e5m2fnuzs());
      case pblczero::XlaShapeProto::F8E4M3FNUZ:
        return print_array(literal.f8e4m3fnuzs());
      case pblczero::XlaShapeProto::PRIMITIVE_TYPE_INVALID:
        s_ << "INVALID";
        break;
      case pblczero::XlaShapeProto::OPAQUE_TYPE:
        s_ << "opaque";
        break;
    }
  }

  void PrintInstructionOperands(
      const pblczero::HloInstructionProto& instruction) {
    s_ << "(";
    if (instruction.opcode() == "parameter") {
      s_ << instruction.parameter_number();
    } else if (instruction.opcode() == "get-tuple-index") {
      s_ << instruction.tuple_index();
    } else if (instruction.opcode() == "constant") {
      PrintLiteral(instruction.literal());
    } else {
      PrintDelimeted(
          instruction.operand_ids(),
          [&](int64_t id) {
            s_ << "%" << current_computation_->instructions(id).name();
          },
          ", ");
    }
    s_ << ")";
  }

  void PrintWindow(const pblczero::XlaWindow& window) {
    PrintDelimeted(
        window.dimensions(), [&](const auto& d) { s_ << d.size(); }, "x",
        "size=");
    PrintDelimeted(
        window.dimensions(),
        [&](const auto& d) {
          s_ << d.padding_low() << "_" << d.padding_high();
        },
        "x", " pads=");
  }

  void PrintDotDimensionNumbers(const pblczero::XlaDotDimensionNumbers& dn) {
    PrintDelimeted(
        dn.lhs_batch_dimensions(), [&](int64_t dim) { s_ << dim; }, ",",
        ", lhs_batch_dims={", "}");
    PrintDelimeted(
        dn.rhs_batch_dimensions(), [&](int64_t dim) { s_ << dim; }, ",",
        ", rhs_batch_dims={", "}");
    PrintDelimeted(
        dn.lhs_contracting_dimensions(), [&](int64_t dim) { s_ << dim; }, ",",
        ", lhs_contracting_dims={", "}");
    PrintDelimeted(
        dn.rhs_contracting_dimensions(), [&](int64_t dim) { s_ << dim; }, ",",
        ", rhs_contracting_dims={", "}");
  }

  void PrintConvolutionDimensionNumbers(
      const pblczero::XlaConvolutionDimensionNumbers& dn) {
    std::string input_dims(dn.input_spatial_dimensions_size() + 2, '?');
    std::string kernel_dims(dn.kernel_spatial_dimensions_size() + 2, '?');
    std::string output_dims(dn.output_spatial_dimensions_size() + 2, '?');
    input_dims[dn.input_batch_dimension()] = 'b';
    input_dims[dn.input_feature_dimension()] = 'f';
    kernel_dims[dn.kernel_output_feature_dimension()] = 'o';
    kernel_dims[dn.kernel_input_feature_dimension()] = 'i';
    output_dims[dn.output_batch_dimension()] = 'b';
    output_dims[dn.output_feature_dimension()] = 'f';
    for (size_t i = 0; i < dn.input_spatial_dimensions_size(); ++i) {
      input_dims[dn.input_spatial_dimensions(i)] = '0' + i;
      kernel_dims[dn.kernel_spatial_dimensions(i)] = '0' + i;
      output_dims[dn.output_spatial_dimensions(i)] = '0' + i;
    }
    s_ << input_dims << "_" << kernel_dims << "->" << output_dims;
  }

  void PrintInstructionAttributes(
      const pblczero::HloInstructionProto& instruction) {
    if (instruction.called_computation_ids_size() > 0) {
      PrintDelimeted(
          instruction.called_computation_ids(),
          [&](int64_t id) { s_ << current_module_->computations(id).name(); },
          ",", ", calls={", "}");
    }
    if (instruction.has_window()) {
      s_ << ", window={";
      PrintWindow(instruction.window());
      s_ << "}";
    }
    if (instruction.has_convolution_dimension_numbers()) {
      s_ << ", dim_labels=";
      PrintConvolutionDimensionNumbers(
          instruction.convolution_dimension_numbers());
    }
    if (instruction.dimensions_size() > 0) {
      PrintDelimeted(
          instruction.dimensions(), [&](int64_t dim) { s_ << dim; }, ", ",
          ", dimensions={", "}");
    }
    if (instruction.has_dot_dimension_numbers()) {
      PrintDotDimensionNumbers(instruction.dot_dimension_numbers());
    }
  }

  void PrintInstructionMetadata(
      const pblczero::HloInstructionProto& instruction) {
    if (instruction.has_metadata()) {
      const auto& m = instruction.metadata();
      s_ << ", metadata={";
      bool first = true;
      auto sep = [&]() -> std::ostream& {
        if (!first) s_ << ", ";
        first = false;
        return s_;
      };
      std::vector<std::string> bits;
      if (m.has_op_type()) sep() << "op_type=" << CEscape(m.op_type());
      if (m.has_op_name()) sep() << "op_name=" << CEscape(m.op_name());
      if (m.has_source_file())
        sep() << "source_file=" << CEscape(m.source_file());
      if (m.has_source_line()) sep() << "source_line=" << m.source_line();
      s_ << "}";
    }
  }

  void PrintInstruction(const pblczero::HloInstructionProto& instruction) {
    s_ << "%" << instruction.name() << " = ";
    PrintShape(instruction.shape());
    s_ << " " << instruction.opcode();
    PrintInstructionOperands(instruction);
    PrintInstructionAttributes(instruction);
    PrintInstructionMetadata(instruction);
  }

  void PrintComputation(const pblczero::HloComputationProto& computation) {
    current_computation_ = &computation;
    s_ << computation.name() << " {\n";
    for (const auto& instruction : computation.instructions()) {
      s_ << "    ";
      if (computation.root_id() == instruction.id()) s_ << "ROOT ";
      PrintInstruction(instruction);
      s_ << "\n";
    }
    s_ << "}\n";
    current_computation_ = nullptr;
  }

  PrettyPrintHloOptions options_;
  const pblczero::HloModuleProto* current_module_ = nullptr;
  const pblczero::HloComputationProto* current_computation_ = nullptr;
  std::ostream& s_;
};

}  // namespace

void PrettyPrintHlo(const pblczero::HloModuleProto& module,
                    PrettyPrintHloOptions options, std::ostream& stream) {
  HloPrettyPrinter(options, stream).PrintModule(module);
}

}  // namespace lczero
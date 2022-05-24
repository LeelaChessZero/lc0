/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include "neural/onnx/builder.h"

#include <initializer_list>

#include "neural/onnx/onnx.pb.h"
#include "utils/random.h"
#include "version.h"

namespace lczero {

OnnxBuilder::OnnxBuilder() {
  model_.set_ir_version(4);
  model_.set_domain("org.lczero.models.*");
  model_.set_producer_name("Lc0");
  model_.set_producer_version(GetVersionStr());
  model_.add_opset_import()->set_version(9);
  model_.mutable_graph()->set_name("org.lczero/converted/" +
                                   Random::Get().GetString(16));
}

namespace {
void FillValueInfo(pblczero::ValueInfoProto* vip, const std::string& name,
                   std::initializer_list<int> dims,
                   pblczero::TensorProto::DataType datatype) {
  vip->set_name(name);
  auto* type = vip->mutable_type()->mutable_tensor_type();
  type->set_elem_type(datatype);
  auto* shape = type->mutable_shape();
  for (const auto d : dims) {
    auto* dim = shape->add_dim();
    if (d < 0) {
      dim->set_dim_param("batch");
    } else {
      dim->set_dim_value(d);
    }
  }
}

void AddIntAttribute(pblczero::NodeProto* node, const std::string& name,
                     int val) {
  auto* attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(pblczero::AttributeProto::INT);
  attr->set_i(val);
}

void AddIntsAttribute(pblczero::NodeProto* node, const std::string& name,
                      std::initializer_list<int> vals) {
  auto* attr = node->add_attribute();
  attr->set_name(name);
  attr->set_type(pblczero::AttributeProto::INTS);
  for (const int x : vals) attr->add_ints(x);
}

}  // namespace

void OnnxBuilder::AddInput(const std::string& name,
                           std::initializer_list<int> dims,
                           pblczero::TensorProto::DataType datatype) {
  FillValueInfo(model_.mutable_graph()->add_input(), name, dims, datatype);
}

void OnnxBuilder::AddOutput(const std::string& name,
                            std::initializer_list<int> dims,
                            pblczero::TensorProto::DataType datatype) {
  FillValueInfo(model_.mutable_graph()->add_output(), name, dims, datatype);
}

std::string OnnxBuilder::AddInitializer(const std::string& name,
                                        const OnnxConst& weights) {
  auto* init = model_.mutable_graph()->add_initializer();
  init->set_name(name);
  init->set_data_type(weights.GetDataType());
  for (const int dim : weights.GetDimensions()) init->add_dims(dim);
  init->set_raw_data(weights.GetRawData());

  return name;
}

namespace {

std::string PopulateStdNodeFields(pblczero::NodeProto* node,
                                  const std::string& name,
                                  const std::string& input,
                                  const std::string& type) {
  node->set_name(name);
  node->set_op_type(type);
  node->add_input(input);
  node->add_output(name);
  return name;
}

}  // namespace

std::string OnnxBuilder::Conv(const std::string& name,
                              const std::string& input_name,
                              const OnnxConst& kernel_weights,
                              const OnnxConst& bias_weights, int pads) {
  auto* node = model_.mutable_graph()->add_node();
  auto out = PopulateStdNodeFields(node, name, input_name, "Conv");
  node->add_input(AddInitializer(name + "/w/kernel", kernel_weights));
  node->add_input(AddInitializer(name + "/w/bias", bias_weights));
  AddIntsAttribute(node, "pads", {pads, pads, pads, pads});
  return out;
}

std::string OnnxBuilder::Add(const std::string& name, const std::string& input1,
                             const std::string& input2) {
  auto* node = model_.mutable_graph()->add_node();
  auto out = PopulateStdNodeFields(node, name, input1, "Add");
  node->add_input(input2);
  return out;
}

std::string OnnxBuilder::Add(const std::string& name, const std::string& input1,
                             const OnnxConst& input2) {
  auto* node = model_.mutable_graph()->add_node();
  auto out = PopulateStdNodeFields(node, name, input1, "Add");
  node->add_input(AddInitializer(name + "/w", input2));
  return out;
}

std::string OnnxBuilder::GlobalAveragePool(const std::string& name,
                                           const std::string& input) {
  auto* node = model_.mutable_graph()->add_node();
  return PopulateStdNodeFields(node, name, input, "GlobalAveragePool");
}

std::string OnnxBuilder::Squeeze(const std::string& name,
                                 const std::string& input) {
  auto* node = model_.mutable_graph()->add_node();
  auto out = PopulateStdNodeFields(node, name, input, "Squeeze");
  AddIntsAttribute(node, "axes", {2, 3});
  return out;
}

std::string OnnxBuilder::Mul(const std::string& name, const std::string& input1,
                             const std::string& input2) {
  auto* node = model_.mutable_graph()->add_node();
  auto out = PopulateStdNodeFields(node, name, input1, "Mul");
  node->add_input(input2);
  return out;
}

std::string OnnxBuilder::MatMul(const std::string& name,
                                const std::string& input1,
                                const OnnxConst& input2) {
  auto* node = model_.mutable_graph()->add_node();
  auto out = PopulateStdNodeFields(node, name, input1, "MatMul");
  node->add_input(AddInitializer(name + "/w", input2));
  return out;
}

std::string OnnxBuilder::Relu(const std::string& name,
                              const std::string& input) {
  auto* node = model_.mutable_graph()->add_node();
  return PopulateStdNodeFields(node, name, input, "Relu");
}

std::string OnnxBuilder::Tanh(const std::string& name,
                              const std::string& input) {
  auto* node = model_.mutable_graph()->add_node();
  return PopulateStdNodeFields(node, name, input, "Tanh");
}

std::string OnnxBuilder::Softmax(const std::string& name,
                                 const std::string& input) {
  auto* node = model_.mutable_graph()->add_node();
  return PopulateStdNodeFields(node, name, input, "Softmax");
}

std::string OnnxBuilder::Reshape(const std::string& name,
                                 const std::string& input,
                                 const std::string& shape) {
  auto* node = model_.mutable_graph()->add_node();
  auto out = PopulateStdNodeFields(node, name, input, "Reshape");
  node->add_input(shape);
  return out;
}

std::string OnnxBuilder::Gather(const std::string& name,
                                const std::string& input1,
                                const std::string& input2, int axis) {
  auto* node = model_.mutable_graph()->add_node();
  auto out = PopulateStdNodeFields(node, name, input1, "Gather");
  node->add_input(input2);
  AddIntAttribute(node, "axis", axis);
  return out;
}

std::pair<std::string, std::string> OnnxBuilder::Split(const std::string& name,
                                                       const std::string& input,
                                                       int axis) {
  auto* node = model_.mutable_graph()->add_node();
  node->set_name(name);
  node->set_op_type("Split");
  node->add_input(input);
  node->add_output(name + "/out1");
  node->add_output(name + "/out2");
  AddIntAttribute(node, "axis", axis);
  return {name + "/out1", name + "/out2"};
}

std::string OnnxBuilder::Sigmoid(const std::string& name,
                                 const std::string& input) {
  auto* node = model_.mutable_graph()->add_node();
  return PopulateStdNodeFields(node, name, input, "Sigmoid");
}

}  // namespace lczero
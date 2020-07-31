#!/usr/bin/env python3

#  This file is part of Leela Chess Zero.
#  Copyright (C) 2020 The LCZero Authors
#
#  Leela Chess is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  Leela Chess is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
#
#  Additional permission under GNU GPL version 3 section 7
#
#  If you modify this Program, or any covered work, by linking or
#  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
#  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
#  modified version of those libraries), containing parts covered by the
#  terms of the respective license agreement, the licensors of this
#  Program grant you additional permission to convey the resulting work.

import sys
from pybind.writer import Writer
from pybind import Module, Class
from pybind.parameters import (StringParameter, ClassParameter,
                               NumericParameter, ArgvObjects, IntegralArgv,
                               ListOfStringsParameter)
from pybind.retval import (StringViewRetVal, StringRetVal, ListOfStringsRetVal,
                           NumericRetVal, ObjCopyRetval, ObjOwnerRetval,
                           ObjTupleRetVal, IntegralTupleRetVal)
from pybind.exceptions import CppException

# Module
mod = Module('backends')
mod.AddInclude('python/weights.h')
mod.AddInitialization('lczero::InitializeMagicBitboards();')
ex = mod.AddException(
    CppException('LczeroException', cpp_name='lczero::Exception'))

# Weights class
weights = mod.AddClass(Class('Weights', cpp_name='lczero::python::Weights'))
weights.constructor.AddParameter(
    StringParameter('filename', optional=True, can_be_none=True)).AddEx(ex)
weights.AddMethod('filename').AddRetVal(StringViewRetVal())
weights.AddMethod('license').AddRetVal(StringViewRetVal())
weights.AddMethod('min_version').AddRetVal(StringRetVal())
weights.AddMethod('input_format').AddRetVal(NumericRetVal('i'))
weights.AddMethod('policy_format').AddRetVal(NumericRetVal('i'))
weights.AddMethod('value_format').AddRetVal(NumericRetVal('i'))
weights.AddMethod('moves_left_format').AddRetVal(NumericRetVal('i'))
weights.AddMethod('blocks').AddRetVal(NumericRetVal('i'))
weights.AddMethod('filters').AddRetVal(NumericRetVal('i'))

# Input class
input = mod.AddClass(Class('Input', cpp_name='lczero::python::Input'))
input.AddMethod('set_mask').AddParameter(NumericParameter('plane'),
                                         NumericParameter(
                                             'mask', type='u64')).AddEx(ex)
input.AddMethod('set_val').AddParameter(NumericParameter('plane'),
                                        NumericParameter('value',
                                                         type='f32')).AddEx(ex)
input.AddMethod('mask').AddParameter(NumericParameter('plane')).AddRetVal(
    NumericRetVal('u64')).AddEx(ex)
input.AddMethod('val').AddParameter(NumericParameter('plane')).AddRetVal(
    NumericRetVal('f32')).AddEx(ex)
input.AddMethod('clone').AddRetVal(ObjOwnerRetval(input))

# Output class
output = mod.AddClass(
    Class('Output',
          cpp_name='lczero::python::Output',
          disable_constructor=True))
output.AddMethod('q').AddRetVal(NumericRetVal('f32'))
output.AddMethod('d').AddRetVal(NumericRetVal('f32'))
output.AddMethod('m').AddRetVal(NumericRetVal('f32'))
output.AddMethod('p_raw').AddParameter(IntegralArgv('samples', 'i')).AddRetVal(
    IntegralTupleRetVal('f32')).AddEx(ex)
output.AddMethod('p_softmax').AddParameter(IntegralArgv(
    'samples', 'i')).AddRetVal(IntegralTupleRetVal('f32')).AddEx(ex)

# Backend capabilities class
backend_caps = mod.AddClass(
    Class('BackendCapabilities',
          cpp_name='lczero::python::BackendCapabilities',
          disable_constructor=True))
backend_caps.AddMethod('input_format').AddRetVal(NumericRetVal('i'))
backend_caps.AddMethod('moves_left_format').AddRetVal(NumericRetVal('i'))

# Backend class
backend = mod.AddClass(Class('Backend', cpp_name='lczero::python::Backend'))
backend.AddStaticMethod('available_backends').AddRetVal(ListOfStringsRetVal())
backend.constructor.AddParameter(
    ClassParameter(weights, 'weights', optional=True),
    StringParameter('backend', optional=True, can_be_none=True),
    StringParameter('options', optional=True, can_be_none=True)).AddEx(ex)
backend.AddMethod('evaluate').AddParameter(ArgvObjects(
    'inputs', input)).AddRetVal(ObjTupleRetVal(output)).AddEx(ex)
backend.AddMethod('capabilities').AddRetVal(ObjCopyRetval(backend_caps))

# PositionHistory class
game_state = mod.AddClass(
    Class('GameState', cpp_name='lczero::python::GameState'))
game_state.constructor.AddParameter(
    StringParameter('fen', optional=True, can_be_none=True),
    ListOfStringsParameter('moves', optional=True),
).AddEx(ex)
game_state.AddMethod('as_input').AddParameter(
    ClassParameter(backend, 'backend',
                   optional=False)).AddRetVal(ObjOwnerRetval(input)).AddEx(ex)
game_state.AddMethod('moves').AddRetVal(ListOfStringsRetVal())
game_state.AddMethod('policy_indices').AddRetVal(IntegralTupleRetVal('i'))
game_state.AddMethod('as_string').AddRetVal(StringRetVal())

with open(sys.argv[1], 'wt') as f:
    writer = Writer(f)
    mod.Generate(writer)
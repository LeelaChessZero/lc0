#!/bin/env python3

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
                               NumericParameter)
from pybind.retval import (StringViewRetVal, StringRetVal, IntRetVal,
                           ListOfStringsRetVal, NumericRetVal)

# Module
mod = Module('backends')
mod.AddInclude('python/weights.h')

# Weights class
weights = Class('Weights', cpp_name='lczero::python::Weights')
weights.constructor.AddParameter(
    StringParameter('filename', optional=True, can_be_none=True))
weights.AddMethod('filename').AddRetVal(StringViewRetVal())
weights.AddMethod('license').AddRetVal(StringViewRetVal())
weights.AddMethod('min_version').AddRetVal(StringRetVal())
weights.AddMethod('input_format').AddRetVal(IntRetVal())
weights.AddMethod('policy_format').AddRetVal(IntRetVal())
weights.AddMethod('value_format').AddRetVal(IntRetVal())
weights.AddMethod('moves_left_format').AddRetVal(IntRetVal())
weights.AddMethod('blocks').AddRetVal(IntRetVal())
weights.AddMethod('filters').AddRetVal(IntRetVal())
mod.AddClass(weights)

# Input class
input = Class('Input', cpp_name='lczero::python::Input')
input.AddMethod('set_mask').AddParameter(NumericParameter('plane'),
                                         NumericParameter('mask', type='u64'))
input.AddMethod('set_val').AddParameter(NumericParameter('plane'),
                                        NumericParameter('value', type='f32'))
input.AddMethod('mask').AddParameter(NumericParameter('plane')).AddRetVal(
    NumericRetVal('u64'))
input.AddMethod('val').AddParameter(NumericParameter('plane')).AddRetVal(
    NumericRetVal('f32'))
mod.AddClass(input)

# Backend class
backend = Class('Backend', cpp_name='lczero::python::Backend')
backend.AddStaticMethod('available_backends').AddRetVal(ListOfStringsRetVal())
backend.constructor.AddParameter(
    StringParameter('backend', optional=True, can_be_none=True),
    ClassParameter(weights, 'weights', optional=True),
    StringParameter('options', optional=True, can_be_none=True),
)
mod.AddClass(backend)

with open(sys.argv[1], 'wt') as f:
    writer = Writer(f)
    mod.Generate(writer)
#!/bin/env python

import pybindgen
from pybindgen import retval, param
from pybindgen import ReturnValue
import sys


def GenEnumValues(*vals):
    return [(x, 'pblczero::NetworkFormat::%s' % x) for x in vals]


class StdStringReturn(ReturnValue):
    CTYPES = ['std::string_view']

    def get_c_error_return(self):
        return "return std::string_view();"

    def convert_python_to_c(self, wrapper):
        ptr = wrapper.declarations.declare_variable("const char *",
                                                    "retval_ptr")
        len_ = wrapper.declarations.declare_variable("Py_ssize_t",
                                                     "retval_len")
        wrapper.parse_params.add_parameter("s#", ['&' + ptr, '&' + len_])
        wrapper.after_call.write_code("%s = std::string(%s, %s);" %
                                      (self.value, ptr, len_))

    def convert_c_to_python(self, wrapper):
        wrapper.build_params.add_parameter(
            "s#", ['(%s).data()' % self.value,
                   '(%s).size()' % self.value],
            prepend=True)


mod = pybindgen.Module('backends', cpp_namespace='lczero::python')

exception = mod.add_exception(
    'Exception',
    custom_name='LCZeroException',
    foreign_cpp_namespace='lczero',
    message_rvalue='%(EXC)s.what()',
)

# Weights
mod.add_include('"python/weights.h"')
weights = mod.add_class('Weights')
mod.add_enum('InputFormat',
             GenEnumValues(
                 'INPUT_UNKNOWN',
                 'INPUT_CLASSICAL_112_PLANE',
                 'INPUT_112_WITH_CASTLING_PLANE',
                 'INPUT_112_WITH_CANONICALIZATION',
             ),
             outer_class=weights)
mod.add_enum('PolicyFormat',
             GenEnumValues(
                 'POLICY_UNKNOWN',
                 'POLICY_CLASSICAL',
                 'POLICY_CONVOLUTION',
             ),
             outer_class=weights)
mod.add_enum('ValueFormat',
             GenEnumValues(
                 'VALUE_UNKNOWN',
                 'VALUE_CLASSICAL',
                 'VALUE_WDL',
             ),
             outer_class=weights)
mod.add_enum('MovesLeftFormat',
             GenEnumValues(
                 'MOVES_LEFT_NONE',
                 'MOVES_LEFT_V1',
             ),
             outer_class=weights)

weights.add_constructor(
    [param(
        'const std::string&',
        'filename',
        default_value='""',
    )],
    throw=[exception])
weights.add_method('filename',
                   retval('std::string & const'), [],
                   is_const=True)
weights.add_method('license', retval('std::string_view'), [], is_const=True)
weights.add_method('min_version', retval('std::string'), [], is_const=True)
weights.add_method('input_format', retval('InputFormat'), [], is_const=True)
weights.add_method('policy_format', retval('PolicyFormat'), [], is_const=True)
weights.add_method('value_format', retval('ValueFormat'), [], is_const=True)
weights.add_method('moves_left_format',
                   retval('MovesLeftFormat'), [],
                   is_const=True)
weights.add_method('blocks', retval('int'), [], is_const=True)
weights.add_method('filters', retval('int'), [], is_const=True)

# Output!
with open(sys.argv[1], 'wt') as f:
    mod.generate(f)

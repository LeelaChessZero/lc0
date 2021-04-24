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

from enum import Enum
from .core import PyObject
from .parameters import Parameter
from .retval import RetVal, NoneRetVal
from .exceptions import CppException


class FunctionType(Enum):
    METH_VARARGS = 1
    METH_KEYWORDS = 2
    METH_FASTCALL = 3
    METH_FASTCALL_KEYWORDS = 4
    METH_NOARGS = 5
    METH_O = 6


class Function(PyObject):
    def __init__(self,
                 name,
                 gen_function_name,
                 self_type=None,
                 param_type=None):
        super().__init__(name)
        self.parameters = []
        self.exceptions = []
        self.gen_function_name = gen_function_name
        self.self_type = self_type
        self.param_typ = param_type
        self.retval = NoneRetVal()

    def AddParameter(self, *params):
        for param in params:
            assert isinstance(param, Parameter)
            self.parameters.append(param)
        return self

    def AddRetVal(self, val):
        assert isinstance(self.retval, NoneRetVal)
        assert isinstance(val, RetVal)
        self.retval = val
        return self

    def AddEx(self, ex):
        assert isinstance(ex, CppException)
        self.exceptions.append(ex)
        return self

    def Generate(self, w):
        w.Open(f'{self._return_cpp_type()} '
               f'{self.gen_function_name}({self._generate_params()}) {{')

        self._generate_parse_params(w)
        self.retval.GenerateDeclaration(w)

        if self.exceptions:
            w.Open('try {')

        self._generate_call(w)
        self.retval.GenerateConversion(w)

        if self.exceptions:
            self._handle_exceptions(w)
            w.Close('}')

        w.Write(f'return {self._success()};')
        w.Close('}\n')

    def param_type(self):
        return self.param_typ or Function.TypeFromParameters(self.parameters)

    def function_meth_flags(self):
        return {
            FunctionType.METH_KEYWORDS: 'METH_VARARGS | METH_KEYWORDS',
            FunctionType.METH_NOARGS: 'METH_NOARGS',
            FunctionType.METH_FASTCALL: 'METH_FASTCALL',
        }[self.param_type()]

    def _return_cpp_type(self):
        return 'PyObject*'

    def _generate_parse_params(self, w):
        if self.param_type() == FunctionType.METH_KEYWORDS:
            for param in self.parameters:
                param.GenerateParseTupleSinkDeclaration(w)
            w.Write('const char* keywords[] = {%s};' %
                    ', '.join([f'"{x.name}"'
                               for x in self.parameters] + ['nullptr']))

            def GatherFormatString():
                res = ''
                is_optional = False
                is_keyword_only = False
                for param in self.parameters:
                    if param.optional and not is_optional:
                        is_optional = True
                        res = res + '|'
                    if param.only_keyword and not param.optional:
                        raise ValueError('Non-optional only_keyword.')
                    if param.only_keyword and not is_keyword_only:
                        is_keyword_only = True
                        res = res + '$'
                    if not param.optional and is_optional:
                        raise ValueError('Non-optional after optional.')
                    res += param.parse_tuple_format()
                return res

            params_list = ',\n    '.join([
                'args',
                'kwargs',
                f'"{GatherFormatString()}"',
                'const_cast<char**>(keywords)',
            ] + [
                x for xs in self.parameters
                for x in xs.parse_tuple_sink_list()
            ])
            w.Open(f'if (!PyArg_ParseTupleAndKeywords({params_list})) {{')
            w.Write(f'return {self._failure()};')
            w.Close('}')
            for param in self.parameters:
                param.GenerateCppParamInitialization(w, self)
            return

        if self.param_type() == FunctionType.METH_NOARGS:
            return

        if self.param_type() == FunctionType.METH_FASTCALL:
            assert len(self.parameters) == 1
            param = self.parameters[0]
            assert param.needs_entire_argv()
            param.GenerateCppParamInitialization(w, self)
            return

        raise NotImplementedError('Not implemented function type %d' %
                                  self.param_type())

    def _generate_params(self):
        params = []
        if self.self_type:
            params.append(f'{self.self_type}* self')
        else:
            params.append('void* /* not used */')
        param_type = self.param_type()
        if param_type == FunctionType.METH_VARARGS:
            params.append('PyObject *args')
        elif param_type == FunctionType.METH_KEYWORDS:
            params.append('PyObject *args')
            params.append('PyObject *kwargs')
        elif param_type == FunctionType.METH_FASTCALL:
            params.append('PyObject **args')
            params.append('int num_args')
        elif param_type == FunctionType.METH_FASTCALL_KEYWORDS:
            params.append('PyObject **args')
            params.append('int num_args')
            params.append('PyObject *kwargs')
        elif param_type == FunctionType.METH_NOARGS:
            params.append('PyObject* /* not used */')
        elif param_type == FunctionType.METH_O:
            params.append('PyObject **arg')
        else:
            raise NotImplementedError('Unknown parameter type')
        return ', '.join(params)

    def _list_caller_params(self):
        return ', '.join([x.name_at_caller() for x in self.parameters])

    def _success(self):
        return self.retval.ret_val()

    def _failure(self):
        return 'nullptr'

    def _handle_exceptions(self, w):
        for ex in self.exceptions:
            ex.GenerateHandle(w, self)

    @staticmethod
    def TypeFromParameters(parameters):
        if len(parameters) == 0:
            return FunctionType.METH_NOARGS
        if len(parameters) == 1 and parameters[0].needs_entire_argv():
            return FunctionType.METH_FASTCALL
        return FunctionType.METH_KEYWORDS


class MemberFunction(Function):
    def __init__(self, name, *args, cpp_name=None, **kwargs):
        self.cpp_name = cpp_name or name
        super().__init__(name, *args, **kwargs)

    def _generate_call(self, w):
        if isinstance(self.retval, NoneRetVal):
            w.Write(
                f'self->value->{self.cpp_name}({self._list_caller_params()});')
        else:
            w.Write(f'{self.retval.cpp_type()} '
                    f'{self.retval.cpp_val()} = self->value->'
                    f'{self.cpp_name}({self._list_caller_params()});')


class StaticFunction(Function):
    def __init__(self, name, cpp_type_name, *args, cpp_name=None, **kwargs):
        self.cpp_name = cpp_name or name
        self.cpp_type_name = cpp_type_name
        super().__init__(name, *args, **kwargs)

    def function_meth_flags(self):
        return super().function_meth_flags() + '| METH_STATIC'

    def _generate_call(self, w):
        if isinstance(self.retval, NoneRetVal):
            w.Write(f'{self.cpp_type_name}::{self.cpp_name}'
                    f'({self._list_caller_params()});')
        else:
            w.Write(f'{self.retval.cpp_type()} '
                    f'{self.retval.cpp_val()} = '
                    f'{self.cpp_type_name}::'
                    f'{self.cpp_name}({self._list_caller_params()});')


class Constructor(Function):
    def __init__(self, cpp_type_name, gen_function_name, *args, **kwargs):
        self.cpp_type_name = cpp_type_name
        super().__init__('__init__',
                         gen_function_name=gen_function_name,
                         param_type=FunctionType.METH_KEYWORDS,
                         *args,
                         **kwargs)

    def _generate_call(self, w):
        w.Write(f'self->value = new '
                f'{self.cpp_type_name}({self._list_caller_params()});')

    def _return_cpp_type(self):
        return 'int'

    def _success(self):
        return '0'

    def _failure(self):
        return '-1'


class DisabledConstructor(Function):
    def __init__(self, gen_function_name):
        super().__init__('__init__',
                         gen_function_name=gen_function_name,
                         param_type=FunctionType.METH_KEYWORDS)

    def Generate(self, w):
        w.Open(f'{self._return_cpp_type()} '
               f'{self.gen_function_name}({self._generate_params()}) {{')
        w.Write('PyErr_SetString(PyExc_TypeError, '
                '"Not possible to create instances of this type.");')
        w.Write(f'return {self._failure()};')
        w.Close('}\n')

    def _return_cpp_type(self):
        return 'int'

    def _success(self):
        return '0'

    def _failure(self):
        return '-1'
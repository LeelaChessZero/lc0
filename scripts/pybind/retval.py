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


class RetVal:
    def cpp_val(self):
        return 'retval_cpp'

    def py_val(self):
        return 'retval'

    def ret_val(self):
        return self.py_val()

    def GenerateDeclaration(self, w):
        w.Write(f'PyObject *{self.py_val()};')

    def GenerateConversion(self, w):
        raise NotImplementedError()


class NoneRetVal(RetVal):
    def GenerateDeclaration(self, w):
        pass

    def GenerateConversion(self, w):
        pass

    def py_val(self):
        return 'Py_BuildValue("")'


class GenericStringRetVal(RetVal):
    def GenerateConversion(self, w):
        w.Write(f'{self.py_val()} = Py_BuildValue("s#", '
                f'{self.cpp_val()}.data(), {self.cpp_val()}.size());')


class StringViewRetVal(GenericStringRetVal):
    def cpp_type(self):
        return 'std::string_view'

    def GenerateConversion(self, w):
        w.Write(f'{self.py_val()} = Py_BuildValue("s#", '
                f'{self.cpp_val()}.data(), {self.cpp_val()}.size());')


class StringRetVal(GenericStringRetVal):
    def cpp_type(self):
        return 'const std::string&'

    def GenerateConversion(self, w):
        w.Write(f'{self.py_val()} = Py_BuildValue("s#", '
                f'{self.cpp_val()}.data(), {self.cpp_val()}.size());')


class ListOfStringsRetVal(RetVal):
    def cpp_type(self):
        return 'std::vector<std::string>'

    def GenerateConversion(self, w):
        w.Write(f'{self.py_val()} = PyList_New({self.cpp_val()}.size());')
        w.Open(f'for (size_t i = 0; i < {self.cpp_val()}.size(); ++i) {{')
        w.Write(f'const std::string& s = {self.cpp_val()}[i];')
        w.Write(f'PyList_SetItem({self.py_val()}, '
                'i, Py_BuildValue("s#", s.data(), s.size()));')
        w.Close('}')


class NumericRetVal(RetVal):
    def __init__(self, type):
        self.type = type

    def cpp_type(self):
        return {
            'i': 'int',
            'u64': 'uint64_t',
            'f32': 'float',
        }[self.type]

    def parse_tuple_format(self):
        return {
            'i': 'i',
            'u64': 'K',
            'f32': 'f',
        }[self.type]

    def GenerateConversion(self, w):
        w.Write(f'{self.py_val()} = Py_BuildValue('
                f'"{self.parse_tuple_format()}", {self.cpp_val()});')


class ObjCopyRetval(RetVal):
    def __init__(self, type):
        self.type = type

    def cpp_type(self):
        return f'const {self.type.cpp_name}&'

    def ret_val(self):
        return f'&{self.py_val()}->ob_base'

    def GenerateDeclaration(self, w):
        w.Write(f'{self.type.object_struct_name()} *{self.py_val()};')

    def GenerateConversion(self, w):
        w.Write(f'{self.py_val()} = PyObject_New('
                f'{self.type.object_struct_name()}, '
                f'&{self.type.type_object_name()});')
        w.Write(f'{self.py_val()}->value = '
                f'new {self.type.cpp_name}({self.cpp_val()});')


class ObjOwnerRetval(RetVal):
    def __init__(self, type):
        self.type = type

    def cpp_type(self):
        return f'std::unique_ptr<{self.type.cpp_name}>'

    def ret_val(self):
        return f'&{self.py_val()}->ob_base'

    def GenerateDeclaration(self, w):
        w.Write(f'{self.type.object_struct_name()} *{self.py_val()};')

    def GenerateConversion(self, w):
        w.Write(f'{self.py_val()} = PyObject_New('
                f'{self.type.object_struct_name()}, '
                f'&{self.type.type_object_name()});')
        w.Write(f'{self.py_val()}->value = ' f'{self.cpp_val()}.release();')


class ObjTupleRetVal(RetVal):
    def __init__(self, type):
        self.type = type

    def cpp_type(self):
        return f'std::vector<std::unique_ptr<{self.type.cpp_name}>>'

    def GenerateConversion(self, w):
        w.Write(f'{self.py_val()} = PyTuple_New({self.cpp_val()}.size());')
        w.Open(f'for (size_t i = 0; i < {self.cpp_val()}.size(); ++i) {{')
        w.Write(f'{self.type.object_struct_name()}* tmp = PyObject_New('
                f'{self.type.object_struct_name()}, '
                f'&{self.type.type_object_name()});')
        w.Write(f'tmp->value = {self.cpp_val()}[i].release();')
        w.Write(f'PyTuple_SetItem({self.py_val()}, i, &tmp->ob_base);')
        w.Close('}')


class IntegralTupleRetVal(RetVal):
    def __init__(self, type):
        self.type = type

    def cpp_item_type(self):
        return {
            'i': 'int',
            'u64': 'uint64_t',
            'f32': 'float',
        }[self.type]

    def parse_tuple_format(self):
        return {
            'i': 'i',
            'u64': 'K',
            'f32': 'f',
        }[self.type]

    def cpp_type(self):
        return f'std::vector<{self.cpp_item_type()}>'

    def GenerateConversion(self, w):
        w.Write(f'{self.py_val()} = PyTuple_New({self.cpp_val()}.size());')
        w.Open(f'for (size_t i = 0; i < {self.cpp_val()}.size(); ++i) {{')
        w.Write(f'PyTuple_SetItem({self.py_val()}, i, Py_BuildValue('
                f'"{self.parse_tuple_format()}", {self.cpp_val()}[i]));')
        w.Close('}')

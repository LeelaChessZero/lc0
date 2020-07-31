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

from .core import PyObject


class CppException(PyObject):
    def __init__(self, name, cpp_name, *args, **kwargs):
        self.cpp_name = cpp_name
        super().__init__(name, *args, **kwargs)

    def type_struct_name(self):
        return f'T{self.name}ExceptionType'

    def Generate(self, w):
        w.Write(f'PyObject *{self.type_struct_name()};')

    def GenerateRegister(self, w, module):
        w.Write(f'{self.type_struct_name()} = PyErr_NewException('
                f'"{module.name}.{self.name}", nullptr, nullptr);')
        w.Write(f'if ({self.type_struct_name()} == nullptr) return nullptr;')
        w.Write(f'Py_INCREF({self.type_struct_name()});')
        w.Write(f'PyModule_AddObject(module, "{self.name}", '
                f'{self.type_struct_name()});')

    def GenerateHandle(self, w, func):
        w.Unindent()
        w.Open(f'}} catch (const {self.cpp_name} &ex) {{')
        w.Write(f'PyErr_SetString({self.type_struct_name()}, ex.what());')
        w.Write(f'return {func._failure()};')

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
from .functions import (Constructor, MemberFunction, StaticFunction,
                        DisabledConstructor)
from .exceptions import (CppException)


class FunctionContainer(PyObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.functions = []

    def _generate_functions(self, w):
        for func in self.functions:
            func.Generate(w)
        self._generate_function_list(w)

    def _generate_function_list(self, w):
        w.Open(f'PyMethodDef {self.function_list_name()}[] = {{')
        for f in self.functions:
            w.Write(f'{{"{f.name}", '
                    f'reinterpret_cast<PyCFunction>(&{f.gen_function_name}), '
                    f'{f.function_meth_flags()}, {f.BuildDocString()}}},')
        w.Write('{nullptr, nullptr, 0, nullptr}')
        w.Close('};\n')


class Class(FunctionContainer):
    def __init__(self,
                 name,
                 *argv,
                 cpp_name=None,
                 disable_constructor=False,
                 **kwargs):
        self.cpp_name = cpp_name or name
        super().__init__(name, *argv, **kwargs)
        if disable_constructor:
            self.constructor = DisabledConstructor(
                gen_function_name=self.constructor_name())
        else:
            self.constructor = Constructor(
                cpp_type_name=self.cpp_name,
                gen_function_name=self.constructor_name(),
                self_type=self.object_struct_name())

    def GenerateForwardDeclarations(self, w):
        w.Write(f'struct {self.object_struct_name()};')
        w.Write(f'extern PyTypeObject {self.type_object_name()};')

    def Generate(self, w, module):
        # Object type.
        w.Open(f'struct {self.object_struct_name()} {{')
        w.Write('PyObject_HEAD')
        w.Write(f'{self.cpp_name} *value;')
        w.Close('};\n')

        # Functions
        self._generate_functions(w)

        # Constructor and destructor.
        self.constructor.Generate(w)
        self._generate_destructor(w)

        # Type object.
        self._generate_class_struct(w, module)

    def GenerateRegister(self, w):
        w.Write(f'if (PyType_Ready(&{self.type_object_name()}) '
                '!= 0) return nullptr;')
        w.Write(f'PyModule_AddObject(module, "{self.name}", '
                f'&{self.type_object_name()}.ob_base.ob_base);')

    def AddMethod(self, name, cpp_name=None, *argv, **kwargs):
        method = MemberFunction(name,
                                cpp_name=cpp_name,
                                self_type=self.object_struct_name(),
                                gen_function_name=f'F{self.name}Method{name}',
                                *argv,
                                **kwargs)
        self.functions.append(method)
        return method

    def AddStaticMethod(self, name, cpp_name=None, *argv, **kwargs):
        method = StaticFunction(
            name,
            cpp_type_name=self.cpp_name,
            cpp_name=cpp_name,
            gen_function_name=f'F{self.name}StaticMethod{name}',
            *argv,
            **kwargs)
        self.functions.append(method)
        return method

    def object_struct_name(self):
        return f'T{self.name}ClassType'

    def type_object_name(self):
        return f'obj{self.name}ClassType'

    def constructor_name(self):
        return f'F{self.name}Constructor'

    def destructor_name(self):
        return f'F{self.name}Destructor'

    def function_list_name(self):
        return f'rg{self.name}ClassFunctions'

    def _generate_class_struct(self, w, module):
        w.Open(f'PyTypeObject {self.type_object_name()} = {{')
        w.Write('.ob_base = PyVarObject_HEAD_INIT(NULL, 0)')
        w.Write(f'.tp_name = "{module.name}.{self.name}",')
        w.Write(f'.tp_basicsize = sizeof({self.object_struct_name()}),')
        w.Write('.tp_dealloc = reinterpret_cast<destructor>'
                f'({self.destructor_name()}),')
        w.Write('.tp_flags = Py_TPFLAGS_DEFAULT,')
        w.Write(f'.tp_doc = {self.BuildDocString()},')
        w.Write(f'.tp_methods = {self.function_list_name()},')
        w.Write('.tp_init = reinterpret_cast<initproc>'
                f'({self.constructor_name()}),')
        w.Write('.tp_alloc = PyType_GenericAlloc,')
        w.Write('.tp_new = PyType_GenericNew,')
        w.Close('};')

    def _generate_destructor(self, w):
        w.Open(f'void {self.destructor_name()}('
               f'{self.object_struct_name()}* self) {{')
        w.Write('delete self->value;')
        w.Write('Py_TYPE(self)->tp_free(&self->ob_base);')
        w.Close('}\n')


class Module(FunctionContainer):
    def __init__(self, *argv, **kwargs):
        super().__init__(*argv, **kwargs)
        self.includes = []
        self.classes = []
        self.exceptions = []
        self.initialization = []

    def AddInclude(self, s):
        assert isinstance(s, str)
        self.includes.append(s)

    def AddClass(self, cls):
        assert isinstance(cls, Class)
        self.classes.append(cls)
        return cls

    def AddException(self, ex):
        assert isinstance(ex, CppException)
        self.exceptions.append(ex)
        return ex

    def AddInitialization(self, s):
        assert isinstance(s, str)
        self.initialization.append(s)

    def Generate(self, w):
        w.Write('// This file is AUTOGENERATED, do not edit.')
        w.Write('#define PY_SSIZE_T_CLEAN')
        w.Write('#include <Python.h>')
        for x in self.includes:
            w.Write(f'#include "{x}"')

        w.Write('\nnamespace {')
        for cls in self.exceptions:
            cls.Generate(w)
        for cls in self.classes:
            cls.GenerateForwardDeclarations(w)
        for cls in self.classes:
            cls.Generate(w, self)
        self._generate_functions(w)
        self._generate_module_stuct(w)
        w.Write('}  // anonymous namespace\n')

        self._generate_main_func(w)

    def struct_name(self):
        return f'T{self.name}Module'

    def function_list_name(self):
        return f'rg{self.name}ModuleFunctions'

    def _generate_module_stuct(self, w):
        w.Open(f'PyModuleDef {self.struct_name()} = {{')
        w.Write('PyModuleDef_HEAD_INIT,')
        w.Write(f'"{self.name}",')
        w.Write(f'{self.BuildDocString()},')
        w.Write('-1,')
        w.Write(f'{self.function_list_name()},')
        w.Write('nullptr, ' * 4)
        w.Close('};')

    def _generate_main_func(self, w):
        w.Open(f'PyMODINIT_FUNC PyInit_{self.name}() {{')
        for x in self.initialization:
            w.Write('%s\n' % x)
        w.Write(f'PyObject* module = PyModule_Create(&{self.struct_name()});')
        w.Write('if (module == nullptr) return nullptr;')
        for x in self.exceptions:
            x.GenerateRegister(w, self)
        for x in self.classes:
            x.GenerateRegister(w)
        w.Write('return module;')
        w.Close('}')

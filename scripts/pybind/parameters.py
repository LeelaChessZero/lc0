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


class Parameter:
    def __init__(self,
                 name,
                 optional=False,
                 cpp_name=None,
                 only_keyword=False):
        self.name = name
        self.cpp_name = cpp_name or name
        self.optional = optional
        self.only_keyword = only_keyword

    def name_at_caller(self):
        return self.cpp_name


class StringParameter(Parameter):
    def __init__(self, *argv, can_be_none=False, **kwargs):
        super().__init__(*argv, **kwargs)
        self.can_be_none = can_be_none

    def GenerateParseTupleSinkDeclaration(self, w):
        w.Write(f'const char* {self.name} = nullptr;')
        w.Write(f'Py_ssize_t {self.name}_len = 0;')

    def GenerateCppParamInitialization(self, w):
        if self.optional or self.can_be_none:
            w.Write(f'std::optional<std::string> {self.name_at_caller()};')
            w.Write(f'if ({self.name} != nullptr) '
                    f'{self.name_at_caller()}.emplace('
                    f'{self.name}, {self.name}_len);')
        else:
            raise NotImplementedError()

    def parse_tuple_sink_list(self):
        return [f'&{self.name}', f'&{self.name}_len']

    def parse_tuple_format(self):
        return 'z#' if self.can_be_none else 's#'

    def name_at_caller(self):
        return f'{self.cpp_name}_cpp'


class ClassParameter(Parameter):
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        super().__init__(*args, **kwargs)

    def GenerateParseTupleSinkDeclaration(self, w):
        w.Write(f'{self.cls.object_struct_name()}* {self.name} = nullptr;')

    def parse_tuple_sink_list(self):
        return [f'&{self.cls.type_object_name()}', f'&{self.name}']

    def parse_tuple_format(self):
        return 'O!'

    def GenerateCppParamInitialization(self, w):
        pass

    def name_at_caller(self):
        if self.optional:
            return f'{self.cpp_name} ? {self.cpp_name}->value : nullptr'
        else:
            return f'*{self.cpp_name}->value'


class NumericParameter(Parameter):
    def __init__(self, *args, type='i', **kwargs):
        self.type = type
        super().__init__(*args, **kwargs)

    def GenerateParseTupleSinkDeclaration(self, w):
        w.Write(f'{self.cpp_type()} {self.name} = 0;')

    def cpp_type(self):
        return {
            'i': 'int',
            'u64': 'uint64_t',
            'f32': 'float',
        }[self.type]

    def parse_tuple_sink_list(self):
        return [f'&{self.name}']

    def parse_tuple_format(self):
        return {
            'i': 'i',
            'u64': 'k',
            'f32': 'f',
        }[self.type]

    def GenerateCppParamInitialization(self, w):
        pass

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


class Writer:
    '''A helper class for writing file line by line with indent.'''
    def __init__(self, fo):
        self.fo = fo
        self.indent = 0

    def Indent(self):
        self.indent += 2

    def Unindent(self):
        self.indent -= 2

    def Write(self, text):
        for line in text.split('\n'):
            if line:
                self.fo.write(' ' * self.indent + line + '\n')
            else:
                self.fo.write('\n')

    def Open(self, text='{'):
        self.Write(text)
        self.Indent()

    def Close(self, text='}'):
        self.Unindent()
        self.Write(text)

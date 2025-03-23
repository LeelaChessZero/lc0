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

import argparse
import os
import re
import sys

VARINT_TYPES = {
    'int32': 'std::int32_t',
    'int64': 'std::int64_t',
    'uint32': 'std::uint32_t',
    'uint64': 'std::uint64_t',
    'sint32': 'std::uint32_t',
    'sint64': 'std::uint64_t',
    'bool': 'bool',
}

FIXED64_TYPES = {
    'fixed64': 'std::uint64_t',
    'sfixed64': 'std::int64_t',
    'double': 'double',
}
FIXED32_TYPES = {
    'fixed32': 'std::uint32_t',
    'sfixed32': 'std::int32_t',
    'float': 'float',
}
BYTES_TYPES = {
    'string': 'std::string_view',
    'bytes': 'std::string_view',
}
ZIGZAG_TYPES = set(['sint32', 'sint64'])
FLOAT_TYPES = set(['float', 'double'])

TYPES = {**VARINT_TYPES, **FIXED32_TYPES, **FIXED64_TYPES, **BYTES_TYPES}

RESERVED_WORDS = [
    'enum',
    'message',
    'optional',
    'package',
    'repeated',
    'required',
    'reserved',
    'syntax',
    'to',
] + list(TYPES.keys())

GRAMMAR = ([(r'%s\b' % x, x)
            for x in RESERVED_WORDS] + [('\\' + x, x) for x in '=;{}.,'] + [
                (r'/\*.*?\*/', None),  # /* Comment */
                (r'//.*?$', None),  # // Comment
                (r'\s+', None),  # Whitespace
                (r'$', 'EOF'),
                (r'"((?:[^"\\]|\\.)*)"', 'string'),
                (r'\d+', 'number'),
                (r'(\w+)', 'identifier'),
            ])


class Lexer:

    def __init__(self, text):
        self.text = text
        self.grammar = [(re.compile(x, re.S + re.M), y) for x, y in GRAMMAR]
        self.cur_token = None
        self.cur_offset = 0

    def Pick(self):
        '''Picks the last token in queue. Doesn't advance the queue.'''
        if self.cur_token is None:
            self.cur_token = self.NextToken()
        return self.cur_token

    def Consume(self, expected_token, value=None, group=0):
        '''Gets the token from the queue and advances the queue.

        If @expected_token if of wrong type, or @value is not equal to regexes
        @group, throws an error.
        '''
        token, match = self.Pick()
        if expected_token != token:
            self.Error(f'Expected token type [{expected_token}], got [{token}]')
        if value is not None and value != match.group(group):
            self.Error('Expected value [%s]' % value)
        self.cur_offset = match.span()[1]
        self.cur_token = None
        return match

    def NextToken(self):
        '''Reads the stream and returns the next token.

        (which is not whitespace or comment)
        '''
        while True:
            token, match = self.NextTokenOrWhitespace()
            if token is None:
                self.cur_offset = match.span()[1]
            else:
                return token, match

    def NextTokenOrWhitespace(self):
        '''Reads the stream and returns the next token (possibly whitespace).'''
        for r, token in self.grammar:
            m = r.match(self.text, self.cur_offset)
            if m:
                return (token, m)
        token_snippet = self.text[self.cur_offset:self.cur_offset + 10]
        self.Error(f'Unparseable token [{token_snippet}...]')

    def Error(self, text):
        '''Throws an error with context in the file read.'''
        line = self.text[:self.cur_offset].count('\n') + 1
        line_start = self.text.rfind('\n', 0, self.cur_offset) + 1
        line_end = self.text.find('\n', line_start)
        if line_end == -1:
            line_end = len(self.text)
        sys.stderr.write('%s:\n' % text)
        sys.stderr.write(self.text[line_start:line_end] + '\n')
        sys.stderr.write(' ' * (self.cur_offset - line_start) + '^^^\n')
        raise ValueError("Parse error: %s at line %d column %d." %
                         (text, line, (self.cur_offset - line_start)))


def ReadIdentifierPath(lexer):
    '''Reads qualified identifier a.b.d into ['a', 'b', 'd'] list'''
    path = []
    while True:
        path.append(lexer.Consume('identifier').group(0))
        if lexer.Pick()[0] != '.':
            return path
        lexer.Consume('.')


def LookupType(name, stack):
    '''Looks up the (possibly qualified) from the innermost scope first.'''
    for y in stack:
        for x in y:
            if x.GetName() == name[0]:
                if len(name) == 1:
                    return x
                else:
                    return LookupType(name[1:], [x.GetTypes()])
    raise ValueError("Cannot find type: %s." % '.'.join(name))


# All *Parser classes have the following semantics:
# * They are called with lexer as input to parse grammar from .proto file.
# * The Generate() function writes relevant portion of .pb.h file.


class ProtoTypeParser:

    def __init__(self, lexer, object_stack):
        token, match = lexer.Pick()
        if token in TYPES:
            self.typetype = 'basic'
            self.name = token
            lexer.Consume(token)
        elif token == 'identifier':
            self.name = ReadIdentifierPath(lexer)
            self.typetype = 'forward'
        else:
            lexer.Error('Type expected')

    def LookupForwardFieldType(self, object_stack):
        if self.IsForward():
            typ = LookupType(self.name, object_stack)
            self.typetype = typ.GetType()
            self.name = [typ.GetFullName()]

    def IsZigzag(self):
        if self.typetype == 'basic':
            return self.name in ZIGZAG_TYPES
        return False

    def GetCppType(self):
        if self.typetype == 'basic':
            return TYPES[self.name]
        else:
            return '_'.join(self.name)

    def GetVariableCppType(self):
        if self.IsBytesType():
            return 'std::string'
        else:
            return self.GetCppType()

    def IsEnumType(self):
        return self.typetype == 'enum'

    def IsVarintType(self):
        return self.typetype == 'enum' or (self.typetype == 'basic'
                                           and self.name in VARINT_TYPES)

    def IsFixedType(self):
        return self.typetype == 'basic' and (self.name in FIXED64_TYPES
                                             or self.name in FIXED32_TYPES)

    def IsBytesType(self):
        return self.typetype == 'basic' and self.name in BYTES_TYPES

    def IsFloatType(self):
        return self.typetype == 'basic' and self.name in FLOAT_TYPES

    def GetWireType(self):
        if self.typetype == 'basic':
            if self.name in VARINT_TYPES:
                return 0
            if self.name in FIXED64_TYPES:
                return 1
            if self.name in BYTES_TYPES:
                return 2
            if self.name in FIXED32_TYPES:
                return 5
            raise ValueError('Unknown type %s' % self.name)
        elif self.typetype == 'enum':
            return 0
        elif self.typetype == 'message':
            return 2
        else:
            raise ValueError('Unknown typetype %s' % self.typetype)

    def IsMessage(self):
        return self.typetype == 'message'

    def IsForward(self):
        return self.typetype == 'forward'

    def IsIntegralType(self):
        if self.typetype == 'basic':
            if self.name == 'double':
                return False
            if self.name == 'float':
                return False
            if self.name in BYTES_TYPES:
                return False
            if self.name in TYPES:
                return True
            raise ValueError('Unknown type %s' % self.name)
        elif self.typetype == 'enum':
            return True
        elif self.typetype == 'message':
            return False
        else:
            raise ValueError('Unknown typetype %s' % self.typetype)


class ProtoFieldParser:

    def __init__(self, lexer, object_stack):
        token, match = lexer.Pick()
        if token not in ['repeated', 'optional', 'required']:
            lexer.Error('repeated, optional or required expected')
        self.category = token
        lexer.Consume(token)
        self.type = ProtoTypeParser(lexer, object_stack)
        self.name = lexer.Consume('identifier')
        lexer.Consume('=')
        self.number = int(lexer.Consume('number').group(0))
        lexer.Consume(';')

    def IsType(self):
        return False

    def LookupForwardFieldType(self, object_stack):
        self.type.LookupForwardFieldType(object_stack)

    def GetParser(self):
        name = self.name.group(0)
        if self.type.IsMessage():
            if self.category == 'repeated':
                return 'add_%s()->MergeFromString(val)' % name
            else:
                return 'mutable_%s()->MergeFromString(val)' % name

        cpp_type = self.type.GetCppType()
        val = 'NOT IMPLEMENTED!'
        if self.type.IsVarintType():
            val_val = 'UnZigZag(val)' if self.type.IsZigzag() else 'val'
            val = 'static_cast<%s>(%s)' % (cpp_type, val_val)
        elif self.type.IsFixedType():
            if self.type.IsFloatType():
                val = 'bit_cast<%s>(val)' % cpp_type
            else:
                val = 'static_cast<%s>(val)' % cpp_type
        elif self.type.IsBytesType():
            val = 'val'

        if self.category == 'repeated':
            return '%s_.emplace_back(%s)' % (name, val)
        else:
            return 'set_%s(%s)' % (name, val)

    def GenerateCaseClause(self, w):
        w.Write('case %d: %s; break;' % (self.number, self.GetParser()))

    def GenerateClear(self, w):
        name   = self.name.group(0)
        if self.category == 'repeated':
            w.Write('%s_.clear();' % name)
        else:
            w.Write('has_%s_ = false;' % name)
            w.Write('%s_ = {};' % name)

    def GenerateOutput(self, w):
        fname = {
            0: 'AppendVarInt',
            1: 'AppendInt64',
            2: 'AppendString',
            5: 'AppendInt32'
        }
        tname = {
            0: 'std::uint64_t',
            1: 'std::uint64_t',
            2: 'std::string_view',
            5: 'std::uint32_t'
        }
        wire_id = self.type.GetWireType()
        if self.category == 'repeated':
            prefix = 'for (const auto& x : %s)' % (self.name.group(0) + '_')
            name = 'x'
        else:
            name = self.name.group(0) + '_'
            prefix = 'if (has_%s)' % (name)
        if self.type.IsMessage():
            name += '.OutputAsString()'
        elif self.type.IsFloatType():
            name = 'bit_cast<%s>(%s)' % (tname[wire_id], name)

        w.Write('%s %s(%d, %s, &out);' %
                (prefix, fname[wire_id], self.number, name))

    def GenerateJsonOutput(self, w):
        name = self.name.group(0)
        if self.category == 'repeated':
            prefix = 'if (!%s_.empty())' % name
            funcname = 'AppendJsonRepeatedField'
        else:
            prefix = 'if (has_%s_)' % name
            funcname = 'AppendJsonField'
        if self.type.IsEnumType():
            value = '%s_Name(%s_)' % (self.type.GetCppType(), name)
        else:
            value = name + "_"
        w.Write('%s %s("%s", %s, &first, &out);' %
                (prefix, funcname, name, value))

    def GenerateFunctionDeclarations(self, w):
        name = self.name.group(0)
        cpp_type = self.type.GetCppType()
        var_cpp_type = self.type.GetVariableCppType()
        if self.category == 'repeated':
            if self.type.IsMessage():
                w.Write("%s* add_%s();" % (cpp_type, name))
            else:
                w.Write("void add_%s(%s val);" % (name, cpp_type))
            # Using a vector here breaks API compatibility with the standard
            # protobuf library, but it is more convenient.
            w.Write("const std::vector<%s>& %s() const;" %
                    (var_cpp_type, name))
            w.Write("std::vector<%s>* mutable_%s();" % (var_cpp_type, name))
            if self.type.IsMessage():
                w.Write("const %s& %s(size_t idx) const;" % (cpp_type, name))
                w.Write("%s* mutable_%s(size_t idx);" % (cpp_type, name))
            else:
                w.Write("%s %s(size_t idx) const;" % (cpp_type, name))
            w.Write("size_t %s_size() const;" % (name))
        else:
            w.Write("bool has_%s() const;" % (name))
            if self.type.IsMessage():
                w.Write("const %s& %s() const;" % (cpp_type, name))
            if self.type.IsMessage() or self.type.IsBytesType():
                w.Write("%s* mutable_%s();" % (var_cpp_type, name))
            if not self.type.IsMessage():
                w.Write("%s %s() const;" % (cpp_type, name))
                w.Write("void set_%s(%s val);" % (name, cpp_type))

    def GenerateFunctionDefinitions(self, w, class_name):
        name = self.name.group(0)
        cpp_type = self.type.GetCppType()
        var_cpp_type = self.type.GetVariableCppType()
        if self.category == 'repeated':
            if self.type.IsMessage():
                w.Write(
                    "inline %s* %s::add_%s() { return &%s_.emplace_back(); }" %
                    (cpp_type, class_name, name, name))
            else:
                w.Write(
                    "inline void %s::add_%s(%s val) { %s_.emplace_back(val); }"
                    % (class_name, name, cpp_type, name))
            w.Write(
                "inline const std::vector<%s>& %s::%s() const { return %s_; }"
                % (var_cpp_type, class_name, name, name))
            w.Write(
                "inline std::vector<%s>* %s::mutable_%s() { return &%s_; }"
                % (var_cpp_type, class_name, name, name))
            if self.type.IsMessage():
                w.Write(
                    "inline const %s& %s::%s(size_t idx) const { return %s_[idx]; }"
                    % (cpp_type, class_name, name, name))
                w.Write(
                    "inline %s* %s::mutable_%s(size_t idx) { return &%s_[idx]; }"
                    % (cpp_type, class_name, name, name))
            else:
                w.Write(
                    "inline %s %s::%s(size_t idx) const { return %s_[idx]; }" %
                    (cpp_type, class_name, name, name))
            w.Write(
                "inline size_t %s::%s_size() const { return %s_.size(); }" %
                (class_name, name, name))
        else:
            w.Write("inline bool %s::has_%s() const { return has_%s_; }" %
                    (class_name, name, name))
            if self.type.IsMessage():
                w.Write("inline const %s& %s::%s() const { return %s_; }" %
                        (cpp_type, class_name, name, name))
            if self.type.IsMessage() or self.type.IsBytesType():
                w.Write("inline %s* %s::mutable_%s() {" %
                        (var_cpp_type, class_name, name))
                w.Indent()
                w.Write('has_%s_ = true;' % (name))
                w.Write('return &%s_;' % name)
                w.Unindent()
                w.Write("}")
            if not self.type.IsMessage():
                w.Write("inline %s %s::%s() const { return %s_; }" %
                        (cpp_type, class_name, name, name))
                w.Write("inline void %s::set_%s(%s val) {" %
                        (class_name, name, cpp_type))
                w.Indent()
                w.Write("has_%s_ = true;" % name)
                w.Write("%s_ = val;" % name)
                w.Unindent()
                w.Write("}")

    def GenerateVariable(self, w):
        name = self.name.group(0)
        cpp_type = self.type.GetVariableCppType()
        if self.category == 'repeated':
            w.Write("std::vector<%s> %s_;" % (cpp_type, name))
        else:
            w.Write("bool has_%s_{};" % (name))
            w.Write("%s %s_{};" % (cpp_type, name))
        return


class ProtoEnumParser:

    def __init__(self, lexer, scope):
        lexer.Consume('enum')
        self.name = lexer.Consume('identifier').group(0)
        self.values = []
        self.scope = scope[:]
        lexer.Consume('{')
        while True:
            token, match = lexer.Pick()
            if token == '}':
                break
            key = lexer.Consume('identifier').group(0)
            lexer.Consume('=')
            value = int(lexer.Consume('number').group(0))
            lexer.Consume(';')
            self.values.append((key, value))
        lexer.Consume('}')

    def GetName(self):
        return self.name

    def GetFullName(self):
        return '_'.join([x.GetName() for x in self.scope] + [self.name])

    def GetType(self):
        return 'enum'

    def IsType(self):
        return True

    def ResolveForwardDeclarations(self, _):
        pass

    def GenerateMessageDeclarations(self, w):
        pass

    def GenerateMessageDefinitions(self, w):
        pass

    def GenerateFunctionDefinitions(self, w):
        pass

    def GenerateEnumDefinitions(self, w):
        # Protobuf enum is mapped directly to C++ enum.
        w.Write('enum %s : int {' % self.GetFullName())
        w.Indent()
        for key, value in self.values:
            w.Write('%s_%s = %d,' % (self.GetFullName(), key, value))
        w.Unindent()
        w.Write('};')
        w.Write('inline std::string %s_Name(%s val) {' %
                (self.GetFullName(), self.GetFullName()))
        w.Indent()
        w.Write('switch (val) {')
        w.Indent()
        for key, _ in self.values:
            w.Write('case %s_%s:' % (self.GetFullName(), key))
            w.Write('  return "%s";' % key)
        w.Unindent()
        w.Write('};')
        w.Write('return "%s(" + std::to_string(val) + ")";' % self.name)
        w.Unindent()
        w.Write('}')

    def GenerateUsingDirectives(self, w):
        w.Write('using %s = %s;' % (self.name, self.GetFullName()))
        for key, _ in self.values:
            w.Write('static constexpr %s %s =' % (self.name, key))
            w.Write('    %s_%s;' % (self.GetFullName(), key))
        w.Write('static constexpr std::array<%s,%d> %s_AllValues = {' %
                (self.name, len(self.values), self.name))
        w.Indent()
        for key, _ in self.values:
            w.Write('%s,' % key)
        w.Unindent()
        w.Write('};')
        # Static function to convert an enum value to its name.
        w.Write('static std::string %s_Name(%s val) {' %
                (self.name, self.name))
        w.Indent()
        w.Write('return %s_Name(val);' % (self.GetFullName()))
        w.Unindent()
        w.Write('}')


def ParseReservedFields(lexer):
    res = set()
    lexer.Consume('reserved')
    while True:
        token, match = lexer.Pick()
        if token == 'number':
            num = int(lexer.Consume('number').group(0))
            if lexer.Pick()[0] == 'to':
                lexer.Consume('to')
                end = int(lexer.Consume('number').group(0))
                res.add(range(num, end + 1))
            else:
                res.add(num)
        elif token in ['identifier', 'string']:
            res.add(lexer.Consume(token).group(1))
        else:
            lexer.Error('Expected number or identifier')
        token, _ = lexer.Pick()
        if token == ';':
            lexer.Consume(';')
            break
        lexer.Consume(',')
    return res


class ProtoMessageParser:

    def __init__(self, lexer, type_stack, scope):
        type_stack[0].append(self)
        self.reserved = set()
        self.types = []
        self.fields = []
        self.scope = scope[:]
        lexer.Consume('message')
        self.name = lexer.Consume('identifier').group(0)
        lexer.Consume('{')
        while True:
            token, match = lexer.Pick()
            if token == '}':
                break
            elif token == 'message':
                ProtoMessageParser(lexer, [self.types, *type_stack],
                                   self.scope + [self])
            elif token == 'enum':
                self.types.append(ProtoEnumParser(lexer, self.scope + [self]))
            elif token in ['repeated', 'optional', 'required']:
                self.fields.append(
                    ProtoFieldParser(lexer, [self.types, *type_stack]))
            elif token == 'reserved':
                self.reserved.update(ParseReservedFields(lexer))
            else:
                lexer.Error('Expected field or type')
        lexer.Consume('}')
        self.CheckReserved()

    def GetName(self):
        return self.name

    def GetFullName(self):
        return '_'.join([x.GetName() for x in self.scope] + [self.name])

    def GetType(self):
        return 'message'

    def IsType(self):
        return True

    def GetTypes(self):
        return self.types

    def GetFieldsGruppedByWireType(self):
        type_to_fields = {}
        for x in self.fields:
            type_to_fields.setdefault(x.type.GetWireType(), []).append(x)
        return type_to_fields
    
    def CheckReserved(self):
        for r in self.reserved:
            if isinstance(r, int):
                if any(x.number == r for x in self.fields):
                    raise ValueError(f'Field number [{r}] is reserved.')
            elif isinstance(r, range):
                if any(x.number in r for x in self.fields):
                    raise ValueError(f'Field range [{r.start} to {r.stop-1}] '
                                     'is reserved.')
            else:
                if any(x.name.group(0) == r for x in self.fields):
                    raise ValueError(f'Field name [{r}] is reserved.')

    def ResolveForwardDeclarations(self, type_stack):
        type_stack.append(self.types)
        for x in self.types:
            x.ResolveForwardDeclarations(type_stack)
        for x in self.fields:
            x.LookupForwardFieldType(type_stack)
        type_stack.pop()

    def WriteFieldParserDeclaration(self, w, wire_id, fields):
        fname = {0: 'SetVarInt', 1: 'SetInt64', 2: 'SetString', 5: 'SetInt32'}
        tname = {
            0: 'std::uint64_t',
            1: 'std::uint64_t',
            2: 'std::string_view',
            5: 'std::uint32_t'
        }
        w.Write('void %s(int field_id, %s val) final;' %
                (fname[wire_id], tname[wire_id]))

    def WriteFieldParserDefinition(self, w, wire_id, fields):
        fname = {0: 'SetVarInt', 1: 'SetInt64', 2: 'SetString', 5: 'SetInt32'}
        tname = {
            0: 'std::uint64_t',
            1: 'std::uint64_t',
            2: 'std::string_view',
            5: 'std::uint32_t'
        }
        w.Write('inline void %s::%s(int field_id, %s val) {' %
                (self.GetFullName(), fname[wire_id], tname[wire_id]))
        w.Indent()
        w.Write('switch (field_id) {')
        w.Indent()
        for field in fields:
            field.GenerateCaseClause(w)
        w.Unindent()
        w.Write('}')
        w.Unindent()
        w.Write('}')

    def GenerateUsingDirectives(self, w):
        w.Write('using %s = %s;' % (self.name, self.GetFullName()))

    def GenerateMessageDeclarations(self, w):
        w.Write(f'class %s;' % self.GetFullName())
        for x in self.types:
            x.GenerateMessageDeclarations(w)

    def GenerateEnumDefinitions(self, w):
        for x in self.types:
            x.GenerateEnumDefinitions(w)

    def GenerateMessageDefinitions(self, w):
        # Writing nested messages.
        for x in self.types:
            if x.GetType() == 'message':
                x.GenerateMessageDefinitions(w)
        # Protobuf message is a C++ class.
        w.Write('class %s final : public lczero::ProtoMessage {' %
                self.GetFullName())
        w.Write(' public:')
        w.Indent()
        # Writing using directives.
        for x in self.types:
            x.GenerateUsingDirectives(w)
        # Writing function declarations.
        for x in self.fields:
            w.Write('')
            x.GenerateFunctionDeclarations(w)
        w.Write('')
        w.Write('std::string OutputAsString() const final;')
        w.Write('std::string OutputAsJson() const final;')
        w.Write('void Clear() final;')

        w.Unindent()
        w.Write('')
        w.Write(' private:')
        w.Indent()
        for k, v in self.GetFieldsGruppedByWireType().items():
            self.WriteFieldParserDeclaration(w, k, v)
        w.Write('')
        for x in self.fields:
            x.GenerateVariable(w)
        w.Unindent()
        w.Write('};')
        w.Write('')

    def GenerateFunctionDefinitions(self, w):
        # Writing nested messages.
        for x in self.types:
            if x.GetType() == 'message':
                x.GenerateFunctionDefinitions(w)
        self.GenerateOutputAsStringFunc(w)
        self.GenerateOutputAsJsonFunc(w)
        self.GenerateClearFunc(w)
        self.GenerateParserFuncs(w)
        self.GenerateFieldAccessorFuncs(w)

    def GenerateOutputAsStringFunc(self, w):
        w.Write('inline std::string %s::OutputAsString() const {' %
                self.GetFullName())
        w.Indent()
        w.Write('std::string out;')
        for x in sorted(self.fields, key=lambda x: x.number):
            x.GenerateOutput(w)
        w.Write('return out;')
        w.Unindent()
        w.Write('}')

    def GenerateOutputAsJsonFunc(self, w):
        w.Write('inline std::string %s::OutputAsJson() const {' %
                self.GetFullName())
        w.Indent()
        if self.fields:
            w.Write('bool first = true;')
        w.Write('std::string out = "{";')
        for x in self.fields:
            x.GenerateJsonOutput(w)
        w.Write('out += "}";')
        w.Write('return out;')
        w.Unindent()
        w.Write('}')

    def GenerateClearFunc(self, w):
        w.Write('inline void %s::Clear() {' % self.GetFullName())
        w.Indent()
        for x in self.fields:
            x.GenerateClear(w)
        w.Unindent()
        w.Write('}')

    def GenerateParserFuncs(self, w):
        for k, v in self.GetFieldsGruppedByWireType().items():
            self.WriteFieldParserDefinition(w, k, v)

    def GenerateFieldAccessorFuncs(self, w):
        for x in self.fields:
            x.GenerateFunctionDefinitions(w, self.GetFullName())


class ProtoFileParser:
    '''Root grammar of .proto file'''

    def __init__(self, lexer):
        self.package = None
        self.types = []
        while True:
            token, match = lexer.Pick()
            if token == 'EOF':
                return
            elif token == 'syntax':
                self.ParseSyntax(lexer)
            elif token == 'package':
                self.ParsePackage(lexer)
            elif token == 'message':
                self.ParseMessage(lexer)
            elif token == 'enum':
                self.ParseEnum(lexer)
            else:
                lexer.Error('Expected message or something similar')

    def ParseSyntax(self, lexer):
        lexer.Consume('syntax')
        lexer.Consume('=')
        lexer.Consume('string', 'proto2', 1)
        lexer.Consume(';')

    def ParsePackage(self, lexer):
        lexer.Consume('package')
        if self.package is not None:
            lexer.Error('Package was already defined')
        self.package = ReadIdentifierPath(lexer)
        lexer.Consume(';')

    def ParseMessage(self, lexer):
        ProtoMessageParser(lexer, [self.types], [])

    def ParseEnum(self, lexer):
        self.types.append(ProtoEnumParser(lexer, []))

    def Generate(self, w):
        w.Write('// This file is AUTOGENERATED, do not edit.')
        w.Write('#pragma once')
        w.Write('#include "utils/protomessage.h"')
        for x in self.package:
            w.Write('namespace %s {' % x)
        w.Write('')
        w.Write('// Forward declarations.')
        for object in self.types:
            object.GenerateMessageDeclarations(w)
        for object in self.types:
            object.GenerateEnumDefinitions(w)
        w.Write('')
        w.Write('// Class declarations.')
        for object in self.types:
            object.GenerateMessageDefinitions(w)
        w.Write('')
        w.Write('// Function definitions.')
        for object in self.types:
            object.GenerateFunctionDefinitions(w)
        for x in reversed(self.package):
            w.Write('}  // namespace %s' % x)

    def ResolveForwardDeclarations(self):
        type_stack = [self.types]
        for object in self.types:
            object.ResolveForwardDeclarations(type_stack)


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
        if text:
            self.fo.write(' ' * self.indent + text + '\n')
        else:
            self.fo.write('\n')


if __name__ == "__main__":
    # Have the same flags as protoc has.
    parser = argparse.ArgumentParser(description="Compile protobuf files.")
    parser.add_argument('input', type=str)
    parser.add_argument('--proto_path', type=str)
    parser.add_argument('--cpp_out', type=str)
    args = parser.parse_args()

    rel_path = os.path.relpath(args.input, args.proto_path)
    dest_name = os.path.splitext(rel_path)[0] + '.pb.h'
    dest_path = os.path.join(args.cpp_out, dest_name)
    dest_dir = os.path.dirname(dest_path)
    os.makedirs(dest_dir, exist_ok=True)

    with open(args.input, 'r') as input, open(dest_path, 'w') as output:
        proto_file = ProtoFileParser(Lexer(input.read()))
        proto_file.ResolveForwardDeclarations()
        writer = Writer(output)
        proto_file.Generate(writer)

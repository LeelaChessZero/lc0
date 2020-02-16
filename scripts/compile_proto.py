#!/usr/bin/env python3

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

TYPES = {**VARINT_TYPES, **FIXED32_TYPES, **FIXED64_TYPES, **BYTES_TYPES}

RESERVED_WORDS = [
    'syntax',
    'package',
    'message',
    'optional',
    'repeated',
    'enum',
] + list(TYPES.keys())

GRAMMAR = ([(x, x)
            for x in RESERVED_WORDS] + [('\\' + x, x) for x in '=;{}.'] + [
                (r'/\*.*?\*/', None),  # /* Comment */
                (r'//.*?$', None),  # // Comment
                (r'\s+', None),  # Whitespace
                (r'$', 'EOF'),
                (r'"((?:[^"\\]|\\.)*)"', 'string'),
                (r'\d+', 'number'),
                (r'\w+', 'identifier'),
            ])


class Lexer:
    def __init__(self, text):
        self.text = text
        self.grammar = [(re.compile(x, re.S + re.M), y) for x, y in GRAMMAR]
        self.cur_token = None
        self.cur_offset = 0

    def Pick(self):
        if self.cur_token is None:
            self.cur_token = self.NextToken()
        return self.cur_token

    def Consume(self, expected_token, value=None, group=0):
        token, match = self.Pick()
        if expected_token != token:
            self.Error('Expected token type [%s]' % expected_token)
        if value is not None and value != match.group(group):
            self.Error('Expected value [%s]' % value)
        self.cur_offset = match.span()[1]
        self.cur_token = None
        return match

    def NextToken(self):
        while True:
            token, match = self.NextTokenOrWhitespace()
            if token is None:
                self.cur_offset = match.span()[1]
            else:
                return token, match

    def NextTokenOrWhitespace(self):
        for r, token in self.grammar:
            m = r.match(self.text, self.cur_offset)
            if m:
                return (token, m)
        self.Error('Unexpected token')

    def Error(self, text):
        line_start = self.text.rfind('\n', 0, self.cur_offset) + 1
        line_end = self.text.find('\n', line_start)
        sys.stderr.write('%s:\n' % text)
        sys.stderr.write(self.text[line_start:line_end] + '\n')
        sys.stderr.write(' ' * (self.cur_offset - line_start) + '^^^\n')
        raise ValueError("Parse error: %s at offset %d." %
                         (text, self.cur_offset))


def ReadIdentifierPath(lexer):
    path = []
    while True:
        path.append(lexer.Consume('identifier').group(0))
        if lexer.Pick()[0] != '.':
            return path
        lexer.Consume('.')


def LookupType(name, stack):
    for y in stack:
        for x in y:
            if not x.IsType():
                continue
            if x.GetName() == name[0]:
                if len(name) == 1:
                    return x.GetType()
                else:
                    return LookupType(name[1:], [x.GetObjects()])
    raise ValueError("Cannot find type: %s." % '.'.join(name))


class ProtoTypeParser:
    def __init__(self, lexer, object_stack):
        token, match = lexer.Pick()
        if token in TYPES:
            self.typetype = 'basic'
            self.name = token
            lexer.Consume(token)
        elif token == 'identifier':
            self.name = ReadIdentifierPath(lexer)
            self.typetype = LookupType(self.name, object_stack)
        else:
            lexer.Error('Type expected')

    def IsZigzag(self):
        if self.typetype == 'basic':
            return self.name in ZIGZAG_TYPES
        return False

    def GetCppType(self):
        if self.typetype == 'basic':
            return TYPES[self.name]
        else:
            return '::'.join(self.name)

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

    def DecodeFunction(self, wire_id, index):
        if self.typetype == 'basic':
            if self.name == 'double':
                return 'GetDoubleVal(%d, %s)' % (wire_id, index)
            if self.name == 'float':
                return 'GetFloatVal(%d, %s)' % (wire_id, index)
            if self.name in VARINT_TYPES:
                return 'GetVarintVal(%d, %s)' % (wire_id, index)
            if self.name in FIXED64_TYPES:
                return 'GetFixed64Val(%d, %s)' % (wire_id, index)
            if self.name in BYTES_TYPES:
                return 'GetBytesVal(%d, %s)' % (wire_id, index)
            if self.name in FIXED32_TYPES:
                return 'GetFixed32Val(%d, %s)' % (wire_id, index)
            raise ValueError('Unknown type %s' % self.name)
        elif self.typetype == 'enum':
            return 'GetVarintVal(%d, %s)' % (wire_id, index)
        elif self.typetype == 'message':
            return '%s::CreateNotOwned(GetBytesVal(%d, %s))' % (
                self.GetCppType(), wire_id, index)
        else:
            raise ValueError('Unknown typetype %s' % self.typetype)

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

    def Generate(self, w):
        name = self.name.group(0)
        index = 'i' if self.category == 'repeated' else '0'
        wire_id = self.number * 8 + self.type.GetWireType()
        func_body = self.type.DecodeFunction(wire_id, index)
        cpp_type = self.type.GetCppType()
        if self.type.IsZigzag():
            func_body = 'UnZigZag(%s)' % func_body
        if self.type.IsIntegralType():
            func_body = 'lczero::kind_of_bit_cast<%s>(%s)' % (
                self.type.GetCppType(), func_body)

        if self.category == 'repeated':
            w.Write('size_t %s_size() const { return WireFieldCount(%d); }' % (
                name,
                wire_id,
            ))
            w.Write('%s %s(size_t i) const { return %s; }' % (
                cpp_type,
                name,
                func_body,
            ))
            w.Write('lczero::ProtoIterator<%s> %s() const {' %
                    (cpp_type, name))
            w.Write('  return lczero::ProtoIterator<%s>(%s_size(), '
                    '[this](size_t i) {' % (cpp_type, name))
            w.Write('    return %s;' % func_body)
            w.Write('  });')
            w.Write('}')

        else:
            w.Write('bool has_%s() const { return WireFieldCount(%d) > 0; }' %
                    (
                        name,
                        wire_id,
                    ))
            w.Write('%s %s() const { return %s; }' % (
                cpp_type,
                name,
                func_body,
            ))


class ProtoEnumParser:
    def __init__(self, lexer):
        lexer.Consume('enum')
        self.name = lexer.Consume('identifier').group(0)
        self.values = []
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

    def GetType(self):
        return 'enum'

    def IsType(self):
        return True

    def Generate(self, w):
        w.Write('enum %s {' % self.name)
        w.Indent()
        for key, value in self.values:
            w.Write('%s = %d,' % (key, value))
        w.Unindent()
        w.Write('};')


class ProtoMessageParser:
    def __init__(self, lexer, object_stack):
        self.objects = []
        lexer.Consume('message')
        self.name = lexer.Consume('identifier').group(0)
        lexer.Consume('{')
        while True:
            token, match = lexer.Pick()
            if token == '}':
                break
            elif token == 'message':
                self.objects.append(
                    ProtoMessageParser(lexer, [self.objects, *object_stack]))
            elif token == 'enum':
                self.objects.append(ProtoEnumParser(lexer))
            elif token in ['repeated', 'optional', 'required']:
                self.objects.append(
                    ProtoFieldParser(lexer, [self.objects, *object_stack]))
            else:
                lexer.Error('Expected field or type')
        lexer.Consume('}')

    def GetName(self):
        return self.name

    def GetType(self):
        return 'message'

    def IsType(self):
        return True

    def GetObjects(self):
        return self.objects

    def Generate(self, w):
        w.Write('class %s : public lczero::ProtoMessage {' % self.name)
        w.Write(' public:')
        w.Indent()
        w.Write('using ProtoMessage::ProtoMessage;')
        w.Write('%s(%s&&) = default;' % (self.name, self.name))
        w.Write(
            'static %s CreateNotOwned(std::string_view s) { return %s(s); }' %
            (self.name, self.name))
        for x in self.objects:
            x.Generate(w)
        w.Unindent()
        w.Write('};')


class ProtoFileParser:
    def __init__(self, lexer):
        self.package = None
        self.objects = []
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
        self.objects.append(ProtoMessageParser(lexer, [self.objects]))

    def Generate(self, w):
        w.Write('// This file is AUTOGENERATED, do not edit.')
        w.Write('#pragma once')
        w.Write('#include "utils/protomessage.h"')
        for x in self.package:
            w.Write('namespace %s {' % x)
        w.Indent()
        for object in self.objects:
            object.Generate(w)
        w.Unindent()
        for x in reversed(self.package):
            w.Write('}  // namespace %s' % x)


class Writer:
    def __init__(self, fo):
        self.fo = fo
        self.indent = 0

    def Indent(self):
        self.indent += 2

    def Unindent(self):
        self.indent -= 2

    def Write(self, text):
        self.fo.write(' ' * self.indent + text + '\n')


if __name__ == "__main__":
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
        writer = Writer(output)
        proto_file.Generate(writer)

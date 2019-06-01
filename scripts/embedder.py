#!/usr/bin/env python3

# Copyright (c) 2015 Jussi Pakkanen.
#
# This file is part of mcdemo.
#
# Mcdemo is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License.
#
# Mcdemo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with Mcdemo; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA

'''This file converts a binary file into a const array source
file that you can embed in your programs.
'''

import sys
import os

header_template = '''#pragma once
extern const unsigned char %s[%d];
'''

source_template_header = '''#include"%s"

const unsigned char %s[%d] = {'''
source_footer= '''
};
'''

assert(len(sys.argv) == 5)

_, datafile, srcfile, headerfile, arrayname = sys.argv

data = open(datafile, 'rb').read()
datasize = len(data)

open(headerfile, 'w').write(header_template % (arrayname, datasize))
ofile = open(srcfile, 'w')
ofile.write(source_template_header % (os.path.split(headerfile)[-1], arrayname, datasize))

for i, c in enumerate(data):
    if i % 16 == 0:
        ofile.write('\n    ')
    ofile.write("{:>3}, ".format(c))
ofile.write(source_footer)

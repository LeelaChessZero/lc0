# This file is part of Leela Chess Zero.
# Copyright (C) 2018-2020 The LCZero Authors
#
# Leela Chess is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Leela Chess is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.


# Check for 'xcrun' for compiling .metal shaders.
xcrun = find_program('xcrun', required: true)

shaders_air = custom_target('ShadersAir',
    output : 'Shaders.air',
    input : 'Shaders.metal',
    command : [xcrun, '-sdk', 'macosx', 'metal', '@INPUT@', '-c', '-o', '@OUTPUT@']
)

files += custom_target('ShadersMetallib',
    output : 'Shaders.metallib',
    input : shaders_air,
    command : [xcrun, '-sdk', 'macosx', 'metallib', '@INPUT@', '-o', '@OUTPUT@']
)

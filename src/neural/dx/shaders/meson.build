# This file is part of Leela Chess Zero.
# Copyright (C) 2020 The LCZero Authors
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

# Check for 'dxc' as 'dxc_helper.py' assumes it is on the path.
dxc = find_program('dxc', required: true)

# Used to work around a 'dxc' limitation with '/' in absolute paths.
dxc_helper = find_program('dxc_helper.py', required: true)

dxc_common = ['/Tcs_6_2', '/Fh', '@OUTPUT@', '@INPUT@']

files += custom_target('MatrixMul_Fp32',
    output : 'MatrixMul_Fp32.h',
    input : 'Gemm.hlsl',
    command : [dxc_helper, '/EMatrixMul', '/Vn', 'g_MatrixMul_Fp32'] + dxc_common
)

files += custom_target('MatrixMul_Fp16',
    output : 'MatrixMul_Fp16.h',
    input : 'Gemm.hlsl',
    command : [dxc_helper, '/EMatrixMul', '/Vn', 'g_MatrixMul_Fp16', '/DUSE_FP16_MATH=1',
               '-enable-16bit-types'] + dxc_common
)

files += custom_target('ExpandPlanes_shader_fp32',
    output : 'ExpandPlanes_shader_fp32.h',
    input : 'ExpandPlanes.hlsl',
    command : [dxc_helper, '/EExpandPlanes_shader_fp32'] + dxc_common
)

files += custom_target('ExpandPlanes_shader_fp16',
    output : 'ExpandPlanes_shader_fp16.h',
    input : 'ExpandPlanes.hlsl',
    command : [dxc_helper, '/EExpandPlanes_shader_fp16'] + dxc_common
)

files += custom_target('input_transform_shader_fp32',
    output : 'input_transform_shader_fp32.h',
    input : 'WinogradTransform.hlsl',
    command : [dxc_helper, '/Einput_transform_shader_fp32'] + dxc_common
)

files += custom_target('output_transform_shader_fp32',
    output : 'output_transform_shader_fp32.h',
    input : 'WinogradTransform.hlsl',
    command : [dxc_helper, '/Eoutput_transform_shader_fp32'] + dxc_common
)

files += custom_target('conv_1x1_shader_fp32',
    output : 'conv_1x1_shader_fp32.h',
    input : 'Conv1x1.hlsl',
    command : [dxc_helper, '/Econv_1x1_shader_fp32'] + dxc_common
)

files += custom_target('add_vectors_shader',
    output : 'add_vectors_shader.h',
    input : 'AddVectors.hlsl',
    command : [dxc_helper, '/Eadd_vectors_shader'] + dxc_common
)

files += custom_target('policy_map_shader_fp32',
    output : 'policy_map_shader_fp32.h',
    input : 'PolicyMap.hlsl',
    command : [dxc_helper, '/EPolicyMapShader', '/Vn', 'g_policy_map_shader_fp32'] + dxc_common
)

files += custom_target('output_transform_shader_fp32_se_128',
    output : 'output_transform_shader_fp32_se_128.h',
    input : 'WinogradTransformSE.hlsl',
    command : [dxc_helper, '/EOutputTransformSE', '/Vn', 'g_output_transform_shader_fp32_se_128',
               '/DBLOCK_SIZE=128'] + dxc_common
)

files += custom_target('output_transform_shader_fp32_se_256',
    output : 'output_transform_shader_fp32_se_256.h',
    input : 'WinogradTransformSE.hlsl',
    command : [dxc_helper, '/EOutputTransformSE', '/Vn', 'g_output_transform_shader_fp32_se_256',
               '/DBLOCK_SIZE=256'] + dxc_common
)

files += custom_target('output_transform_shader_fp32_se_320',
    output : 'output_transform_shader_fp32_se_320.h',
    input : 'WinogradTransformSE.hlsl',
    command : [dxc_helper, '/EOutputTransformSE', '/Vn', 'g_output_transform_shader_fp32_se_320',
               '/DBLOCK_SIZE=320'] + dxc_common
)

files += custom_target('output_transform_shader_fp32_se_384',
    output : 'output_transform_shader_fp32_se_384.h',
    input : 'WinogradTransformSE.hlsl',
    command : [dxc_helper, '/EOutputTransformSE', '/Vn', 'g_output_transform_shader_fp32_se_384',
               '/DBLOCK_SIZE=384'] + dxc_common
)

files += custom_target('output_transform_shader_fp32_se_512',
    output : 'output_transform_shader_fp32_se_512.h',
    input : 'WinogradTransformSE.hlsl',
    command : [dxc_helper, '/EOutputTransformSE', '/Vn', 'g_output_transform_shader_fp32_se_512',
               '/DBLOCK_SIZE=512'] + dxc_common
)

files += custom_target('output_transform_shader_fp32_se_640',
    output : 'output_transform_shader_fp32_se_640.h',
    input : 'WinogradTransformSE.hlsl',
    command : [dxc_helper, '/EOutputTransformSE', '/Vn', 'g_output_transform_shader_fp32_se_640',
               '/DBLOCK_SIZE=640'] + dxc_common
)

files += custom_target('output_transform_shader_fp32_se_768',
    output : 'output_transform_shader_fp32_se_768.h',
    input : 'WinogradTransformSE.hlsl',
    command : [dxc_helper, '/EOutputTransformSE', '/Vn', 'g_output_transform_shader_fp32_se_768',
               '/DBLOCK_SIZE=768'] + dxc_common
)

files += custom_target('output_transform_shader_fp32_se_1024',
    output : 'output_transform_shader_fp32_se_1024.h',
    input : 'WinogradTransformSE.hlsl',
    command : [dxc_helper, '/EOutputTransformSE', '/Vn', 'g_output_transform_shader_fp32_se_1024',
               '/DBLOCK_SIZE=1024'] + dxc_common
)

files += custom_target('se_128',
    output : 'se_128.h',
    input : 'SE.hlsl',
    command : [dxc_helper, '/ESE', '/Vn', 'g_se_128', '/DBLOCK_SIZE=128'] + dxc_common
)

files += custom_target('se_256',
    output : 'se_256.h',
    input : 'SE.hlsl',
    command : [dxc_helper, '/ESE', '/Vn', 'g_se_256', '/DBLOCK_SIZE=256'] + dxc_common
)

files += custom_target('se_320',
    output : 'se_320.h',
    input : 'SE.hlsl',
    command : [dxc_helper, '/ESE', '/Vn', 'g_se_320', '/DBLOCK_SIZE=320'] + dxc_common
)

files += custom_target('se_384',
    output : 'se_384.h',
    input : 'SE.hlsl',
    command : [dxc_helper, '/ESE', '/Vn', 'g_se_384', '/DBLOCK_SIZE=384'] + dxc_common
)

files += custom_target('se_512',
    output : 'se_512.h',
    input : 'SE.hlsl',
    command : [dxc_helper, '/ESE', '/Vn', 'g_se_512', '/DBLOCK_SIZE=512'] + dxc_common
)

files += custom_target('se_640',
    output : 'se_640.h',
    input : 'SE.hlsl',
    command : [dxc_helper, '/ESE', '/Vn', 'g_se_640', '/DBLOCK_SIZE=640'] + dxc_common
)

files += custom_target('se_768',
    output : 'se_768.h',
    input : 'SE.hlsl',
    command : [dxc_helper, '/ESE', '/Vn', 'g_se_768', '/DBLOCK_SIZE=768'] + dxc_common
)

files += custom_target('se_1024',
    output : 'se_1024.h',
    input : 'SE.hlsl',
    command : [dxc_helper, '/ESE', '/Vn', 'g_se_1024', '/DBLOCK_SIZE=1024'] + dxc_common
)


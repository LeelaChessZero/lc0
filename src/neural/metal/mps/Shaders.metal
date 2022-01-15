/*
 This file is part of Leela Chess Zero.
 Copyright (C) 2021 The LCZero Authors
 
 Leela Chess is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 Leela Chess is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.
 
 Additional permission under GNU GPL version 3 section 7
 
 If you modify this Program, or any covered work, by linking or
 combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
 Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
 modified version of those libraries), containing parts covered by the
 terms of the respective license agreement, the licensors of this
 Program grant you additional permission to convey the resulting work.
 */

#include <metal_stdlib>
using namespace metal;

//constant bool deviceSupportsNonuniformThreadgroups [[ function_constant(0) ]];

kernel void seMultiply(texture2d<float, access::read> source   [[ texture(0) ]],
                       texture2d<float, access::read> conv1    [[ texture(1) ]],
                       texture2d<float, access::read> conv2    [[ texture(2) ]],
                       texture2d<float, access::write> out     [[ texture(3) ]],
                       device float *buffer                    [[ buffer(0) ]],
                       uint2 gid                               [[ thread_position_in_grid ]],
                       uint2 tgid                              [[ thread_position_in_threadgroup ]],
                       uint2 tgsize                            [[ threads_per_threadgroup ]],
                       uint2 tgpos                             [[ threadgroup_position_in_grid ]],
                       uint width                              [[ thread_execution_width ]]) {

    //if (!deviceSupportsNonuniformThreadgroups) {
        if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
            return;
        }
    //}
    
    const auto c1 = conv1.read(gid);
    const auto c2 = conv2.read(gid);
    const auto s = source.read(gid);
    out.write(5.6381f, gid);
    buffer[gid.x + width * gid.y] = gid.x;
    buffer[1011] = gid.x;
    buffer[1012] = gid.y;
//    buffer[1013] = c1.z;
//    buffer[1014] = c1.a;
    buffer[1015] = c1.x;
    buffer[1016] = c1.y;
    buffer[1017] = c1.z;
    buffer[1018] = c1.a;
//    buffer[1019] = width;
//    buffer[1020] = tgsize.x;
//    buffer[1021] = tgsize.y;
//    buffer[1022] = tgpos.x;
//    buffer[1023] = tgpos.y;
    //threadgroup_barrier(mem_flags::mem_none);
}

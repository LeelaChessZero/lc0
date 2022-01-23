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

struct GridInfo {
    uint3 grid_size;
};

constant uint color_depth = 4;
constant uint2 board_size = uint2(8, 8);

inline float4 relu(float4 val) {
    float4 ret = float4(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < 4; i++)
        ret[i] = val[i] > 0.0f ? val[i] : 0;
    return ret;
}

inline float4 sigmoid(float4 val) { return 1.0f / (1.0f + exp(-val)); }

kernel void seMultiplyAdd(texture2d_array<float, access::read> sefc_a [[ texture(0) ]],
                       texture2d_array<float, access::read> conv_a [[ texture(1) ]],
                       texture2d_array<float, access::read> skip_a [[ texture(2) ]],
                       texture2d_array<float, access::write> out_a [[ texture(3) ]],
                       device float *buffer                     [[ buffer(0) ]],
                       constant GridInfo &grid_info             [[ buffer(1) ]],
                       uint3 gid                                [[ thread_position_in_grid ]]) {

    uint grid_width = grid_info.grid_size.x;
    uint grid_height = grid_info.grid_size.y;

    //if (!deviceSupportsNonuniformThreadgroups) {
        if (gid.x >= grid_width || gid.y >= grid_height) {
            return;
        }
    //}
    
    const auto slice = gid.y;
    const auto w = gid.x % board_size.x;
    const auto h = gid.x / board_size.x;
    const auto slices = conv_a.get_array_size();
    
    const auto conv = conv_a.read(uint2(w,h), slice);
    const auto skip = skip_a.read(uint2(w,h), slice);
    const auto gamma = sigmoid(sefc_a.read(uint2(0,0), slice));
    const auto beta = sefc_a.read(uint2(0,0), slice + slices);

    // Update output image texture.
    out_a.write(relu(gamma * conv + beta + skip), uint2(w,h), slice);
}


kernel void flatten(texture2d_array<float, access::read>  inp_a [[ texture(0) ]],
                    texture2d_array<float, access::write> out_a [[ texture(1) ]],
                    constant GridInfo &grid_info                [[ buffer(1) ]],
                    uint3 gid [[ thread_position_in_grid ]]) {

    uint grid_width = grid_info.grid_size.x;
    uint grid_height = grid_info.grid_size.y;

    //if (!deviceSupportsNonuniformThreadgroups) {
    if (gid.x >= grid_width || gid.y >= grid_height) {
        return;
    }
    //}
    
    const auto slice = gid.y / color_depth;
    const auto color = gid.y % color_depth;
    
    // float4 to use for collecting the 4 adjacent channel values.
    float4 flat = float4(0.0, 0.0, 0.0, 0.0);
    
    uint w, h;
    for (uint i = 0; i < color_depth; i++) {
        // Get actual pixel location matching current thread.
        w = (gid.x * color_depth + i) % board_size.x;
        h = (gid.x * color_depth + i) / board_size.x;
        
        // Set corresponding pixel in flattened array.
        flat[i] = inp_a.read(uint2(w,h), slice)[color];
    }
    // Write to output.
    out_a.write(flat, uint2(0,0), gid.x + gid.y * grid_width);

}

kernel void policyMap(texture2d_array<float, access::read>  inp_a [[ texture(0) ]],
                      texture2d_array<float, access::write> out_a [[ texture(1) ]],
                      constant short * map_buffer                 [[ buffer(0) ]],
                      constant GridInfo &grid_info                [[ buffer(1) ]],
                      uint2 gid [[ thread_position_in_grid ]]) {
    
    uint grid_width = grid_info.grid_size.x;
    uint grid_height = grid_info.grid_size.y;
    uint board = board_size.x * board_size.y;
    
    //if (!deviceSupportsNonuniformThreadgroups) {
    if (gid.x >= grid_width || gid.y >= grid_height) {
        return;
    }
    //}
    
    // float4 to use for collecting the 4 adjacent channel values.
    float4 policy = float4(0.0, 0.0, 0.0, 0.0);
    
    uint slice, color, w, h;
    short index;
    for (uint i = 0; i < color_depth; i++) {
        // Get policy map location matching current thread.
        index = map_buffer[gid.x * color_depth + i];

        // Map index to 8 x 8 x 80 convolution input.
        slice = (index / board) / color_depth;
        color = (index / board) % color_depth;
        w = (index % board) % board_size.x;
        h = (index % board) / board_size.x;
        
        // Set corresponding pixel in mapped policy.
        policy[i] = inp_a.read(uint2(w,h), slice)[color];
    }
    // Write to output.
    out_a.write(policy, uint2(0,0), gid.x);

}

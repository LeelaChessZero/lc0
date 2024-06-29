/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2020 The LCZero Authors

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

#pragma once

#include "ExpandPlanes_shader_fp32.h"
#include "ExpandPlanes_shader_fp16.h"
#include "input_transform_shader_fp32.h"
#include "output_transform_shader_fp32.h"
#include "conv_1x1_shader_fp32.h"
#include "add_vectors_shader.h"
#include "policy_map_shader_fp32.h"

#include "output_transform_shader_fp32_se_128.h"
#include "output_transform_shader_fp32_se_256.h"
#include "output_transform_shader_fp32_se_320.h"
#include "output_transform_shader_fp32_se_384.h"
#include "output_transform_shader_fp32_se_512.h"
#include "output_transform_shader_fp32_se_640.h"
#include "output_transform_shader_fp32_se_768.h"
#include "output_transform_shader_fp32_se_1024.h"

#include "se_128.h"
#include "se_256.h"
#include "se_320.h"
#include "se_384.h"
#include "se_512.h"
#include "se_640.h"
#include "se_768.h"
#include "se_1024.h"

#include "MatrixMul_Fp32.h"
#include "MatrixMul_Fp16.h"

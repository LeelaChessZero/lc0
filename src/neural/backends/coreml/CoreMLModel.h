/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2024 The LCZero Authors

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

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

/// Model Prediction Input Type
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0))
__attribute__((visibility("hidden")))
@interface CoreMLInput : NSObject<MLFeatureProvider>

/// input_planes as 1 × 112 × 8 × 8 4-dimensional array of floats
@property(readwrite, nonatomic, strong) MLMultiArray* _Nullable input_planes;
- (instancetype _Nonnull)init NS_UNAVAILABLE;
- (instancetype _Nonnull)initWithInput_planes:(MLMultiArray* _Nonnull)input_planes
    NS_DESIGNATED_INITIALIZER;

@end

/// Model Prediction Output Type
API_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0))
__attribute__((visibility("hidden")))
@interface CoreMLOutput : NSObject<MLFeatureProvider>

/// output_policy as 1 by 1858 matrix of floats
@property(readwrite, nonatomic, strong) MLMultiArray* _Nullable output_policy;

/// output_value as 1 by 3 matrix of floats
@property(readwrite, nonatomic, strong) MLMultiArray* _Nullable output_value;

/// output_moves_left as 1 by 1 matrix of floats
@property(readwrite, nonatomic, strong) MLMultiArray* _Nullable output_moves_left;
- (instancetype _Nonnull)init NS_UNAVAILABLE;
- (instancetype _Nonnull)initWithOutput_policy:(MLMultiArray* _Nonnull)output_policy
                                  output_value:(MLMultiArray* _Nonnull)output_value
                             output_moves_left:(MLMultiArray* _Nonnull)output_moves_left
    NS_DESIGNATED_INITIALIZER;

@end

@interface CoreMLModel : NSObject

+ (MLModel* _Nullable)getMLModel;
+ (void)setMLModel:(MLModel* _Nonnull)mlmodel;
+ (nullable NSArray<CoreMLOutput*>*)
    predictionsFromInputs:(NSArray<CoreMLInput*>* _Nonnull)inputArray
                  options:(MLPredictionOptions* _Nonnull)options
                    error:(NSError* _Nullable __autoreleasing* _Nullable)error;

@end
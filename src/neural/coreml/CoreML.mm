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

#import "CoreML.h"
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import "CoreMLModel.h"

namespace lczero {
namespace coreml_backend {

CoreML::CoreML() {
  NSString* modelPath = @"lc0.mlpackage";
  NSURL* modelURL = [NSURL fileURLWithPath:modelPath];
  dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
  NSLog(@"Compiling model: %@", modelURL);

  [MLModel compileModelAtURL:modelURL
           completionHandler:^(NSURL* compiledModelURL, NSError* error) {
             // Completion Handler
             if (error) {
               NSLog(@"Error compiling model: %@", error.localizedDescription);
             } else {
               NSLog(@"Compiled model URL: %@", compiledModelURL);
               NSLog(@"Initializing model with the compiled model URL...");
               NSError* modelInitError = nil;
               MLModelConfiguration* configuration = [[MLModelConfiguration alloc] init];
               // configuration.computeUnits = MLComputeUnitsCPUOnly;
               // configuration.computeUnits = MLComputeUnitsCPUAndGPU;
               // configuration.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
               configuration.computeUnits = MLComputeUnitsAll;
               MLModel* mlmodel = [MLModel modelWithContentsOfURL:compiledModelURL
                                                    configuration:configuration
                                                            error:&modelInitError];
               [CoreMLModel setMLModel:mlmodel];

               if (modelInitError) {
                 NSLog(@"Error initializing model: %@", modelInitError.localizedDescription);
               } else {
                 NSLog(@"Model successfully initialized");
               }
             }

             // Signal the semaphore to indicate that the task is complete
             dispatch_semaphore_signal(semaphore);
           }];

  // Wait for the semaphore
  dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
}

CoreML::~CoreML() {}

void CoreML::forwardEval(float* inputs, int batchSize, float* output_policy, float* output_value,
                         float* output_moves_left) {
  // Setup input array
  int inputChannels = 112;
  int inputLength = 8;
  int spatialSize = inputChannels * inputLength * inputLength;
  NSArray<NSNumber*>* inputShape = @[ @1, @(inputChannels), @(inputLength), @(inputLength) ];
  MLMultiArrayDataType inputDataType = MLMultiArrayDataTypeFloat;
  NSArray<NSNumber*>* inputStrides = @[
    @(inputChannels * inputLength * inputLength), @(inputLength * inputLength), @(inputLength), @1
  ];
  NSMutableArray<CoreMLInput*>* inputArray =
      [NSMutableArray arrayWithCapacity:(NSUInteger)batchSize];

  for (int i = 0; i < batchSize; i++) {
    NSError* arrayInitError = nil;
    MLMultiArray* input_planes =
        [[MLMultiArray alloc] initWithDataPointer:inputs + (i * spatialSize)
                                            shape:inputShape
                                         dataType:inputDataType
                                          strides:inputStrides
                                      deallocator:nil
                                            error:&arrayInitError];

    if (arrayInitError) {
      NSLog(@"Error initializing array: %@", arrayInitError.localizedDescription);
      break;
    }

    CoreMLInput* input = [[CoreMLInput alloc] initWithInput_planes:input_planes];
    [inputArray addObject:input];
  }

  // Setup prediction
  MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
  NSError* predictionError = nil;
  NSArray<CoreMLOutput*>* results = [CoreMLModel predictionsFromInputs:inputArray
                                                               options:options
                                                                 error:&predictionError];

  // Check error
  if (predictionError) {
    NSLog(@"Error predicting output: %@", predictionError.localizedDescription);
  } else {
    for (int i = 0; i < batchSize; i++) {
      MLMultiArray* policy = results[i].output_policy;
      for (int j = 0; j < policy.count; j++) {
        output_policy[j + (i * policy.count)] = [policy objectAtIndexedSubscript:j].floatValue;
      }

      MLMultiArray* value = results[i].output_value;
      float w = [value objectAtIndexedSubscript:0].floatValue;
      float d = [value objectAtIndexedSubscript:1].floatValue;
      float l = [value objectAtIndexedSubscript:2].floatValue;
      float m = std::max({w, d, l});
      w = std::exp(w - m);
      d = std::exp(d - m);
      l = std::exp(l - m);
      float sum = w + d + l;
      w /= sum;
      d /= sum;
      l /= sum;
      output_value[(i * 3)] = w;
      output_value[1 + (i * 3)] = d;
      output_value[2 + (i * 3)] = l;

      MLMultiArray* moves_left = results[i].output_moves_left;
      output_moves_left[i] = [moves_left objectAtIndexedSubscript:0].floatValue;
    }
  }
}

}  // namespace coreml_backend
}  // namespace lczero

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

CoreML::CoreML(bool wdl, bool moves_left) {
  wdl_ = wdl;
  moves_left_ = moves_left;

  NSString* modelPath = @"lc0.mlpackage";
  NSURL* modelURL = [NSURL fileURLWithPath:modelPath];
  dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
  NSLog(@"Compiling model: %@", modelURL);

  [MLModel compileModelAtURL:modelURL
           completionHandler:^(NSURL* compiledModelURL, NSError* error) {
             // Completion Handler
             @try {
               if (error) {
                 NSString* exceptionReason = [NSString
                     stringWithFormat:@"Failed to compile model: %@", error.localizedDescription];
                 @throw [NSException exceptionWithName:@"ModelCompilationException"
                                                reason:exceptionReason
                                              userInfo:@{@"error" : error}];
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
                   NSString* exceptionReason =
                       [NSString stringWithFormat:@"Failed to initialize model: %@",
                                                  modelInitError.localizedDescription];
                   @throw [NSException exceptionWithName:@"ModelInitializationException"
                                                  reason:exceptionReason
                                                userInfo:@{@"error" : modelInitError}];
                 } else {
                   NSLog(@"Model successfully initialized");
                 }
               }
             } @finally {
               // Signal the semaphore to indicate that the task is complete
               dispatch_semaphore_signal(semaphore);
             }
           }];

  // Wait for the semaphore
  dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
}

CoreML::~CoreML() {}

NSMutableArray<CoreMLInput*>* setupInputArray(float* inputs, int batchSize) {
  // Define the parameters for the input MLMultiArray
  int inputChannels = 112;
  int inputLength = 8;
  int spatialSize = inputChannels * inputLength * inputLength;
  NSArray<NSNumber*>* inputShape = @[ @1, @(inputChannels), @(inputLength), @(inputLength) ];
  MLMultiArrayDataType inputDataType = MLMultiArrayDataTypeFloat;
  NSArray<NSNumber*>* inputStrides =
      @[ @(spatialSize), @(inputLength * inputLength), @(inputLength), @1 ];

  // Initialize an array to hold CoreMLInput objects
  NSMutableArray<CoreMLInput*>* inputArray =
      [NSMutableArray arrayWithCapacity:(NSUInteger)batchSize];

  for (int i = 0; i < batchSize; i++) {
    NSError* arrayInitError = nil;

    // Calculate the pointer offset for the current batch item
    float* inputPointer = inputs + (i * spatialSize);

    // Initialize an MLMultiArray with the input data
    MLMultiArray* inputPlanes = [[MLMultiArray alloc] initWithDataPointer:inputPointer
                                                                    shape:inputShape
                                                                 dataType:inputDataType
                                                                  strides:inputStrides
                                                              deallocator:nil
                                                                    error:&arrayInitError];

    if (arrayInitError) {
      NSString* exceptionReason =
          [NSString stringWithFormat:@"Failed to initialize MLMultiArray for input: %@",
                                     arrayInitError.localizedDescription];
      @throw [NSException exceptionWithName:@"MLMultiArrayInitializationException"
                                     reason:exceptionReason
                                   userInfo:@{@"error" : arrayInitError}];
    }

    // Wrap the MLMultiArray in a CoreMLInput object and add it to the inputArray
    CoreMLInput* input = [[CoreMLInput alloc] initWithInput_planes:inputPlanes];
    [inputArray addObject:input];
  }

  return inputArray;
}

NSArray<CoreMLOutput*>* performPredictions(NSMutableArray<CoreMLInput*>* inputArray) {
  // Setup prediction options if necessary
  MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
  NSError* predictionError = nil;
  // Perform the prediction using the CoreML model
  NSArray<CoreMLOutput*>* results = [CoreMLModel predictionsFromInputs:inputArray
                                                               options:options
                                                                 error:&predictionError];

  if (predictionError) {
    NSString* exceptionReason =
        [NSString stringWithFormat:@"Prediction failed: %@", predictionError.localizedDescription];
    @throw [NSException exceptionWithName:@"PredictionException"
                                   reason:exceptionReason
                                 userInfo:@{@"error" : predictionError}];
  } else {
    return results;
  }
}

void processOutputs(CoreML* coreml, NSArray<CoreMLOutput*>* results, int batchSize,
                    float* output_policy, float* output_value, float* output_moves_left) {
  if (results) {
    for (int i = 0; i < batchSize; i++) {
      // Process policy output
      MLMultiArray* policy = results[i].output_policy;
      for (int j = 0; j < policy.count; j++) {
        output_policy[j + (i * policy.count)] = [policy objectAtIndexedSubscript:j].floatValue;
      }

      // Process value output
      MLMultiArray* value = results[i].output_value;
      float w = [value objectAtIndexedSubscript:0].floatValue;

      if (coreml->isWdl()) {
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
      } else {
        output_value[i] = w;
      }

      if (coreml->isMovesLeft()) {
        // Process moves left output
        MLMultiArray* moves_left = results[i].output_moves_left;
        output_moves_left[i] = [moves_left objectAtIndexedSubscript:0].floatValue;
      }
    }
  }
}

void CoreML::forwardEval(float* inputs, int batchSize, float* output_policy, float* output_value,
                         float* output_moves_left) {
  NSMutableArray<CoreMLInput*>* inputArray = setupInputArray(inputs, batchSize);
  NSArray<CoreMLOutput*>* results = performPredictions(inputArray);
  processOutputs(this, results, batchSize, output_policy, output_value, output_moves_left);
}

bool CoreML::isWdl() { return wdl_; }

bool CoreML::isMovesLeft() { return moves_left_; }

}  // namespace coreml_backend
}  // namespace lczero

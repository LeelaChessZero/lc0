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
               /* FIXME: MLComputeUnitsAll */
               configuration.computeUnits = MLComputeUnitsCPUAndGPU;
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

void CoreML::forwardEval(float* inputs, int batchSize, std::vector<float*> output_mems) {
  NSLog(@">>> CoreML::forwardEval");

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
    MLMultiArray* output_policy = results[0].output_policy;
    NSLog(@"CoreML::output_policy[0]: %f", [output_policy objectAtIndexedSubscript:0].floatValue);
    NSLog(@"CoreML::output_policy[1]: %f", [output_policy objectAtIndexedSubscript:1].floatValue);
    NSLog(@"CoreML::output_policy[2]: %f", [output_policy objectAtIndexedSubscript:2].floatValue);
    NSLog(@"CoreML::output_policy[3]: %f", [output_policy objectAtIndexedSubscript:3].floatValue);
    NSLog(@"CoreML::output_policy[4]: %f", [output_policy objectAtIndexedSubscript:4].floatValue);
    MLMultiArray* output_value = results[0].output_value;
    float w = [output_value objectAtIndexedSubscript:0].floatValue;
    float d = [output_value objectAtIndexedSubscript:1].floatValue;
    float l = [output_value objectAtIndexedSubscript:2].floatValue;
    float m = std::max({w, d, l});
    w = std::exp(w - m);
    d = std::exp(d - m);
    l = std::exp(l - m);
    float sum = w + d + l;
    w /= sum;
    d /= sum;
    l /= sum;
    NSLog(@"CoreML::w=%f d=%f l=%f", w, d, l);
    MLMultiArray* output_moves_left = results[0].output_moves_left;
    NSLog(@"CoreML::output_moves_left[0]: %f",
          [output_moves_left objectAtIndexedSubscript:0].floatValue);
  }

  NSLog(@"<<< CoreML::forwardEval");
}

}  // namespace coreml_backend
}  // namespace lczero

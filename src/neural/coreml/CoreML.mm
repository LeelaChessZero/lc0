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
               MLModel* model = [MLModel modelWithContentsOfURL:compiledModelURL
                                                  configuration:configuration
                                                          error:&modelInitError];

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

void CoreML::forwardEval(float* inputs, int batchSize, std::vector<float*> output_mems) {}

}  // namespace coreml_backend
}  // namespace lczero

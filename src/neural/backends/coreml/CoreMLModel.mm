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

#import "CoreMLModel.h"

@implementation CoreMLInput

- (instancetype)initWithInput_planes:(MLMultiArray*)input_planes {
  self = [super init];
  if (self) {
    _input_planes = input_planes;
  }
  return self;
}

- (NSSet<NSString*>*)featureNames {
  return [NSSet setWithArray:@[ @"input_planes" ]];
}

- (nullable MLFeatureValue*)featureValueForName:(NSString*)featureName {
  if ([featureName isEqualToString:@"input_planes"]) {
    return [MLFeatureValue featureValueWithMultiArray:self.input_planes];
  }
  return nil;
}

@end

@implementation CoreMLOutput

- (instancetype)initWithOutput_policy:(MLMultiArray*)output_policy
                         output_value:(MLMultiArray*)output_value
                    output_moves_left:(MLMultiArray*)output_moves_left {
  self = [super init];
  if (self) {
    _output_policy = output_policy;
    _output_value = output_value;
    _output_moves_left = output_moves_left;
  }
  return self;
}

- (NSSet<NSString*>*)featureNames {
  return [NSSet setWithArray:@[ @"output_policy", @"output_value", @"output_moves_left" ]];
}

- (nullable MLFeatureValue*)featureValueForName:(NSString*)featureName {
  if ([featureName isEqualToString:@"output_policy"]) {
    return [MLFeatureValue featureValueWithMultiArray:self.output_policy];
  }
  if ([featureName isEqualToString:@"output_value"]) {
    return [MLFeatureValue featureValueWithMultiArray:self.output_value];
  }
  if ([featureName isEqualToString:@"output_moves_left"]) {
    return [MLFeatureValue featureValueWithMultiArray:self.output_moves_left];
  }
  return nil;
}

@end

@implementation CoreMLModel

static MLModel* _Nullable sharedMLModel = nil;

+ (MLModel* _Nullable)getMLModel {
  @synchronized(self) {
    return sharedMLModel;
  }
}

+ (void)setMLModel:(MLModel* _Nonnull)mlmodel {
  @synchronized(self) {
    sharedMLModel = mlmodel;
  }
}

+ (nullable NSArray<CoreMLOutput*>*)
    predictionsFromInputs:(NSArray<CoreMLInput*>* _Nonnull)inputArray
                  options:(MLPredictionOptions* _Nonnull)options
                    error:(NSError* _Nullable __autoreleasing* _Nullable)error {
  MLModel* mlmodel = [CoreMLModel getMLModel];
  id<MLBatchProvider> inBatch =
      [[MLArrayBatchProvider alloc] initWithFeatureProviderArray:inputArray];
  id<MLBatchProvider> outBatch = [mlmodel predictionsFromBatch:inBatch options:options error:error];
  if (!outBatch) {
    return nil;
  }
  NSMutableArray<CoreMLOutput*>* results =
      [NSMutableArray arrayWithCapacity:(NSUInteger)outBatch.count];
  for (NSInteger i = 0; i < outBatch.count; i++) {
    id<MLFeatureProvider> resultProvider = [outBatch featuresAtIndex:i];
    CoreMLOutput* result = [[CoreMLOutput alloc]
        initWithOutput_policy:(MLMultiArray*)[resultProvider featureValueForName:@"output_policy"]
                                  .multiArrayValue
                 output_value:(MLMultiArray*)[resultProvider featureValueForName:@"output_value"]
                                  .multiArrayValue
            output_moves_left:(MLMultiArray*)[resultProvider
                                  featureValueForName:@"output_moves_left"]
                                  .multiArrayValue];
    [results addObject:result];
  }
  return results;
}

@end
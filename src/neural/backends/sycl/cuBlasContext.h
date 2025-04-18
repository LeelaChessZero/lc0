/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2023 Intel Corporation

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
   
  SPDX-License-Identifier:GNU General Public License v3.0 or later
*/

#include <iostream>
#include <sycl.hpp>

#ifdef USE_CUBLAS

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

class cuBlasContextManager;
static cuBlasContextManager *_cuBlasContextManager;

class cuBlasContextManager{
     
    //~cuBlasContextManager() { cublasDestroy(handle); }
    cublasHandle_t handle;

    cuBlasContextManager() {
        cublasCreate(&handle);
    }

    public:
     static cublasHandle_t getcuBlasHandle_t(){
        if(_cuBlasContextManager == NULL){
            _cuBlasContextManager = new cuBlasContextManager(); 
        }
        return _cuBlasContextManager->handle;
    }

    static cublasHandle_t destroycuBlasHandle_t(){
        if(_cuBlasContextManager != NULL){
           cublasDestroy(_cuBlasContextManager->getcuBlasHandle_t()); 
           free(_cuBlasContextManager); 
        }

        return _cuBlasContextManager->handle;
    }
};

#elif defined(USE_HIPBLAS)


#include "hip/hip_runtime.h" 
#include "hipblas.h"

class hipBlasContextManager;
static hipBlasContextManager *_hipBlasContextManager;

class hipBlasContextManager{
     
    //~cuBlasContextManager() { cublasDestroy(handle); }
    hipblasHandle_t handle;

    hipBlasContextManager() {
        hipblasCreate(&handle);
    }

    public:
     static hipblasHandle_t gethipBlasHandle_t(){
        if(_hipBlasContextManager == NULL){
            _hipBlasContextManager = new hipBlasContextManager(); 
        }
        return _hipBlasContextManager->handle;
    }

    static hipblasHandle_t destroycuBlasHandle_t(){
        if(_hipBlasContextManager != NULL){
           hipblasDestroy(_hipBlasContextManager->gethipBlasHandle_t()); 
           free(_hipBlasContextManager); 
        }

        return _hipBlasContextManager->handle;
    }
};



#endif

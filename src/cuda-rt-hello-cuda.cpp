#include<iostream>
#include<string>
#include"cuda_runtime.h"
#include "cuda.h"

#define checkRuntime(op) __cuda_check_runtime_api_((op),#op,__FILE__,__LINE__)

bool __cuda_check_runtime_api_(cudaError_t code,const char* op,const char* file,int line){
    if(code!= cudaSuccess){
        const char* err_name = nullptr;
        const char* err_message = nullptr;
        err_name = cudaGetErrorName(code);
        err_message = cudaGetErrorString(code);
        printf("%s:%d  %s  failed. error_name = %s,error_message = %s",file,line,op,err_name,err_message);
        return false;
    };
    return true;
};

int main(){

    CUcontext ctxA = nullptr;
    cuCtxGetCurrent(&ctxA);

    int device_count = 0;
    checkRuntime(cudaGetDeviceCount(&device_count));

    int device = 0;
    checkRuntime(cudaSetDevice(device));

    int current_device = 0;
    checkRuntime(cudaGetDevice(&current_device))


}
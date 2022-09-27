#include<iostream>
#include<string>
#include "cuda.h"

#define checkDriver(op) __check_cuda_driver((op),#op,__FILE__,__LINE__)

bool __check_cuda_driver(CUresult code,const char* op,const char* file,int line){
    if(code!= CUresult::CUDA_SUCCESS){
        const char* err_name = nullptr;
        const char* err_message = nullptr;
        cuGetErrorName(code,&err_name);
        cuGetErrorString(code,&err_message);
        printf("%s:%d,error_name = %s error_message = %s",file,line,err_name,err_message);
        return false;
    }
    return true;
}


int main(){
    checkDriver(cuInit(0));
    CUcontext ctxA = nullptr;
    CUcontext ctxB = nullptr;
    CUdevice device = 0;

    checkDriver(cuCtxCreate(&ctxA,CU_CTX_SCHED_AUTO,device));
    checkDriver(cuCtxCreate(&ctxB,CU_CTX_SCHED_AUTO,device));
    printf("ctxA = %p\n", ctxA);
    printf("ctxB = %p\n", ctxB);

    CUcontext current_ctx = nullptr;
    checkDriver(cuCtxGetCurrent(&current_ctx));

    CUcontext popped_ctx = nullptr;
    checkDriver(cuCtxPopCurrent(&popped_ctx));

    checkDriver(cuCtxPushCurrent(ctxA));

    checkDriver(cuCtxDestroy(ctxB));
    checkDriver(cuCtxDestroy(ctxA));

    return 0;
}
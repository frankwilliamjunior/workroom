#include<iostream>
#include<string>

#include"cuda.h"
#include"cuda_runtime.h"

#define checkRuntime(op) __check_cuda_runtime__((op),#op,__FILE__,__LINE__);

bool __check_cuda_runtime__(cudaError_t code,const char* op, const char* file,int line){
    if(code!= cudaSuccess){
        const char* err_name = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);
        printf("%s:%d   %s failed.  ERROR_NAME=%s,ERROR_MESSAGE=%s",file,line,op,err_name,err_message);
    return false;
    };
    return true;
}

int main(){
    int device = 0;
    checkRuntime(cudaSetDevice(device));

    
    float* device_mem;
    checkRuntime(cudaMalloc(&device_mem,100* sizeof(float)));


    float * pinned_mem;
    float* value = new float[100];
    value[2] = 255.5;
    pinned_mem[1]= 200.55;
    checkRuntime(cudaMallocHost((void **)&pinned_mem,100*sizeof(float)));

    checkRuntime(cudaMemcpy(&pinned_mem,&value,100*sizeof(float),cudaMemcpyHostToHost));
    checkRuntime(cudaMemcpy(&device_mem,&value,100*sizeof(float),cudaMemcpyHostToDevice));

    // 释放设备内存
    checkRuntime(cudaFree(device_mem));

    // 释放页锁定内存
    checkRuntime(cudaFreeHost(pinned_mem));

    // 释放主机内存
    delete[] value;

    return 0;
}
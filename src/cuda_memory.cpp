#include<string>
#include<iostream>
#include"cuda.h"

#define checkDriver(op) __cuda_check_driver((op),#op,__FILE__,__LINE__)

bool __cuda_check_driver(CUresult code,const char* op,const char* file,int line){
    if(code!=CUresult::CUDA_SUCCESS){
        const char* err_message;
        const char* err_name;
        cuGetErrorName(code,&err_name);
        cuGetErrorString(code,&err_message);
        printf("%s:%d  %sfailed. message:%s, error_name:%s",file,line,op,err_message,err_name);
        return false;
    }
    return true;
}

int main(){
    CUdeviceptr device_memory_ptr = 0;
    CUdevice device = 0;

    CUcontext ctx = nullptr;
    checkDriver(cuCtxCreate(&ctx,CU_CTX_SCHED_AUTO,device));

    //分配设备内存
    checkDriver(cuMemAlloc(&device_memory_ptr,100));

    //分配页锁定内存
    char* host_mem = nullptr;
    checkDriver(cuMemAllocHost_v2((void **)&host_mem,100));

    //设定内存初始值
    checkDriver(cuMemsetD32(device_memory_ptr,0,100));

    checkDriver(cuMemFreeHost((void *)host_mem));

    checkDriver(cuMemFree_v2(device_memory_ptr));


    return 0;

}
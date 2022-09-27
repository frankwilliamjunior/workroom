#include<iostream>
#include<string>

#include"cuda.h"
#include"cuda_runtime.h"

#define checkRuntime(op) __check_cuda_runtime__((op),#op,__FILE__,__LINE__)

bool __check_cuda_runtime__(cudaError_t code,const char *op,const char* file,int line){
    if(code!= cudaSuccess){
        const char* err_name = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);

        printf("%s:%d  %s failed. ERROR_NAME=%s  ERROR_MESSAGE=%s",file,line,op,err_name,err_message);
        return false
    };
    return true;
}

int main(){
    cudaStream_t f_stream = 0;

    //设定设备id，创建stream流
    int device = 0;
    checkRuntime(cudaSetDevice(device));
    checkRuntime(cudaStreamCreate(&f_stream));

    // 分配设备内存
    float* device_ptr=nullptr;
    checkRuntime(cudaMalloc((void **)&device_ptr,100*sizeof(float)));

    // 分配页锁定内存
    float * host_mem = nullptr;
    checkRuntime(cudaMallocHost((void **) host_mem,100*sizeof(float)));

    // 内存拷贝
    float* value = new float[100];
    value[1]=200.11;
    checkRuntime(cudaMemcpyAsync((void *)host_mem,(void *) value,100*sizeof(float),cudaMemcpyHostToHost,f_stream));

    checkRuntime(cudaMemcpyAsync((void*) device_ptr,(void *) value,100*sizeof(float),cudaMemcpyHostToDevice,f_stream));

    //同步流
    checkRuntime(cudaStreamSynchronize(f_stream));

    //释放内存  释放stream流
    checkRuntime(cudaFree(device_ptr));
    checkRuntime(cudaFreeHost(host_mem));
    checkRuntime(cudaStreamDestroy(f_stream));
    delete[] value;
    return 0;

}
#include<iostream>
#include<string>

#include"cuda.h"
#include"cuda_runtime.h"

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

void test_print(const float * pdata,int ndata);

int main(){
    float * parrayhost = nullptr;
    float * parraydevice = nullptr;
    int narray = 10;
    int arraybyte = narray*sizeof(float);

    parrayhost = new float[narray];

    checkRuntime(cudaMalloc(&parraydevice,arraybyte));
    for(int i=0;i<narray;++i)
        parrayhost[i]=i;
    
    checkRuntime(cudaMemcpy(parraydevice,parrayhost,arraybyte,cudaMemcpyHostToDevice));
    test_print(parraydevice,narray);

    checkRuntime(cudaDeviceSynchronize());
    checkRuntime(cudaFree(parraydevice));

    delete[] parrayhost;

    return 0;
}
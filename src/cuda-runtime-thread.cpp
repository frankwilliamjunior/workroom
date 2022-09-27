#include<iostream>
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

void launch(int* grids, int* blocks);

int main(){
    cudaDeviceProp prop;
    int device = 0;
    checkRuntime(cudaGetDeviceProperties(&prop,device));

    printf("max gridsize %d:%d:%d",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);
    printf("max threadsize %d:%d:%d",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);
    printf("warpsize:%d",prop.warpSize);
    printf("maxthreadsperblock:%d",prop.maxThreadsPerBlock);
    printf("maxBlocksPerMultiProcessor:%d",prop.maxBlocksPerMultiProcessor);

    int grid[] = {1,2,3};
    int block[]= {1024,1,1};
    launch(grid,block);
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaDeviceSynchronize());
    printf("Done\n.");

    return 0;
}
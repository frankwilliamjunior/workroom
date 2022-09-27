#include<stdio.h>

#include"cuda_runtime.h"
using namespace std;

__global__ void func(float* ptr){
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    if(tid == 999)
        ptr[999] = 5;
}


int main(){
    float* ptr = nullptr;
    func<<<100,10>>>(ptr);
    cudaError_t code = cudaPeekAtLastError();
    cout<<cudaGetErrorString(code)<<endl;

    cudaError_t code1 = cudaDeviceSynchronize();
    cout<<cudaGetErrorString(code1)<<endl;

    float* new_ptr = nullptr;
    auto code2 = cudaMalloc(&new_ptr,100*sizeof(float));
    cout<<cudaGetErrorString(code2)<<endl;
    return 0;

}
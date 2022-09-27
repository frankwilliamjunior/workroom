#include<iostream>
#include<string>
#include"cuda_runtime.h"
#include<math.h>

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

#define min(a,b) ((a) > (b) ? (b) : (a))

void launch_reduce_sum(const float* array,int n ,float *output);

int main(){
    const int n = 100;
    float* input_host = new float[n];
    float* input_device = nullptr;
    float ground_truth = 0;

    for(int i = 0;i<=n;++i){
        input_host[i] = i;
        ground_truth += i;
    }

    checkRuntime(cudaMalloc(&input_device,n * sizeof(float)));
    checkRuntime(cudaMemcpy(input_device,input_host,n * sizeof(float),cudaMemcpyHostToDevice));

    float output_host = 0;
    float* output_device = nullptr;
    checkRuntime(cudaMalloc(&output_device,sizeof(float)));
    checkRuntime(cudaMemset(output_device,0,sizeof(float)));

    launch_reduce_sum(input_device,n,output_device);
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(&output_host,output_device,n * sizeof(float),cudaMemcpyDeviceToHost));
    checkRuntime(cudaDeviceSynchronize());

    printf("output = %d,ground_truth = %d ",output_host,ground_truth);
    if(fabs(output_host - ground_truth) < __FLT_EPSILON__){
        printf("结果正确");
    }else{
        printf("结果错误");
    }
    checkRuntime(cudaFree(input_device));
    checkRuntime(cudaFree(output_device));
    delete[] input_host;
    printf("Done.\n");
    return 0;

}
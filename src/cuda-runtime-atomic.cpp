#include<iostream>
#include<string>
#include"cuda_runtime.h"

#define min(a, b)  ((a) < (b) ? (a) : (b))
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

void launch_atomic_add(float * input_arry,int input_size,float threshold,float * output_array,int output_capacity);

int main(){
    const int n = 1000000;
    float* input_device = nullptr;
    float* input_host = new float[n];
    for(int i = 0 ; i< n; ++i){
        input_host[i] = i % 100;
    };

    checkRuntime(cudaMalloc(&input_device, n *sizeof(float)));
    checkRuntime(cudaMemcpy(input_device,input_host, n * sizeof(float),cudaMemcpyHostToDevice));

    int output_capacity = 20;
    float * output_device = nullptr;
    float * output_host = new float[1+output_capacity*2];
    checkRuntime(cudaMalloc(&output_device,sizeof(float)*(1+output_capacity*2)));
    checkRuntime(cudaMemset(output_device,0,sizeof(float)*(1+output_capacity*2)));

    float threshold = 99;
    launch_atomic_add(input_device,n,threshold,output_device,output_capacity);
    checkRuntime(cudaPeekAtLastError());

    checkRuntime(cudaMemcpy(output_host,output_device,(1+output_capacity*2)* sizeof(float),cudaMemcpyDeviceToHost));
    checkRuntime(cudaDeviceSynchronize());

    printf("output_size = %f\n", output_host[0]);
    int output_size = min(output_capacity,output_host[0]);
    for(int i = 0; i< output_size;++i){
        float * output_item = output_host + 1 + i *2;
        float value = output_item[0];
        int index = output_item[1];
        printf("output_host[%d] = %f,%d",i,value,index);

    }

    checkRuntime(cudaFree(input_device));
    checkRuntime(cudaFree(output_device));
    
    delete[] input_host;
    delete[] output_host;
    printf("Done.\n");
    return 0;

}
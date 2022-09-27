#include<iostream>
#include"cuda_runtime.h"


__global__ void atomic_add(float * input_array,int input_size,float threshold,float* output_array,int output_capacity){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(input_array[tid] < threshold)
        return;
    // atomicAdd() 函数返回值为old值 即add之前的值 新值存储在&dst地址中
    int output_index = atomicAdd(output_array,1);
    if(output_index >= output_capacity)
        return;
    float * output_item = output_array + 1 + output_index * 2;
    output_item[0] = input_array[tid];
    output_item[1] = tid;
}

void launch_atomic_add(float * input_arry,int input_size,float threshold,float * output_array,int output_capacity){
    const int nthreads = 512;
    int block_size = nthreads > input_size ? input_size : nthreads;
    int grid_size = (input_size + block_size - 1)/block_size;

    atomic_add<<<grid_size,block_size,0,nullptr>>>(input_arry,input_size,threshold,output_array,output_capacity)

}
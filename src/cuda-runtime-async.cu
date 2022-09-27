#include<iostream>
#include<string>
#include"cuda_runtime.h"
#include<chrono>

using namespace std;

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

__global__ void add_vector(const float *a,const float * b,float* c,int count){
    int tid = threadIdx.x+blockDim.x*blockIdx.x;
    if(tid>=count)
        return;
    c[tid]= a[tid] + b[tid];
}
    /*
                dims                idx
            gridDim.z           blockIdx.z
            gridDim.y           blockIdx.y
            gridDim.x           blockIdx.x
            blockDim.z          threadIdx.z
            blockDim.y          threadIdx.y
            blockDim.x          threadIdx.x
tid = ((((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.z+threadIdx.z)
        *blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x

    */

__global__ void mul_vector(const float* a,const float* b,float* c,int count){
    int index = ((((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.z+threadIdx.z)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
    if(index>=count) return;
    c[index] = a[index] * b[index];
}

cudaStream_t stream1,stream2;
float *a,*b,*c1,*c2;
const int num_element = 100000;
const size_t bytes = num_element*sizeof(float);
const int blocks = 512;
const int grids = (num_element + blocks - 1)/blocks;
const int ntry = 100;

// 多个流异步
void async(){
    cudaEvent_t event_start1,event_start2;
    cudaEvent_t event_stop1,event_stop2;

    checkRuntime(cudaEventCreate(&event_start1));
    checkRuntime(cudaEventCreate(&event_start2));
    checkRuntime(cudaEventCreate(&event_stop1));
    checkRuntime(cudaEventCreate(&event_stop2));

    auto tic = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / (float) ntry;
    checkRuntime(cudaEventRecord(event_start1,stream1));
    for(int i = 0; i<= ntry;i++)
        add_vector<<<grids,blocks,0,stream1>>>(a,b,c1,num_element);
    checkRuntime(cudaEventRecord(event_stop1,stream1));
    
    checkRuntime(cudaEventRecord(event_start2,stream2));
    for(int i = 0;i<= ntry;i++)
        mul_vector<<<grids,blocks,0.stream2>>>(a,b,c2,num_element)
    checkRuntime(cudaEventRecord(event_stop2,stream2));

    checkRuntime(cudaStreamSynchronize(stream1));
    checkRuntime(cudaStreamSynchronize(stream2));

    auto toc = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / (float) ntry;
    
    float time1,time2;
    checkRuntime(cudaEventElapsedTime(&time1,event_start1,event_stop1));
    checkRuntime(cudaEventElapsedTime(&time1,event_start2,event_stop2));
    printf("sync time stream1 = %.2f stream2 = %.2f,count = %.2f",time1,time2,toc-tic);

}

// 单个流串行
void sync(){
    cudaEvent_t event_start1,event_stop1;

    checkRuntime(cudaEventCreate(&event_start1));
    checkRuntime(cudaEventCreate(&event_stop1));

    auto tic = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / (float) ntry;
    checkRuntime(cudaEventRecord(event_start1,stream1));
    for(int i = 0; i<= ntry;i++)
        add_vector<<<grids,blocks,0,stream1>>>(a,b,c1,num_element);
    
    for(int i = 0;i<= ntry;i++)
        mul_vector<<<grids,blocks,0.stream1>>>(a,b,c2,num_element);

    checkRuntime(cudaEventRecord(event_stop1,stream1));
    checkRuntime(cudaStreamSynchronize(stream1));

    auto toc = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / (float) ntry;
    
    float time1;
    checkRuntime(cudaEventElapsedTime(&time1,event_start1,event_stop1));
    printf("sync time %.2f,count = %.2f",time1,toc-tic);

}
// 多个流之间并行
void multi_stream_async(){
    #define step1 add_vector
    #define step2 mul_vector
    #define step3 add_vector
    #define step4 mul_vector
    #define stepa add_vector

    cudaEvent_t event_async;
    checkRuntime(cudaEventCreate(&event_async));

    step1<<<grids,blocks,0,stream1>>>(a,b,c1,num_element);
    step2<<<grids,blocks,0,stream1>>>(a,b,c1,num_element);

    checkRuntime(cudaStreamWaitEvent(stream1,event_async));
    step3<<<grids,blocks,0,stream1>>>(a,b,c2,num_element);
    step4<<<grids,blocks,0,stream1>>(a,b,c2,num_element);

    //流2
    stepa<<<grids,blocks,0,stream2>>>(a,b,c2,num_element);
    
    checkRuntime(cudaEventRecord(event_async,stream2));
    checkRuntime(cudaStreamSynchronize(stream1));
    printf("multi_stream_async done.\n");
}

int main(){
    checkRuntime(cudaStreamCreate(&stream1));
    checkRuntime(cudaStreamCreate(&stream2));

    checkRuntime(cudaMalloc(&a,bytes));
    checkRuntime(cudaMalloc(&b,bytes));
    checkRuntime(cudaMalloc(&c1,bytes));
    checkRuntime(cudaMalloc(&c2,bytes));

    async();

    sync();

    multi_stream_async();
    return 0;

}
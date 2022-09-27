#include<cuda_runtime.h>
#include<math.h>
#include<stdio.h>

__global__ void reduce_sum_kernel(const float* array,int n,const float* output){
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    /*
                Dims            idx
            gridDim.z        blockIdx.z
            gridDim.y        blockIdx.y
            gridDim.x        blockIdx.x
            blockDim.z       threadIdx.z
            blockDim.y       threadIdx.y
            blockDim.x       threadIdx.x
    */
    extern __shared__ float cache[];
    int block_size = blockDim.x;
    int lane = threadIdx.x;
    float  value = 0;

    if(tid < n)
    value = array[tid];

    for(int i = block_size/2;i > 0; i /=2){
        cache[lane] = value;
        __syncthreads();
        if(lane < i) value += cache[lane + 1];
        __syncthreads();  
    }

    if(lane == 0){
        printf("block %d value = %f\n", blockIdx.x, value);

        // 这里相当于每个block内的第一个线程会将block内的sum值加到output上，从而得到最终的sum值
        atomicAdd(output,value)
    }
   
}

void launch_reduce_sum(const float* array,int n,const float* output){
    const int nthreads = 512;
    int blocksize = n < nthreads ? n : nthreads;
    int grid_size = (n + blocksize -1)/blocksize;

    // blocksize 必须为2的倍数
    int block_sqrt = log2(blocksize);
    block_sqrt = ceil(block_sqrt);
    blocksize = pow(2,block_sqrt);

    reduce_sum_kernel<<<grid_size,blocksize,blocksize * sizeof(float),nullptr>>>(array,n,output);

}
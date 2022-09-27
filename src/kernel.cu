#include<stdio.h>
#include"cuda_runtime.h"

__global__ void test_print_kernel(const float* pdata, int ndata){
    int idx = threadIdx.x + blockIdx.x;
    int index = ((((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*blockDim.z+threadIdx.z)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x
    
    /*          dim             idx
            gridDim.z       blockIdx.z
            gridDim.y       blockIdx.y
            gridDim.x       blockIdx.x
            blockDim.z      threadIdx.z
            blockDim.y      threadIdx.y
            blockDim.x      threadIdx.x

            idx = (gridDim.x*blockIdx.y)+gridDim.y
    */
    printf("Element[%d] = %f, threadIdx.x=%d, blockIdx.x=%d, blockDim.x=%d\n", idx, pdata[idx], threadIdx.x, blockIdx.x, blockDim.x);


}

void test_print(const float* pdata, int ndata){

    // <<<gridDim, blockDim, bytes_of_shared_memory, stream>>>
    test_print_kernel<<<1,ndata,0,nullptr>>>(pdata,ndata);

    cudaError_t code = cudaPeekAtLastError();
    if(code!=cudaSuccess){
        const char* err_name = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);
        printf("kernel error %s:%d  test_print_kernel failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, err_name, err_message);   

    }

}

// #include<iostream>
// #include"cuda_runtime.h"

// void launch();

// __shared__ char stsatic_shared_mem;
// extern __shared__ char dynamic_shared_mem;


// // __syncthreads()同步block内所有线程

// int __snycthreads();


// int main(){
//     cudaDeviceProp prop;
//     checkRuntime(cudaGetDeviceProperties(&prop,0));
//     printf("prop.sharedMemPerBlock =%.2f KB\n",prop.sharedMemPerBlock/1024.0f);
//     launch();
//     checkRuntime
// }
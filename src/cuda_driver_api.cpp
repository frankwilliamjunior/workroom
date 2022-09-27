#include"cuda.h"
#include<stdio.h>
#include<string.h>

//没有对应的cuDestroy(),不需要释放

// int main(){
//     CUresult code = cuInit(0);
//     if (code != CUresult::CUDA_SUCCESS){
//     const char* err_message = nullptr;
//     cuGetErrorString(code,&err_message);
//     printf("Initializ failed. code = %d, message = %s\n",code,err_message);
//     return -1;
//     }
//     CUresult all_code[10];


//     return 0;
// }

// #define checkDriver(op) \
// do{                     \
//     auto code = (op);   \
//     if(code != CUresult::CUDA_SUCCESS){     \
//         const char* err_name = nullptr;     \
//         const char* err_message = nullptr;  \
//         cuGetErrorName(code,&err_name);     \
//     }                                       \
// }while(0)


// #define checkDriver(op) __check_cuda_driver((op),#op,__FILE__,__LINE__)

// bool __check_cuda_driver(CUresult code,const char* op, const char* file,int line){
//     if(code!=CUresult::CUDA_SUCCESS){
//         const char* err_name = nullptr;
//         const char* err_message = nullptr;
//         cuGetErrorName(code,&err_name);
//         cuGetErrorString(code,&err_message);
//         printf("%s:%d  %s failed. \n code = %s,message = %s\n",file,line,op,err_name,err_message);
//         return false;
//     }
// }


#define checkDriver(op) __cuda_check_driver((op),#op,__FILE__,__LINE__)

bool __cuda_check_driver(CUresult code,const char* op,const char* file,int line){
    if(code!=CUresult::CUDA_SUCCESS){
        const char* err_message;
        const char* err_name;
        cuGetErrorName(code,&err_name);
        cuGetErrorString(code,&err_message);
        printf("%s:%d  %sfailed. message:%s, error_name:%s",file,line,op,err_message,err_name);
        return false;
    }
    return true;
}

int main(){
    CUresult code;
    // checkDriver(cuInit(0))
    code = cuInit(0);
    const char* err_message = nullptr;
    const char* err_name = nullptr;
    cuGetErrorString(code,&err_message);
    cuGetErrorName(code,&err_name);
    printf("cu Initialize failed. message:%s, error_name:%s",err_message,err_name);
    return 0;
}
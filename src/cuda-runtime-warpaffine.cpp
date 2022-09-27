#include<iostream>
#include<string>
#include"cuda_runtime.h"
#include"opencv2/opencv.hpp"

using namespace cv;
#define min(a,b)  ((a) > (b) ? (b) : (a))

#define checkRuntime(op) __check_cuda_runtime((op),#op,__FILE__,__LINE__)

bool __check_cuda_runtime(cudaError_t code,const char* op,const char* file,int line){
    if(code!= cudaSuccess){
        const char* err_message = cudaGetErrorString(code);
        const char* err_name = cudaGetErrorName(code);
        printf("%s:%d,%s failed. error messsage = %s,error name = %s",file,line,op,err_message,err_name);
        return false;
    }
    return true;
}

void warp_affine_bilinear(
    uint8_t * src,int src_line_size,int src_width,int src_height,
    uint8_t * dst,int dst_line_size,int dst_width,int dst_height,
    uint8_t * fill_value
)

Mat warpaffine_center_align(Mat &image,const Size&size){

    Mat output(size,CV_8UC3);
    uint8_t psrc_device = nullptr;
    uint8_t pdst_device = nullptr;

    size_t src_device = image.cols * image.rows * 3;
    size_t dst_device = size.width * size.height * 3;

    checkRuntime(cudaMalloc(&psrc_device,src_device));
    checkRuntime(cudaMalloc(&pdst_device,dst_device))
    checkRuntime(cudaMemcpy(psrc_device,image.data,src_device,cudaMemcpyHostToDevice);

    warp_affine_bilinear(psrc_device,image.cols * 3,image.cols,image.rows,
                            pdst_device,size.width * 3,size.width,size.height,114);
    
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(output.data,pdst_device,dst_device,cudaMemcpyDeviceToHost));
    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));

    return output;
}

int main(){
    Mat image = imread("test.jpg");
    Mat output = warpaffine_center_align(image,Size(640,640))
    imread("output.jpg",output);
    printf("Done. save to output.jpg\n")
    return 0;
}
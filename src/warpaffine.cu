#include<cuda_runtime.h>
#include<cmath>
#define num_thread 512

typedef unsigned char uint8_t;
using namespace std;
struct Size{
    int width = 0,height = 0;
    Size() = default;
    Size(int w,int h):width(w),height(h){}
};

struct affine {
    float img2dst[6];
    float dst2img[6];

    void invertmatrix(){
        float D = img2dst[0]*img2dst[4] - img2dst[2] * img2dst[3];
        D = D!=0 ? 1/D :0;
        float i00 = img2dst[0], i01 = img2dst[1], i02 = img2dst[2];
        float i10 = img2dst[3], i11 = img2dst[4], i12 = img2dst[5];
        //              0                  0                1
        dst2img[0] = D*(i11);
        dst2img[1] = D*(-i01);
        dst2img[2] = D*(i01 * i12 - i02*i11);
        dst2img[3] = D*(-i10);
        dst2img[4] = D*(i00);
        dst2img[5] = D*(i10*i02 - i00*i12);

    };
    void compute(const Size src_size,const Size dst_size){
        float scale_x = dst_size.width/src_size.width;
        float scale_y = dst_size.height/src_size.height;
        float scale = min(scale_x,scale_y);
        img2dst[0] = scale,img2dst[1] = 0,img2dst[2] = (dst_size.width - scale*src_size.width + scale-1)*0.5;
        img2dst[3] = 0,img2dst[4] = scale,img2dst[5] = (dst_size.height - scale*src_size.height + scale-1)*0.5;
        invertmatrix();
    };
};

__global__ void warpaffine_kernel(
    uint8_t* src, int src_width, int src_height, int src_line_size,
    uint8_t* dst, int dst_width, int dst_height, int dst_line_size,
    uint8_t fill_value,affine matrix
    ){
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    float src_x = matrix.dst2img[0]* dx + matrix.dst2img[1] * dy + matrix.dst2img[2];
    float src_y = matrix.dst2img[3]* dx + matrix.dst2img[4] * dy + matrix.dst2img[5];
    // 设置默认填充值
    uint8_t c0 = fill_value,c1 = fill_value,c2 = fill_value;

    if(src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height){

    }else{
        int x_low = floorf(src_x);
        int y_low = floorf(src_y);
        int x_high = floorf(src_x) + 1;
        int y_high = floorf(src_y) + 1;
        float lx = src_x - x_low;
        float ly = src_y - x_low;
        float hx = x_high - src_x;
        float hy = y_high - src_y;
        // 求原图目标像素 周围四个像素点的权重 注意这里直接求解
        float w1 = hx * hy;      // 左上点 topleft
        float w2 = lx * hy;      // 右上点 topright
        float w3 = hx * ly;      // 左下点 bottomleft
        float w4 = lx * ly;      // 右下点 bottomright

        // 设置默认值
        uint8_t const_value[] = {fill_value,fill_value,fill_value};
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;
        
        // 找到四个像素点的地址
        if(y_low >= 0){
            if(x_low >= 0)
                v1 = src + y_low*src_line_size + x_low *3;
            if(x_high <= src_width);
                v2 = src + y_low * src_line_size + x_high *3;
        }
        if(y_high <= src_height){
            if(x_low >= 0 )
                v3 = src + y_high*src_line_size + x_low *3;
            if(x_high <= src_width)
                v4 = src + y_high*src_line_size + x_high *3;
        }

        c0 = floorf(w1*v1[0] + w2 *v2[0] + w3*v3[0] + w4*v4[0] + 0.5f);
        c1 = floorf(w1*v1[1] + w2 *v2[1] + w3*v3[1] + w4*v4[1] + 0.5f);
        c2 = floorf(w1*v1[2] + w2 *v2[2] + w3*v3[2] + w4*v4[2] + 0.5f);
    }
    uint8_t* pdst = dst + dy * dst_line_size + dx *3;
    pdst[0] = c0, pdst[1] = c1, pdst[2] = c2;
}

void warpaffine(
    uint8_t* src, int src_width, int src_height, int src_line_size,
    uint8_t* dst, int dst_width, int dst_height, int dst_line_size,
    uint8_t fill_value
){
    dim3 blocksize = (32,32);
    dim3 gridsize = ((int) (dst_width + 31)/32,(int)(dst_height +31)/32);

    affine matrix;
    matrix.compute(Size(src_width,src_height),Size(dst_width,dst_height));

    warpaffine_kernel<<<gridsize,blocksize,0,nullptr>>>(
    src, src_width, src_height, src_line_size,
    dst, dst_width, dst_height, dst_line_size,
    fill_value,matrix);

}
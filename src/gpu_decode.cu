#include<cuda_runtime.h>
#include<stdio.h>

static __device__  float BoxIou(
    float aleft,float aright,float atop,float abottom,
    float bleft,float bright,float btop,float bbottom
){
    float cleft = max(aleft,bleft);
    float cright = min(aright,bright);
    float ctop = max(atop,btop);
    float cbottom = min(abottom,bbottom);
    float carea = max(cright - cleft,0.0f) * max(cbottom - ctop,0.0f);

    if(carea==0.0f)
        return 0.0f;
    float aarea = max(0.0f,aright- aleft) * max(0.0f,abottom-atop);
    float barea = max(0.0f,bright- bleft) * max(0.0f,bbottom-btop);

    float iou = carea/(aarea + barea -carea);
    return iou    
}

static __device__ void fast_nms_kernel(
    float* boxes,
    float iou_threshold,int max_objects,
    int NUM_BOX_ELEMENT
){
    int ids = blockIdx.x * blockDim.x + threadIdx.x;
    // 此处boxes的第一个位置是储存了box objects 的数量
    int count = min((int)*boxes,max_objects);
    if(ids >= count)
        return;
    float* pcurrent = boxes + 1 +ids * NUM_BOX_ELEMENT;
    for(int i = 0;i < count;++i){
        float* pitem = boxes + 1 + i *NUM_BOX_ELEMENT;
        if(i == ids || pcurrent[5] != pitem[5]) continue;

        if(pitem[4] >= pcurrent[4]){
            if(pitem[4] == pcurrent[4] && i< ids)
                continue;
            float iou = BoxIou(
            pcurrent[0],pcurrent[1],pcurrent[2],pcurrent[3],
            pitem[0],   pitem[1],   pitem[2],   pitem[3]
            );
            if(iou > iou_threshold){
                pcurrent[6] = 0;        // remove flag  1 for keep  0 for ignore
                return;
            }
        }
    }

}

vector<Box> cpu_decode(float* predict,int rows,int cols,float confidence_threshold = 0.25f,
float nms_threshold = 0.45f){
    vector<Box> boxes;
    int num_classes = cols - 5;
    for(int i = 0;i<=rows;++i){
        float * pitem = predict + i*cols;
        float objectness = pitem[4];
        if( objectness < confidence_threshold)
            continue;
        
        float* pclass = pitem + 5;
        int label = std::max_element(pclass,pclass+num_classes) - pclass;
        float prob = pclass[label];
        float confidence = objectness * prob;
        if(confidence < confidence_threshold)
            continue;

        float cx    = pitem[0];
        float cy    = pitem[1];
        float width = pitem[2];
        float height = pitem[3];
        float left = cx - width * 0.5f;
        float right = cx + width * 0.5f;
        float top = cy - width * 0.5f;
        float bottom = cy + width * 0.5f;
        boxes.emplace_back(left,right,top,bottom,confidence,(int)label);
            
    }
    std::sort(boxes.begin(),boxes.end(),[](Box& a,Box& b){return a.confidence > b.confidence;});
    std::vector<bool> remove_flags(boxes.size());       // 默认值全为0
    std::vector<Box> box_result;
    box_result.reserve(boxes.size());

    // 计算iou值
    auto iou = [](const Box& a,const Box& b){
        float cross_left  = std::max(a.left,b.left);
        float cross_right = std::min(a.right,b.right);
        float cross_top = std::max(a.top,b.top);
        float cross_bottom = std::min(a.bottom,b.bottom);

        float cross_area = std::max(cross_right-cross_left,0.0f) * std::max(cross_bottom - cross_top,0.0f);
        float union_area = std::max(a.right - a.left,0.0f) * std::max(a.bottom - a.top,0.0f) + 
        std::max(b.right - b.left,0.0f) * std::max(b.bottom - b.top,0.0f) - cross_area;
        if( cross_area == 0||union_area== 0) return 0.0f;

        return cross_area / union_area;
    };

    for(int i = 0;i < boxes.size();++i){
        if(remove_flags(i)) continue;
        auto& ibox = boxes[i];
        box_result.emplace_back(ibox);
        for(int j = i+1; j <boxes.size();++j){
            if(remove_flags[j]) continue;
            auto& jbox = boxes[j];
            if(ibox.label == jbox.label){
                if(iou(ibox,jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }
    return box_result;
}

// 依据阈值对预测结果进行过滤
static __global__ void decode_kernel(
    float* predict,int num_bboxes,int num_classes,float conf_threshold,
    float* invertAffinematrix,float* parray,int max_objects,int NUM_BOX_ELEMENT
){
    int ids = blockIdx.x*blockDim.x + threadIdx.x;
    if(ids >= num_bboxes)
        return;
    
    float* pitem = predict + ids* ( 5+num_classes);
    float objectness = pitem[4];
    if(objectness < conf_threshold)
        return;
    // 找到最大类别置信度
    float* class_confidence = pitem + 5;
    float confidence = *class_confidence++;     // 从第一个类别概率开始
    int label = 0;
    for (int i = 0;i<num_classes;++i,++class_confidence){
        if(*class_confidence > confidence){
            confidence = *class_confidence;
            label = i;
        }

    }
    // 最大类别概率乘以置信度
    confidence *= objectness;
    if (confidence < conf_threshold)
        return;
    
    int index =atomicAdd(parray,1);
    if( index >= max_objects)
        return;
    // 此处*pitem++ 先执行*pitem 取值操作  后执行pitem++ 指针加1 操作
    // center_x  center_y width  height
    float cx    = *pitem++;
    float cy    = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float right = cx + width * 0.5f;
    float top = cy - height*0.5f;
    float bottom = cy + height*0.5f;

    // 此处parray + 1 的原因是第一个位置储存了objects 数量
    float* pout_item = parray + 1 + ids * NUM_BOX_ELEMENT;
    // 此处同理 先执行*pout_item = left 操作 后执行 pout_item++
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;           // 1  for keep;  0 for ignore
}

void decode_result(
    float* predict,int num_bboxes,int num_classes,float confidence_threshold,
    float nms_threshold,float* invertAffinematrix,float* parray,int max_objects,
    int NUM_BOX_ELEMENT,cudastream_t stream
){
    auto block = num_bboxes > 512 ? 512: num_bboxes;
    auto grid  = (num_bboxes + block -1)/block;
    decode_kernel<<<grid,block,0 stream>>>(
        predict, num_bboxes, num_classes, conf_threshold,
        invertAffinematrix, parray, max_objects, NUM_BOX_ELEMENT
    );
    block = max_objects > 512 ? 512 : max_objects;
    grid = (max_objects + block - 1)/block;

    fast_nms_kernel<<<grid, block ,0 , stream>>>(boxes, iou_threshold, max_objects, NUM_BOX_ELEMENT)

}

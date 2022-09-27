#ifndef BOX_HPP
#define BOX_HPP
struct Box{
    float top, bottom, right, left,confidence;
    int label;
    Box()= default;
    Box(float top,float bottom,float right,float left, float confidence,int label):
    top(top),bottom(bottom),right(right),left(left),confidence(confidence),label(label){}
}

#endif
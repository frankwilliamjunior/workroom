#include<iostream>
#include<string>
#include"cuda_runtime.h"
#include"NvInfer.h"
#include"NvInferRuntime.h"


class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kVERBOSE){
            printf("%d: %s\n", severity, msg);
        }
    }
};

nvinfer1::Weights make_weights(float* ptr,int n){
    nvinfer1::Weights w;
    w.count = n;                    // 元素个数
    w.type = nvinfer1::DataType::kFLOAT;    // 数据类型
    w.values = ptr;                 // 权重值  指针
    return w;
}

int main(){
    // 实例化logger
    TRTLogger logger;
    // 调用Nvinfer1的createInferBuilder 方法创建 Nvinfer1::IBuilder* 指针
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

    // 调用 builder 的createBuilderConfig 方法创建 nvinfer1::IBuilderConfig* 指针
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    // builder->createNetworkV2(1) 采用显性batchsize  0则为隐性batchsize
    nvinfer1::INetworkDefinition* network = builder-> createNetworkV2(1);

    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0,2.0,0.5,0.1,0.2,0.5};
    float layer1_bias_values[] = {0.3,0.8};

    // 调用network 的addInput 方法 节点名 数据类型 节点形状
    nvinfer1::ITensor* input = network->addInput("image",nvinfer1::DataType::kFLOAT,nvinfer1::Dims4(1,num_input,1,1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values,6);
    nvinfer1::Weights layer1_bias  = make_weights(layer1_bias_values,2);

    // 添加全连接层
    auto layer1 = network-> addFullyConnected(*input,num_output,layer1_weight,layer1_bias);
    // 添加激活层
    auto prob = network-> addActivation(*layer1->getOutput(0),nvinfer1::ActivationType::kSIGMOID);

    // 将需要的prob标记为输出
    network->markOutput(*prob->getOutput(0));

    printf("Workspace size = %.2f MB\n",(1<<28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1<<28);
    builder->setMaxBatchSize(1);

    // 生成engine文件
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network,*config);
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return -1;
    }

    // 序列化模型文件并保存
    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen("engine.trtmodel","wb");
    fwrite(model_data->data(),1,model_data->size(),f);
    fclose(f);

    // 按倒序释放指针
    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return 0;

}

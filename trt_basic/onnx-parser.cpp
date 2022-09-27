#include<NvInfer.h>
#include<NvOnnxParser.h>

#include<NvInferRuntime.h>

#include<cuda_runtime.h>

#include<stdio.h>
#include<math.h>
#include<iostream>
#include<fstream>
#include<vector>

using namespace std;

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:return "error";
        case nvinfer1::ILogger::Severity::kWARNING:return "warning";
        case nvinfer1::ILogger::Severity::kINFO: return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default:return "unknow";
    }
}

class TRTLogger:public nvinfer1::ILogger{
public:
    virtual void log(Severity severity,nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kINFO){
            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);

            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;

bool build_model(){

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(network,logger);
    if(!parser->parseFromFile("demo.onnx", 1)){
        printf("Failed to parser demo.onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }

    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n",(1 << 28) / 1024.0f/1024.0f);
    config->setMaxWorkspaceSize(1<<28);

    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    int input_channel = input_tensor->getDimensions().d[1];

    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, input_channel, 5, 5));
    // 添加到配置
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // -------------------------- 3. 序列化 ----------------------------------
    // 将模型序列化，并储存为文件
    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    parser->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return true;
}
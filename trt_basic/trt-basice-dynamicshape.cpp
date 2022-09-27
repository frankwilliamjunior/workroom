#include<stdio.h>
#include<fstream>
#include<vector>
#include"cuda_runtime.h"
#include"NvInfer.h"
#include"NvInferRuntime.h"

using namespace std;

std::vector<unsigned char> load_file(const string& file){
    ifstream in(file,ios::in|ios::binary);
    if(!in.is_open()){
        return;
    }
    int length;
    in.seekg(0,ios::end);
    length = in.tellg();
    std::vector<uint8_t> data;
    if(length > 0){
        in.seekg(0,ios::beg);
        data.resize(length);
        in.read((char*)&data[0],length);
    }
    in.close();
    return data;
}

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity,nvinfer1::AsciiChar const * msg) noexcept override{
        if(severity<=Severity::kINFO){
            printf("%d:%s\n",severity,msg);
        }
    }
}logger;

nvinfer1::Weights make_weights(float* ptr,int n){
    nvinfer1::Weights w;
    w.count = n;
    w.values = ptr;
    w.type = nvinfer1::DataType::kFLOAT;
    return w;
};

bool build_model(){
    TRTLogger logger;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    const int num_input = 1;
    const int num_output = 1;
    float layer1_weight_values[] = {
        1.0, 2.0, 3.1, 
        0.1, 0.1, 0.1, 
        0.2, 0.2, 0.2
    };
    float layer1_bias_values[] = {0.0};
    nvinfer1::Weights layer1_weights = make_weights(layer1_weight_values,9);
    nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values,1);

    nvinfer1::ITensor* input = network->addInput("image",nvinfer1::DataType::kFLOAT,nvinfer1::Dims4(-1,num_input,-1,-1));
    auto layer1 = network->addConvolution(*input,num_output,nvinfer1::DimsHW(3,3),layer1_weights,layer1_bias);

    layer1->setPadding(nvinfer1::DimsHW(1,1));

    auto prob = network->addActivation(*layer1->getOutput(0),nvinfer1::ActivationType::kRELU);

    network->markOutput(*prob->getOutput(0));

    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n",(1<<28)/1024.0f/1024.0f);
    config->setMaxWorkspaceSize(1<<28);

    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(input->getName(),nvinfer1::OptProfileSelector::kMIN,nvinfer1::Dims4(1,num_input,3,3));
    profile->setDimensions(input->getName(),nvinfer1::OptProfileSelector::kOPT,nvinfer1::Dims4(1,num_output,3,3));

    profile->setDimensions(input->getName(),nvinfer1::OptProfileSelector::kMAX,nvinfer1::Dims4(maxBatchSize,num_input,5,5));
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network,*config);
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen("engine.trtmodel","wb");
    fwrite(model_data->data(),1,model_data->size(),f);
    fclose(f);

    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return true;
}


void inference(){
    TRTLogger logger;
    auto engine_data = load_file("engine.trtmodel");
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(),engine_data.size());
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }
    nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    float input_data_host[] = {
        // batch 0
        1,   1,   1,
        1,   1,   1,
        1,   1,   1,

        // batch 1
        -1,   1,   1,
        1,   0,   1,
        1,   1,   -1
    };
    float* input_data_device = nullptr;
    int ib = 2;
    int iw = 3;
    int ih = 3;
    float output_data_host[ib * iw * ih];
    float* output_data_device = nullptr;
    cudaMalloc(&input_data_device,sizeof(input_data_host));
    cudaMalloc(&output_data_device,sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device,input_data_host,sizeof(input_data_host),cudaMemcpyHostToDevice);

    execution_context->setBindingDimensions(0,nvinfer1::Dims4(ib,1,ih,iw));
    float* bindings[] = {input_data_device,output_data_device};
    bool success = execution_context->enqueueV2((void**)bindings,stream,nullptr);
    cudaMemcpyAsync(output_data_host,output_data_device,sizeof(output_data_device),cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);

    for(int b = 0;b < ib; ++b){
        printf("batch %d.output_data_host = \n",b);
        for(int i = 0; i < iw * ih; ++i){
            printf("%f,",output_data_host[b * iw * ih + i]);
            if((i + 1)% iw == 0)
            printf("\n");

        }
    }
    printf("Clean memory\n");
    cudaStreamDestroy(stream);
    cudaFree(input_data_device);
    cudaFree(output_data_device);
    execution_context->destroy();
    engine->destroy();
    runtime->destroy();

}

int main(){
    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}
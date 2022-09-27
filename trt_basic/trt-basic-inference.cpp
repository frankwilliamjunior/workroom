#include<iostream>
#include<fstream>
#include<vector>
#include<math.h>
#include"cuda_runtime.h"
#include"NvInfer.h"
#include"NvInferRuntime.h"

using namespace std;

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity,nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kINFO){
            printf("%d:%s\n",severity,msg);
        }
    }
}logger;

nvinfer1::Weights make_weights(float * ptr,int n){
    nvinfer1::Weights w;
    w.values = ptr;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    return w;
}

bool build_model(){
    TRTLogger logger;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0,2.0,0.5,0.1,0.2,0.5};
    float layer1_bias_values[]= {0.3,0.8};

    nvinfer1::ITensor* input = network->addInput("image",nvinfer1::DataType::kFLOAT,nvinfer1::Dims4(1,num_input,1,1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values,6);
    nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values,2);

    auto layer1 = network-> addFullyConnected(*input,num_output,layer1_weight,layer1_bias);
    auto prob = network->addActivation(*layer1->getOutput(0),nvinfer1::ActivationType::kSIGMOID);

    network->markOutput(*prob->getOutput(0));

    printf("Workspace size = %.2f MB\n",(1<<28) / 1024.0f/1024.0f);
    config->setMaxWorkspaceSize(1<<28);
    builder->setMaxBatchSize(1);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network,*config);

    if(engine == nullptr){
        printf("build engine failed.\n");
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

vector<unsigned char> load_file(const string& file){
    ifstream in(file,ios::in | ios::binary);
    if(!in.is_open())
    return {};

    in.seekg(0,ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if(length > 0){
        in.seekg(0,ios::beg);
        data.resize(length);

        in.read((char*)&data[0],length);

    }
    in.close();
    return data;
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

    float input_data_host[] = {1,2,3};
    float* input_data_device = nullptr;

    float output_data_host[2];
    float* output_data_device = nullptr;
    cudaMalloc(&input_data_device,sizeof(input_data_host));
    cudaMalloc(&output_data_device,sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device,input_data_host,sizeof(input_data_host),cudaMemcpyHostToDevice,stream);

    float* bindings[] = {input_data_device,output_data_device};

    bool success = execution_context->enqueueV2((void**)bindings,stream,nullptr);
    cudaMemcpyAsync(output_data_host,output_data_device,sizeof(output_data_host),cudaMemcpyHostToDevice,stream);
    cudaStreamSynchronize(stream);

    printf("output_data_host = %f,%f\n",output_data_host[0],output_data_host[1]);

    printf("Clean memory\n");

    cudaStreamDestroy(stream);
    execution_context->destroy();
    engine->destroy();
    runtime->destroy();

    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0,2.0,0.5,0.1,0.2,0.5};
    float layer1_bias_values[] = {0.3,0.8};

    printf("手动验证计算结果：\n");
    for(int io = 0; io < num_output;++io){
        float output_host = layer1_bias_values[io];
        for(int ii = 0; ii < num_input;++ii){
            output_host += layer1_weight_values[io * num_input + ii] * input_data_host[ii];
        }
        float prob = 1 / (1+exp(-output_host));
        printf("output_prob[%d] = %f\n",io,prob);

    }
}

int main(){
    if(!build_model()){
        return -166;
    }
    inference();
    return 0;
}
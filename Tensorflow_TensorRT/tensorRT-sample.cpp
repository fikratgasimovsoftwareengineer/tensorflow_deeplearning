#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <cuda_runtime_api.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <algorithm>
#include <numeric>


//declaration of Logger class, which will warn while building during run-time, in case of errors
//
class Logger:public nvinfer1::Ilogger{

    public:
	    void log(Severity severity, const char* msg) override{
	        if ((severity==kError) || (severity==kINTERNAL_ERROR)){
		    std::cout << msg << '\n';
		
		}
	    
	    }

} gLogger;


// struct for tensorrt 
// desttory tensorrt object if build goes wrong
struct TRTDestroy{

    template<class T>
    void operator() (T *obj ) const{
    
        if (obj){
	
	    obj->destroy();
	}
    
    }
}

template<class T>

using TRTUniquePtr = std::unique_ptr<T, TRTDestroy>

//get size of tensor
ssize_t getSizeByDims(const nvinfer1::Dims& dims){

    ssize_t size = 1;

    for(ssize_t i=1; i < dims.nbDims, ++i){
    
        size *= dims.d[i];
    }
    return size;
}


std::vector<std::string>getClassNames(const std::string& imagenet_classes){
    // read imaginet classes
    std::ifstream class_names(imagenet_classes);
  
    std::vector<std::string> class_labels;

    if (!class_names.good()){
        std::cerr<< "ERROR: can not read imagenet file\n";
	return class_labels;
    }

    std::string labels;

    while(std::getline(class_names,labels)){
    
        class_labels.push_back(labels);
    }

    return class_labels;


}


void preprocessingImage(const std::string& image_path, float* gpu_input, const nvinfer1::Dims& dims){

    //read input image
    cv::Mat input_frame;
    input_frame = cv::imread(image_path);

    cv::cuda::GpuMat gpu_frame;

    // upload image to gpu
    gpu_frame.upload(input_frame);

    auto input_width = dims.d[2];
    auto input_height = dims.d[1];

    auto channels = dims.d[0];

    auto input_size= cv::Size(input_width, input_height);

    //resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);

    //normalize
    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f /255.f);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

    // to tensor
    std::vector<cv::cuda::GpuMat> chw;
    for (ssize_t i=0; i< channels; ++i){
    
       chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));

    }
    cv::cuda::split(flt_image, chw);

}

// post processing stage
void postprocessinResults(float * gpu_output, const nvinfer1::Dims &dims, int batch_size){


    //get classnames
    auto classes = getClassNames("imagenet_classes.txt");

    //copy result from gpu to cpu
    std::vector<float>cpu_output (getSizeByDims(dims) + batch_size);

    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    //calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val){
		    
		    return std::exp(val);});

    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0, 0);

    // find top classes predicted by model
    std::vector<int>indices(getSizeByDims(dims) * batch_size);

    std::iota(indices.begin(), indices.end(),0);
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2){return cpu_output[i1] > cpu_output[i2];});

    //print results
    int i=0;
    while(cpu_output[indices[i]] / sum > 0.005){
    
        if (classes.size() > indices[i]){
	
	    std::cout << "class: " << classes[indices[i]] << "| ";
	
	}
	std::cout << "confidence : "<< 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i]<< "\n";

	++i;
    }
}

// initialize tensorrt engine and parse onnx model
void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
		TRTUniquePtr<nvinfer1::IExecutionContext>& context){

    //Builder takes gLogger
    TRTUniquePtr<nvinfer1::IBuilder> builder{nvinfer1::createInferBuilder(gLogger)};
    //Network definition for builder
    TRTUniquePtr<nvinfer1::INetworkDefinition>network{builder->createNetwork()};

    // Create Parser passing network cong and gLogger
    TRTUniquePtr<nvonnxparser::IParser> parser{nvonnxparser::createParser(*network, gLogger)};

    //Builder Config
    TRTUniquePtr<nvinfer1::IBuilderConfig> config{builder->createBuilderConfig()};

    // parse Onnx
    if (!parser->parseFromFile(model_path.c_str()), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)){
        std::cerr << "ERROR: could not parse model \n";
    	return -1;
    }

    // allow tensorrt to use 1 gb of gpu memory
    config->setMaxWorkspaceSize(1ULL << 30);

    //use fp16 if possible
    if(builder->platformHasFastFp16()){
    
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    //we have only one image in batch
    builder->setMaxBathSize(1);
    //generate TensortRT engine optimizer for target platform
    engine.reset(builder->buildEngineWithConfig(*network, *config));

    context.reset(engine->createExecutionContext());

}

int main(int argc, char** argv){


    if(argc<3){
        std::cerr<<"usage: " << argv[0]<< "model.onnx image.jpg\n";
	return -1;
    
    }

    std::string model_path(argv[1]);
    std::string image_path(argv[2]);

    int batch_size=1;


    //Initialize tensorrt rt and parse onnx model
    TRTUniquePtr<nvinfer1::ICudaEngine> engine{nullptr};
    TRTUniquePtr<nvinfer1::IExecutionContext>context{nullptr};

    parseOnnxModel(model_path, engine, context);

    // get sizes of input and output and allocate memory required for input data and for output data
    std::vector<nvinfer1::Dims> input_dims;
    std::vector<nvinfer1::Dims> output_dims;//one output

     // buffers for input and output data
    std::vector<void*>buffers(engine->getNbBindings());

    for(size_t i=0; i< engine->getNbBindings(); ++i){
    
    
        auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);

	cudaMalloc(&buffer[i, binding_size]);

	if(engine->bindingIsInput(i)){
	
	    input_dims.emplace_back(engine->getBindingDimensions(i));

	}
	else{
	    output_dims.emplace_back(engine->getBindingDimensions(i));
	}
    }

    if(input_dims.empty() || output_dims.empty()){
    
        std::cerr<< "Expect at least one input and one output for network\n";
	return -1;
    }

    // preprocess input image
    preprocessingImage(image_path, float(*)buffers[0], input_dims[0]);

    //inference
    context->enqueue(batch_size, buffers.data(), 0, nullptr);

    //postprocess results
    postprocessinResults((float*)buffers[1], output_dims[0], batch_size);

    for(void* buf : buffers){
    
        cudaFree(buf);
    }    

    return 0;



}


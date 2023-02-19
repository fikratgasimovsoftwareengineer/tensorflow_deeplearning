void preprocessImage(const std::string& img_path, float* gpu_input, const nvinfer1::Dims& dims){

    cv::Mat frame;
    frame = cv::imread(image_path);

    if (frame.empty()){
    
        std::cerr << "Input Image : " << image_path << "Load Failed \n";

	return -1;
    
    }

    cv::cuda::GpuMat gpu_frame;

    gpu_frame.upload(frame)

}

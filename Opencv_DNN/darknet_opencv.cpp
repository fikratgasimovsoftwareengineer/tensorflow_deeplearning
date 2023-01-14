#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

//NameSpaces
using namespace std;
using namespace cv;
using namespace cv::dnn;


const float input_width = 640.0;
const float input_height = 640.0;
const float score_threshold = 0.5;
const float nms_threshold = 0.45;
const float confidence_thresh = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);

void draw_label(Mat& input_image, string label_name, int left, int top){

    // Display label at the top of bounding box
    //
    int baseline;
    cv::Size label_size = cv::getTextSize(label_name,FONT_FACE, FONT_SCALE,THICKNESS,&baseline);

    top = cv::max(top, label_size.height);
    // top left corner
    cv::Point tlc = cv::Point(left, top);

    // Bottom right corner
    cv::Point brc = cv::Point(left+label_size.width, top+label_size.height+baseline);

    // draw rectangle
    cv::rectangle(input_image, tlc, brc, (0,0,0), cv::FILLED);

    // put the text on the black rectangle
    //
    putText(input_image, label_name, Point(left, top+label_size.height), FONT_FACE, FONT_SCALE, (0,255,255), THICKNESS);

}


// return vector image
vector<Mat>pre_processing(Mat& input_image, Net& net){
   
   
   Mat blob;

   blobFromImage(input_image, blob, 1.0/255., cv::Size(input_width, input_height), Scalar(), true, false);

   net.setInput(blob);

   //forward propagete
   //
   vector<Mat>outputs;

   net.forward(outputs, net.getUnconnectedOutLayersNames());

   return outputs;
}



Mat post_process(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name){

    vector<int>class_ids;

    vector<float>confidences;

    vector<Rect> boxes;

    //resizing factor
    //
    float x_factor = input_image.cols / input_width;
    float y_factor = input_image.rows / input_height;

    float *data = (float *) outputs[0].data;

    const int dimensions = 85;

    const int rows = 25200;

    for (int i=0; i < rows; ++i){
    
        float confidence = data[4];
	// discard bad detections and continue
	if (confidence >= confidence_thresh){
	
	    float *classes_scores = data+5;
	    // create matrix and store class score
	    Mat scores(1, class_name.size(), CV_32FC1, classes_scores);

	    Point class_id;

	    double max_class_scores;
	   
	    minMaxLoc(scores, 0, &max_class_scores, 0, &class_id);

	    if (max_class_scores > score_threshold){
	    
	        //Store ID and confidence in pre-defined respective 
		//
		confidences.push_back(confidence);
		class_ids.push_back(class_id.x);

		// center points
		float cx = data[0];
		float cy = data[1];

		//Box dimension
		float w = data[2];
		float h = data[3];

		// Bounding Box coordinates
		int left = int((cx - w/2)*x_factor);
		int top = int ((cy - h/2)*y_factor);
		int width = int(w * x_factor);
		int height = int(h * y_factor);

		//stores good detections in boxes vectors
		boxes.push_back(Rect(left, top, width, height));

	    }
	}
	data+=85;
    }



    // Perform Non Max Suppression and draw predictions
    //
    vector<int> indices;
    NMSBoxes(boxes, confidences, score_threshold, nms_threshold, indices);

    for(int i =0; i< indices.size(); i++){
    
        int idx = indices[i];
	Rect box = boxes[idx];

    	int left = box.x;
	int top = box.y;
	int width = box.width;
	int height = box.height;

	// draw bounding box
	rectangle(input_image, Point(left, top), Point(left+width, top+height), (255, 178, 50), 3*THICKNESS);
        
	// get label for class and its confidence
	string label = format("%.2f", confidences[idx]);
	label = class_name[class_ids[idx]] + ":" + label;

	//draw labels;
	draw_label(input_image, label, left, top);	

    }

    return input_image;

}

int main(int argc, char** argv){

   // load class list
    vector<string> class_list;
    
    //read file
    ifstream ifs("coco.names");

    string line;

    while(getline(ifs, line)){
    
    
        class_list.push_back(line);
    
    }

   


    // Load image
    Mat frame;
    Mat cloned_frame;
 
    frame = imread("sample.jpg");

    //Load model
    Net net;
   
    net = readNet("models/yolov5s.onnx");

    vector <Mat> detections;
   
    // call pre_processing
    detections = pre_processing(frame, net);
    cloned_frame = frame.clone();

    // poss processing call
    Mat img = post_process(cloned_frame, detections, class_list);


    
    // Put efficiency information.
    // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    //
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net.getPerfProfile(layersTimes) / freq;

    string label  = format("Inference time: %.2f ms", t);

    putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, (0, 255, 255));

    imshow("Detections With YOLOV5:", img );
    waitKey(0);

    return 0;    


}


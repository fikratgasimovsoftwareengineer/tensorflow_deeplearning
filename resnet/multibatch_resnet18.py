import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import cv2
import numpy as np
import os
from PIL import Image

import matplotlib.pyplot as plt

class TRTInference:
    
    
    # specify engine file path and input and output shape
    def __init__(self, engine_file_path, input_shape, output_shape, class_labels_file):
        self.logger = trt.Logger(trt.Logger.WARNING)

        ## load engine here
        self.engine = self.load_engine( engine_file_path)

        # craete context
        self.context = self.engine.create_execution_context()

        # input shape
        self.input_shape = input_shape
        
        # output shape
        self.output_shape = output_shape

        with open(class_labels_file, 'r') as class_read:
            self.class_labels = [line.strip() for line in class_read.readlines()]

    def load_engine(self, engine_file_path):
        with open(engine_file_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine_desentriliazed = runtime.deserialize_cuda_engine(f.read())

            return engine_desentriliazed
        
    def preprocess_image(self, images_path):

        imgs_list = []

        for image_path in os.listdir(images_path):

            full_path = os.path.join(images_path, image_path)

            if full_path.endswith('.jpg') or full_path.endswith('.jpeg') or full_path.endswith('png'):

                img = Image.open(full_path).convert('RGB')

                img = img.resize((self.input_shape[2], self.input_shape[3]), Image.NEAREST)

                img_np = np.array(img).astype(np.float32) / 255.0

                img_np = img_np.transpose((2,0,1))
                
                imgs_list.append(img_np)

        imgs_np = np.stack(imgs_list, axis=0)
     

        return imgs_np
        
   
    
    def postprocess_img(self,outputs):

        classes = []
        for output in outputs:

            class_idx = np.argmax(output)
            classes.append(self.class_labels[class_idx])

        print("Class Detected: ", classes)

        return classes


    def inference_detection(self,image_path):

        inputs = self.preprocess_image(image_path)
        
        inputs = np.ascontiguousarray(inputs)

        ## changed
        #outputs = np.empty(self.output_shape, dtype=np.float32)
      
        outputs = np.empty((len(inputs),) + self.output_shape[1:], dtype=np.float32)
       

        d_inputs = cuda.mem_alloc(1 * inputs.nbytes)

        d_outpus = cuda.mem_alloc(1 * outputs.nbytes)

        bindings = [d_inputs ,d_outpus]

        cuda.Context.synchronize()
        cuda.memcpy_htod(d_inputs, inputs)

        self.context.execute_v2(bindings=bindings)

        cuda.Context.synchronize()
        # copy output back to host
        cuda.memcpy_dtoh(outputs, d_outpus) 

      

        result = self.postprocess_img(outputs)        

        d_inputs.free()

        d_outpus.free()

      #  for images in os.listdir(image_path):
          

       #    self.display_recognized_image(images)

        return result   
    
    '''def display_recognized_image(self, image_path):
      
        count = 0
      
        image = Image.open(image_path)
        
        plt.title('Recognized Image')

        plt.savefig('image_detected_' + count)
        count +=1

        return image'''


engine_file_path ='/deeplearning/resnet/resnet_engine.trt'

# Load the TensorRT engine
input_shape = (10,3,224, 224)

output_shape = (10, 1000)

image_path = '/deeplearning/resnet/images'

path_to_class = "/deeplearning/resnet/imagenet_classes.txt"

inference = TRTInference(engine_file_path, input_shape, output_shape, path_to_class)

class_name = inference.inference_detection(image_path)
print(class_name)
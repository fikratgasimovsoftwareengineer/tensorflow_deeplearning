import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from PIL import Image

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(filename):
    with open(filename, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def preprocess_image(image_path):
    # Load image and resize to expected shape (3x224x224)
    img = Image.open(image_path)
    img = img.resize((224, 224), Image.NEAREST)
    
    # Convert to Numpy array and normalize
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # HWC -> CHW
    img_np = img_np.transpose((2, 0, 1))
    
    # Add batch dimension
    img_np = np.expand_dims(img_np, axis=0)
    
    return img_np


def load_classes(class_path):
    with open(class_path, 'r') as read_file:
        read_file = [line.strip() for line in read_file.readlines()]
    return read_file

def postprocess(outputs):

    class_idx = outputs[0].argmax()
    
    # Add any post processing here
    return class_idx

# Load the TensorRT engine
engine_file_path = 'resnet_engine.trt'
engine = load_engine(engine_file_path)


path_to_class = "imagenet_classes.txt"

# Create an execution context
context = engine.create_execution_context()

# Load input data and allocate output buffers
image_path = '/deeplearning/resnet/cane.png'
inputs = preprocess_image(image_path)
outputs = np.empty(engine.get_binding_shape(1), dtype=np.float32)

# Allocate device memory
d_inputs = cuda.mem_alloc(1 * inputs.nbytes)
d_outputs = cuda.mem_alloc(1 * outputs.nbytes)

bindings = [int(d_inputs), int(d_outputs)]

# Transfer input data to device
inputs = np.ascontiguousarray(inputs)  # Ensure the array is contiguous

# Transfer input data to device
cuda.memcpy_htod(d_inputs, inputs)

# Execute model
context.execute(1, bindings)

# Transfer predictions back
cuda.memcpy_dtoh(outputs, d_outputs)

# Postprocess the output
result = postprocess(outputs)

print(result)
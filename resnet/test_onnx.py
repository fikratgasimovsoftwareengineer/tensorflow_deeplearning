
import onnx

def apply(transform, infile, outfile):
    model = onnx.load(infile)
    transform(model)
    onnx.save(model, outfile)
    
def change_input_dim(model):
    # Use some symbolic name not used for any other dimension
    sym_batch_dim = '1'
    # or an actual value
    actual_batch_dim = 10

    # The following code changes the first dimension of every input to be batch-dim
    # Modify as appropriate ... note that this requires all inputs to
    # have the same batch_dim 
    inputs = model.graph.input
    for input in inputs:
        # Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
        # Add checks as needed.
        dim1 = input.type.tensor_type.shape.dim[0]
        # update dim to be a symbolic value
        dim1.dim_param = sym_batch_dim
        print(dim1.dim_param)
        # or update it to be an actual value:
        dim1.dim_value = actual_batch_dim
        print(dim1.dim_value)

    onnx.save(model, '/deeplearning/resnet/resnet18_test.onnx')
# load the model file
model_path = '/deeplearning/resnet/resnet18.onnx'
model = onnx.load(model_path)
change_input_dim(model)


import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)

example = torch.rand(1, 3, 224, 224)

torch.onnx.export(model, example, "resnet18.onnx")
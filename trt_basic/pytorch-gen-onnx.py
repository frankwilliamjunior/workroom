import onnx
import torch
import numpy as np
import onnx.helper as helper
import os

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        
        self.conv = nn.Conv2d(1,1,3,padding=1)
        self.relu = nn.ReLU()
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)

    def forward(self,x)
        x = self.conv(x)
        x = self.relu(x)
        return x
#这个包对应opset11的导出代码，如果想修改导出的细节，可以在这里修改代码
#import torch.onnx.symbolic_opset11
print("对应opset文件夹代码在这里：",os.path.dirname(torch.onnx.__file__))

model = Model()
dummy = torch.zeros(1,1,3,3)
torch.onnx.export(
    model,
    (dummy,),
    "demo.onnx",
    verbose = True,
    input_names = ["image"],
    output_names = ["output"],
    opset_version = 11,
    dynamic_axes = {
        "image":{0:"batch",2:"height",3:"width"},
        "output":{0:"batch",2:"height",3:"width"},
    }

)
print("Done!.")
    

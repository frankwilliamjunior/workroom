import onnx
import onnx.helper as helper
import numpy as np

model = onnx.load("demo.change.onnx")

print("=================node信息")

print(model)

conv_weight = model.graph.initializer[0]
conv_bias = model.graph.initializer[1]

print(f"===================={conv_weight.name}===============")
print(conv_weight.name,np.frombuffer(conv_weight.raw_data,dtype=np.float32))

print(f"====================={conv_bias.name}==================")
print(conv_bias.name,np.frombuffer(conv_bias.raw_data,dtype=np.float32))

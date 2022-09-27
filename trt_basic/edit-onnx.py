import onnx
import onnx.helper as helper
import numpy as np
model = onnx.load("demo.onnx")

conv_weight = model.graph.initializer[0]
conv_bias = model.graph.initializer[1]

conv_weight.raw_data = np.arange(9,dtype = np.float32).tobytes()

onnx.save_model(model,"demo.change.onnx")
print("Done.!")
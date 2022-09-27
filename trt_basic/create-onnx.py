import onnx
import numpy as np
import onnx.helper as helper
import os

nodes = [
    helper.make_node(
        name = "Conv_0",
        op_type = "Conv",
        inputs = ["image","conv.weight","conv.bias"],
        outputs = ["3"],
        pads = [1,1,1,1],
        group = 1,
        dilations = [1,1],
        kernel_shape = [3,3],
        strides = [1,1]
    ),
    helper.make_node(
        name = "ReLU_1",
        op_type = "Relu",
        inputs = ["3"],
        outputs = ["output"]
    )
]

initializer = [
    helper.make_tensor(
        name = "conv.weight",
        data_type = helper.TensorProto.DataType.FLOAT,
        dims = [1,1,3,3],
        vals = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],dtype=np.float32).tobytes(),
        raw=True
    ),
    helper.make_tensor(
        name = "conv.bias",
        data_type=helper.TensorProto.DataType.FLOAT,
        dims = [1],
        vals = np.array([0.0],dtype=np.float32).tobytes(),
        raw = True
    )
]

inputs = [
    helper.make_value_info(
        name = "image",
        type_proto = helper.make_tensor_type_proto(
            elem_type = helper.TensorProto.DataType.FLOAT,
            shape = ["batch",1,3,3]
        )
    )
]

outputs = [
    helper.make_value_info(
        name = "output",
        type_proto = helper.make_tensor_type_proto(
            elem_type = helper.TensorProto.DataType.FLOAT,
            shape = ["batch",1,3,3]
        )
    )
]

graph = helper.make_graph(
    name = "mymodel",
    inputs = inputs,
    outputs = outputs,
    nodes = nodes,
    initializer = initializer
)

opset = [
    helper.make_operatorsetid("ai.onnx",11)
]

model = helper.make_model(graph,opset_imports = opset,producer_name = "pytorch",producer_version="1.9")
onnx.save_model(model,"my.onnx")

print(model)
print("Done.!")
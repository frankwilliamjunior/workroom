import torch

class siluIMPL(torch.autograd.Function):
    
    # 设定onnx导出参数
    # 自定义实现的op 只需要添加静态方法symbolic即可，除了用g代替forward中的context外 其他参数需相同
    # 对于附加属性 以“名称_简写” 方式定义 类型简写 参考torch/onnx/symbolic parse_arg
    @staticmethod
    def symbolic(graph,input,bias):
        return graph.op("Plugin",input,bias,
        name_s = "Swish",info_s = json.dumps({
            "size" :555
        }))

    @staticmethod
    def forward(context,input,bias):
        context.save_for_backward(input)
        return input * torch.sigmoid(input) + bias
    # dL/dx = dL/dy * dy/dx 链式求导法则 L 为loss
    # 此处应该返回dL/dx   grad_output即为dL/dy
    def backward(context,grad_output):
        input = content.saved_tensors[0]
        sigmoid_input = torch.sigmoid(input)
        # 返回 dL/dx  dL/d bias   = dL/dy * dy/dx    dL/dy * dy/d bias
        return grad_output*(sigmoid_input*(1+input*(1-sigmoid_input))),grad_output

class silu(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.parameter.Parameter(torch.arange(n).float())
    
    def forward(self,input):
        return siluIMPL.apply(input,self.param)

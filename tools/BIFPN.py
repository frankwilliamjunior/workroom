import torch

# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

 
# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    # 如果需导出onnx则必须实现symbolic方法
    # symbolic的参数除用g:graph 代替ctx外 其他参数必须完全一样
    @staticmethod
    def symbolic(g,i):
        pass

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result
 
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
 
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class Concat_BIFPN(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1, c2):
        super(Concat_BIFPN, self).__init__()
        self.relu = nn.ReLU()
        self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.swish = MemoryEfficientSwish()
 
    def forward(self, x):
        outs = self._forward(x)         # whats the problem with you?
        return outs
 
    def _forward(self, x):
        if len(x) == 2:
            # w = self.relu(self.w1)
            w = self.w1
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            # Connections for P6_0 and P7_0 to P6_1 respectively
            x = self.conv(self.swish(weight[0] * x[0] + weight[1] * x[1]))
        elif len(x) == 3:
            # w = self.relu(self.w2)
            w = self.w2
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            x = self.conv(self.swish(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
        return x
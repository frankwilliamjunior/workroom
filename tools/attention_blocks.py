import torch
import torch.nn as nn


# 通道注意力机制 squeeze and excitation network
class SElayer(nn.Module):
    def __init__(self,in_channels,reduction = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_channels,in_channels//reduction),
                                nn.ReLU(inplace = True),
                                nn.Linear(in_channels//16,in_channels),
                                nn.Sigmoid(inplace = True))
    
    def forward(self,input):
        b,c,h,w = input.shape
        out = self.avg_pool(input).view(b,c)
        out = self.mlp(out).view(b,c,1,1)
        return out.expand_as(input) * input

# channel attention module 通道注意力模块 类似SEnet 不过采用了maxpooling 和avgpooling 两种pooling方式
class CAM(nn.Module):
    def __init__(self,in_channels,reduction = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.maxpooling = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            # 此处采用1*1卷积代替Linear层  避免维度操作
            nn.Conv2d(in_channels,in_channels//reduction,kernel_size = 1,stride = 1),
            # full connect 后输出会变得较大，需接activation层 将数值拉回到较小范围
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels//reduction,in_channels,kernel_size =1,stride =1))

        self.act = nn.Sigmoid(inplace=True)
    def forward(self,input):
        # 此处注意 avg 和max 分支通过的是同一个mlp层  至于为什么 可能是强迫模型学习各通道的重要性？
        avg_out = self.mlp(self.avg_pool(input))
        max_out = self.mlp(self.maxpooling(input))
        attention = self.act(avg_out + avg_out)
        return input*attention

# spatial attention module 空间注意力模块
class SAM(nn.Module):
    def __init__(self,kernel_size = 7)
        super().__init__()
        # dont know why 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        self.conv = nn.Conv2d(2,1,kernel_size=kernel_size,stride=1,padding = kernel_size//2)
        self.sigmoid = nn.Sigmoid(inplace=True)
    def forward(self,input):
        avg_out = torch.mean(input,dim=1,keepdim=True)
        max_out,_ = torch.max(input,dim=1,keepdim=True)         # torch.max 会返回indice
        spatial_out = torch.cat([avg_out,max_out],dim = 1)
        spatial_out = self.sigmoid(self.conv(spatial_out))
        return input * spatial_out

# convolutional block attention module
class CBAM(nn.Module):
    def __init__(self,in_channels,reduction = 8,kernel_size = 7):
        super().__init__()
        self.cam = CAM(in_channels,reduction = reduction)
        self.sam = SAM(in_channels,kernel_size = kernel_size)
    
    def forward(self,input):
        output = self.sam(self.cam(input))
        return output

# efficient channel attention block 轻量通道注意力机制
class ECAlayer(nn.Module):
    """Constructs a ECA module.
        Args:
            channel: Number of channels of the input feature map
            k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_channels, k_size=5):
        super().__init__()
        # 图中GAP = global average pooling 即 将hw pooling压缩成一个值
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # kernel_size 根据in_channels数量选择  (k_size - 1)//2 等同于k_size //2 当k_size为奇数
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self,input):
        b, c, h, w = input.size()
 
        # feature descriptor on the global spatial information
        # b,c,h,w -> b,c,1,1
        y = self.avg_pool(input)
 
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
 
        # Multi-scale information fusion
        y = self.sigmoid(y)
 
        return input * y.expand_as(input)


    def __init__(self):
        super().__init__()

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

# coordinate attention for efficient mobile network
class CAlayer(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CAlayer,self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

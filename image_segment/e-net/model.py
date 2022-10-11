import torch
import torch.nn as nn

class InitialBlock(nn.Module):
    def __init__(self,in_channels,out_channels,bias=True,relu=True):
        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        self.main_branch = nn.Conv2d(
            in_channels,out_channels-3,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=bias)

        self.ext_branch = nn.MaxPool2d(3,stride=2,padding=1)

        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.out_activation = activation()

    def forward(self,x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        out = torch.cat((main,ext),1)

        out = self.batch_norm(out)
        return self.out_activation(out)

class RegularBottleneck(nn.Module):
    def __init__(self,channels,internal_ratio=4,
        kernel_size=3,padding=0,
        dilation=1,
        asymmetric=False,
        dropout_prob=0,
        bias=False,
        relu=True):
        super().__init__()

        if internal_ratio <= 1 or internal_ration > channels:
            raise RuntimeError("Value out of range. Expected value in the"
            "interval [1,{0}],got internal_scale={1}."
            .format(channels,internal_ration))

        internal_channels = channels // internal_ration

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels,internal_channels,kernel_size=1,stride=1,bias=bias),
            nn.BatchNorm2d(internal_channels),
            activation()
        )
        
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size,1),
                    stride=1,
                    padding=(padding,0),
                    dilation=dilation,
                    bias=bias
                ),nn.BatchNorm2d(internal_channels),
                activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1,kernel_size),
                    stride=1,
                    padding=(0,padding),
                    dilation=dilation,
                    bias=bias),
                    nn.BatchNorm2d(internal_channels),activation())
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding = padding,
                    dilation = dilation,
                    bias=bias
                ),nn.BatchNorm2d(internal_channels),activation())
            self.ext_conv3 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    channels,kernel_size=1,stride=1,bias=bias
                ),nn.BatchNorm2d(channels),activation())
            
            self.ext_regul = nn.Dropout2d(p=dropout_prob)

            self.out_activation = activation()
    
    def forward(self,x):
        main = x
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        out = main + ext
        return self.out_activation(out)

class DownsamplingBottleneck(nn.Module)
    def __init__(self,
        in_channels,
        out_channels,
        internal_ratio=4,
        return_indices=False,
        dropout_prob=0,
        bias=False,
        relu=True):
        super().__init__()

        self.return_indices = return_indices

        if internal_ration <= 1 or internal_ration > in_channels:
            raise RuntimeError("Value out of range.Expected value in the"
            "interval [1,{0}],got internal_scale={1}."
            .format(in_channels,internal_ratio))
        
        internal_channels = in_channels // internal_ration

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        
        self.main_max1 = nn.MaxPool2d(2,stride=2,return_indices=return_indices)

        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias),nn.BatchNorm2d(internal_channels),activation())

        self.ext_conv2 = nn.sequential(
            nn.Conv2d(internal_channels,
            internal_channels,
            kernel_size=3,
            stride=1,padding=1,bias=bias),
            nn.BatchNorm2d(internal_channels),activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        self.out_activation = activation()
    
    def forward(self,x):
        if self.return_indices:
            main,max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        n,ch_ext,h,w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n,ch_ext - ch_main,h,w)

        if main.is_cuda:
            padding = padding.cuda()
        
        main = torch.cat((main,padding),1)

        out = main + ext

        return self.out_activation(out),max_indices

class UpsamplingBottleneck(nn.Module):
    def __init__(self,in_channels,
        out_channels,internal_ratio=4,
        dropout_prob=0,
        bias=False,
        relu=True)
        super().__init__()
    
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. expected value in the "
                            "inerval [1,{0}], got internal_scale={1}."
                            .format(in_channels,internal_ratio))
        internal_channels = in_channels // internal_ration

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_conv1 = nn.sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=bias),
            nn.BatchNorm2d(out_channels))

        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        self.ext_conv1 = nn.sequential(
            nn.Conv2d(
                in_channels,internal_channels,kernel_size=1,bias= bias),
            nn.BatchNorm2d(internal_channels),activation())

        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size= 2,
            stride=2,
            bias=bias)
        
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)

        self.ext_tconv1_activation = activation()

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        self.out_activation = activation()

    def forward(self,x,max_indices,output_size):
        main = self.main_conv1(x)
        main = self.main_unpool1(
            main,max_indices,output_size = output_size)
        
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext,output_size=output_size)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regul(ext)

        out = main + ext

        return self.out_activation(out)
    

class ENet(nn.Module)
    def __init__(self,num_classes,encoder_relu=False,decoder_relu=True):
    super().__init__()
    self.initial_block = InitialBlock(3,16,relu=encoder_relu)

    self.downsample1_0 = DownsamplingBottleneck(16,64,return_indices=True,dropout_prob=0.01,relu=encode_relu)

    self.regular1_1 = RegularBottleneck(2
    64,padding=1,dropout_prob=0.01,relu=encoder_relu)

    def forward(self)
        pass
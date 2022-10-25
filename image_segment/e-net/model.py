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
        ext = self.ext_conv1(x)         # 1*1 降维
        ext = self.ext_conv2(ext)       # 3*3 提取特征 维度不变
        ext = self.ext_conv3(ext)       # 1*1 升维
        ext = self.ext_regul(ext)       # dropout

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
        
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), nn.BatchNorm2d(out_channels), activation())

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        self.out_activation = activation()
    
    def forward(self,x):
        if self.return_indices:
            main,max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)        # kernel_size = 2 maxpooling

        ext = self.ext_conv1(x)             # 2*2 stride 2 降维
        ext = self.ext_conv2(ext)           # 3*3 提取特征 维度不变
        ext = self.ext_conv3(ext)           # 1*1 升维
        ext = self.ext_regul(ext)           # dropout

        n,ch_ext,h,w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n,ch_ext - ch_main,h,w)

        if main.is_cuda:
            padding = padding.cuda()
        
        main = torch.cat((main,padding),1)    # main 分支缺少的通道 补0 padding

        out = main + ext                      # 由于是相加

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

        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels))

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        self.out_activation = activation()

    def forward(self,x,max_indices,output_size):
        main = self.main_conv1(x)                           # 1*1
        main = self.main_unpool1(
            main,max_indices,output_size = output_size)     # maxunpooling 2*2 上采样
        
        ext = self.ext_conv1(x)                             # 1*1 降维
        ext = self.ext_tconv1(ext,output_size=output_size)  # 2*2 stride 2 transposedconv 上采样
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)                           # 1*1 升维
        ext = self.ext_regul(ext)                           # dropout

        out = main + ext

        return self.out_activation(out)
    

class ENet(nn.Module):
    """Generate the ENet model.

        Keyword arguments:
        - num_classes (int): the number of classes to segment.
        - encoder_relu (bool, optional): When ``True`` ReLU is used as the
        activation function in the encoder blocks/layers; otherwise, PReLU
        is used. Default: False.
        - decoder_relu (bool, optional): When ``True`` ReLU is used as the
        activation function in the decoder blocks/layers; otherwise, PReLU
        is used. Default: True.

    """

    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()

        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            16,
            64,
            return_indices=True,
            dropout_prob=0.01,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            64,
            128,
            return_indices=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(
            128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(
            64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(
            16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(
            16,
            num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Initial block
        input_size = x.size()
        x = self.initial_block(x)

        # Stage 1 - Encoder
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular5_1(x)
        x = self.transposed_conv(x, output_size=input_size)
        x = self.sigmoid(x)

        return x

    def get_param_groups(self):

        param_groups = [[]] # 
        
        for param in list(self.parameters()):

            param_groups[0].append(param)

        return param_groups
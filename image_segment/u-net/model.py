import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None)
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,out_channels,3,1,1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,input)
        output = self.double_conv(input)
        return output

class Down(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels,mid_channels)
        )

    def forward(self,input):
        output = self.maxpool_conv(input)

class Up(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True,mid_channels=None):
        super(Up,self).__init__()
        if bilinear:
            self.upsample_conv = nn.Sequential(
            nn.Upsample(scale_factor=2,mode= "bilinear",align_corners=True),
            DoubleConv(in_channels,out_channels,mid_channels)
            )
        else:
            self.upsample_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels,in_channels//2,kernel_size=2,stride=2),
                DoubleConv(in_channels,out_channels,mid_channels))

    def forward(self,x1,x2):
        """x1为低分辨率特征图
           x2为高分辨率特征图
        """
        # NCHW
        x1 = self.upsample_conv(x1)
        diffx = x2.size()[3] - x1.size()[3]
        diffy = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1,[diffx//2,diffx - diffx//2,
                        diffy//2,diffy - diffy//2])
        output = torch.cat([x1,x2],dim=1)

        return output

class UNet(nn.Module):
    def __init__(self,n_channels,n_classes,bilinear=False):
        super(UNet,self).__init__()
        self.in_layer = DoubleConv(n_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        self.down3 = Down(256,512)
        n = 2 if bilinear else 1
        self.down4 = Down(512,1024//n)
        self.up1 = Up(1024,512//n,bilinear)
        self.up2 = Up(512,256//n,bilinear)
        self.up3 = Up(256,128//n,bilinear)
        self.up4 = Up(128,64,bilinear)
        self.out_layer = nn.Conv2d(64,n_classes,1,1,0)

    def forward(self,input):
        x1 = self.in_layer(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y = self.up1(x5,x4)
        y = self.up2(y,x3)
        y = self.up3(y,x2)
        y = self.up4(y,x1)
        output = self.out_layer(y)
        return output

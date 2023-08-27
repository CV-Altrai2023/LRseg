
from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial
import math
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        # 320, 320, 12 => 320, 320, 64
        return self.conv(
            # 640, 640, 3 => 320, 320, 12
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2]
                ], 1
            )
        )
class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4
        super(FCNHead, self).__init__(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

class BSConvU(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros", with_bn=False, bn_kwargs=None,norm = True):
        super().__init__()

        # check arguments
        if bn_kwargs is None:
            bn_kwargs = {}

        # pointwise
        self.add_module("pw", torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=False,
        ))

        # batchnorm
        if with_bn:
            self.add_module("bn", torch.nn.BatchNorm2d(num_features=out_channels, **bn_kwargs))

        # depthwise
        self.add_module("dw", torch.nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=out_channels,
                bias=bias,
                padding_mode=padding_mode,
        ))
        if norm:
            self.add_module("bn", torch.nn.BatchNorm2d(num_features=out_channels))
            self.add_module("active", nn.ReLU(inplace=True))

class BSConvU_res(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode="zeros", with_bn=False, bn_kwargs=None,norm = True):
        super(BSConvU_res,self).__init__()
        self.bsconv = BSConvU(in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=True, padding_mode="zeros", with_bn=False, bn_kwargs=None,norm = True)
        # self.bsconv = BSConvS(in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=True, padding_mode="zeros", p=0.25, min_mid_channels=4, with_bn=False, bn_kwargs=None,norm=True)
        self.inchannel = in_channels
        self.outchannel = out_channels
    def forward(self, x):
        if self.inchannel==self.outchannel:
            out = self.bsconv(x)+x
        else:
            out = self.bsconv(x)
        return out

def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )
class _PyramidPooling(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
class _PSPHead(nn.Module):
    def __init__(self, in_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_PSPHead, self).__init__()
        self.psp = _PyramidPooling(in_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)


        out_channels = in_channels//4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Conv2d(out_channels, nclass, 1)

    def forward(self, x):
        x = self.psp(x)
        feature_psp = x
        x = self.block(x)
        feature = x
        x = self.classifier(x)
        return x, feature, feature_psp

class split_space(nn.Module):
    def __init__(self):  # ch_in, ch_out, kernel, stride, padding, groups
        super(split_space, self).__init__()
        # self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):
        # 320, 320, 12 => 320, 320, 64
        return torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2]
                ], 1)
# The SC-FFM
class CA_Fusion(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(CA_Fusion, self).__init__()
        self.aten1 = nn.Sequential(
            BSConvU(256, 256, 1),
            BSConvU(256, 256, 1),
        )
        self.avgpool_add_3 = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_d, x_s):
        out_avg = self.avgpool_add_3(torch.cat([x_s, x_d], dim=1))
        out_avg = torch.sigmoid(self.aten1(out_avg))
        x_d_aten, out_aten = out_avg.chunk(2, 1)
        out_finnal = x_s * (out_aten.expand_as(x_s)) \
              + x_d * (x_d_aten.expand_as(x_d))

        return out_finnal

class The_model(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = BSConvU_res):
        super(The_model, self).__init__()

        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.stem = nn.Sequential(
            Focus(3,24),
            BSConvU_res(24, 24, 3, 1, 1),
            BSConvU_res(24, 24, 3, 1, 1)
        )
        self.stage1: nn.Sequential
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential

        input_channels = output_channels
        stage_names = ["stage{}".format(i) for i in [1, 2, 3]]

        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 3, 2, 1)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 3, 1, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        self.fusion1 = CA_Fusion(96, 48)
        self.fusion2 = CA_Fusion(96, 48)
        self.psphead = _PSPHead(128, num_classes)
        self.context_path = nn.Sequential(
            BSConvU(192, 128, 1)
        )

        self.split_space = split_space()
        self.sp = nn.Sequential(
            BSConvU(96, 128, 1)
        )
        self.sp_128 = BSConvU(192, 128, 1)

    def forward(self, x: Tensor) -> Tensor:
        # if input size is (3, 512,512)
        x = self.stem(x) # output size(24,256,256)
        x1 = self.stage1(x)# output size(48,128,128)
        x2 = self.stage2(x1)# output size(96,64,64)
        x3 = self.stage3(x2)# output size(192,32,32)
        # channels: 192-->128, size: (32,32)-->(64,64)
        x3 = self.context_path(x3)
        x3 = F.interpolate(x3, size=x2.size()[2:], mode='bilinear', align_corners=True)
        # channels: 192-->128
        sp_out = self.sp(x2)
        # In interval sampling, the length and width must be even.
        # Then, channels:48-->48*4-->128
        _,_,h,w = x1.size()
        if h%2==1: h+=1
        if w%2==1: w+=1
        x1_192 = F.interpolate(x1, (h,w), mode='bilinear', align_corners=True)
        x1_192 = self.split_space(x1_192)
        x1_128 = self.sp_128(x1_192)
        # Self-correcting decoder
        # SC-FFM and cascade structure fusion features
        feature1 = self.fusion1(sp_out, x3)
        feature2 = self.fusion2(feature1+x1_128, x3)
        # Segmentation head of PPM and FCNhead
        out,_,_ = self.psphead(feature2)

        return [out]



def Our_model(num_classes=1000):

    model = The_model(stages_repeats=[3, 6, 8],
                         stages_out_channels=[24, 48, 96, 192],
                         num_classes=num_classes)

    return model
if __name__ == '__main__':
    device = 'cpu'
    my_model = Our_model(num_classes=2)
    my_model.eval()
    from thop import profile
    from thop import clever_format
    input=torch.randn(1,3,512,512)
    print(my_model)
    input = input.to(device)
    import time
    t1 = time.time()
    out = my_model(input)
    print(time.time()-t1)
    flops, params = profile(my_model, inputs=(input,))
    print(flops, params)
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params) # 1.468G 781.778K

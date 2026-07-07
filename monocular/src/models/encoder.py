import torch
import torch.nn as nn
import numpy as np
from timm.layers import DropPath


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, padding_mode='zeros', act=nn.ReLU,
                 last=True, drop_path=0.0):
        super().__init__()
        bias = False
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        self.conv1 = Conv3x3(inplanes, planes, stride, padding_mode=padding_mode, bias=bias)
        self.bn1 = norm_layer(planes)
        self.relu1 = act()
        self.conv2 = Conv3x3(planes, planes, padding_mode=padding_mode, bias=bias)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if last:
            self.relu2 = act()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.drop_path(out) + identity
        if self.last:
            out = self.relu2(out)
        return out


class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1, padding_mode='zeros',
                 act=nn.ReLU, stride=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False, padding_mode=padding_mode)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=True, padding_mode=padding_mode)

        self.conv = nn.Sequential()
        self.conv.add_module('conv', conv)

        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))

        self.conv.add_module('relu', act())

    def forward(self, x):
        out = self.conv(x)
        return out


def Conv1x1(in_planes, out_planes, stride=1, bias=False, groups=1, dilation=1, padding_mode='zeros'):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding_mode='zeros', bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, padding_mode=padding_mode, groups=groups, bias=bias, dilation=dilation)


class ImageEncoder(nn.Module):
    def __init__(self, block=BasicBlock, bc=16, img_layers=[2, 2, 2, 2, 2, 2], drop_path=0.1, norm_layer=nn.BatchNorm2d, padding_mode='zeros', drift=1e6):
        super(ImageEncoder, self).__init__()

        self._norm_layer = norm_layer
        self._padding_mode = padding_mode
        self.drift = drift
        self._img_dpc = 0
        self._img_dprs = np.linspace(0, drop_path, sum(img_layers))

        self.inplanes = bc
        self.conv_img = nn.Sequential(
            Basic2d(3, bc, norm_layer=norm_layer, kernel_size=3, padding=1, stride=1),
            self._make_layer(block, bc, img_layers[0], stride=1)
        )

        self.inplanes = bc
        self.layer1_img = self._make_layer(block, bc * 1, img_layers[1], stride=2)

        self.inplanes = bc * 1
        self.layer2_img = self._make_layer(block, bc * 2, img_layers[2], stride=2)

        self.inplanes = bc * 2
        self.layer3_img = self._make_layer(block, bc * 4, img_layers[3], stride=2)

        self.inplanes = bc * 4
        self.layer4_img = self._make_layer(block, bc * 8, img_layers[4], stride=2)

        self.inplanes = bc * 8
        self.layer5_img = self._make_layer(block, bc * 16, img_layers[5], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        padding_mode = self._padding_mode
        downsample = None
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        else:
            bias = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride, bias=bias),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer, padding_mode=padding_mode,
                  drop_path=self._img_dprs[self._img_dpc]))
        self._img_dpc += 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, padding_mode=padding_mode,
                                drop_path=self._img_dprs[self._img_dpc]))
            self._img_dpc += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        output = [x]
        output.append(self.conv_img(output[-1]))    # [b,32,480,640]
        output.append(self.layer1_img(output[-1]))  # [b,64,240,320]
        output.append(self.layer2_img(output[-1]))  # [b,128,120,160]
        output.append(self.layer3_img(output[-1]))  # [b,256,60,80]
        output.append(self.layer4_img(output[-1]))  # [b,256,30,40]
        output.append(self.layer5_img(output[-1]))  # [b,256,15,20]

        return output[2:]


class DepthEncoder(nn.Module):
    def __init__(self, input_channel=4):
        super(DepthEncoder, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # original
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=5, stride=1, padding=4, dilation=2),
            nn.BatchNorm2d(16),
            self.lrelu,
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            self.lrelu, )  # [1,8,240,320]
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            self.lrelu,
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            self.lrelu,
        )  # [1,16,120,160]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            self.lrelu,
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            self.lrelu,
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            self.lrelu,
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            self.lrelu,
        )  # [1,32,60,80]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            self.lrelu,
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            self.lrelu,
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            self.lrelu,
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            self.lrelu,
        )  # [1,64,30,40]
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            self.lrelu,
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            self.lrelu,
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            self.lrelu,
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            self.lrelu,
        )  # [1,128,15,20]
        self.cbam = CBAM(256)

    def forward(self, x):
        features = []
        features.append(self.conv0(x))
        features.append(self.conv1(features[-1]))
        features.append(self.conv2(features[-1]))
        features.append(self.conv3(features[-1]))
        features.append(self.cbam(self.conv4(features[-1])))

        return features


class ImageEncoder_S(nn.Module):
    def __init__(self, block=BasicBlock, bc=16, img_layers=[1, 1, 1, 1, 1, 1], drop_path=0.1, norm_layer=nn.BatchNorm2d, padding_mode='zeros', drift=1e6):
        super(ImageEncoder_S, self).__init__()

        self._norm_layer = norm_layer
        self._padding_mode = padding_mode
        self.drift = drift
        self._img_dpc = 0
        self._img_dprs = np.linspace(0, drop_path, sum(img_layers))

        self.inplanes = bc
        self.conv_img = nn.Sequential(
            Basic2d(3, bc, norm_layer=norm_layer, kernel_size=3, padding=1, stride=1),
            self._make_layer(block, bc, img_layers[0], stride=1)
        )

        self.inplanes = bc
        self.layer1_img = self._make_layer(block, bc * 1, img_layers[1], stride=2)

        self.inplanes = bc * 1
        self.layer2_img = self._make_layer(block, bc * 2, img_layers[2], stride=2)

        self.inplanes = bc * 2
        self.layer3_img = self._make_layer(block, bc * 4, img_layers[3], stride=2)

        self.inplanes = bc * 4
        self.layer4_img = self._make_layer(block, bc * 8, img_layers[4], stride=2)

        self.inplanes = bc * 8
        self.layer5_img = self._make_layer(block, bc * 16, img_layers[5], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        padding_mode = self._padding_mode
        downsample = None
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        else:
            bias = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride, bias=bias),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer, padding_mode=padding_mode,
                  drop_path=self._img_dprs[self._img_dpc]))
        self._img_dpc += 1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, padding_mode=padding_mode,
                                drop_path=self._img_dprs[self._img_dpc]))
            self._img_dpc += 1
        return nn.Sequential(*layers)

    def forward(self, x):
        output = [x]
        output.append(self.conv_img(output[-1]))    # [b,32,480,640]
        output.append(self.layer1_img(output[-1]))  # [b,64,240,320]
        output.append(self.layer2_img(output[-1]))  # [b,128,120,160]
        output.append(self.layer3_img(output[-1]))  # [b,256,60,80]
        output.append(self.layer4_img(output[-1]))  # [b,256,30,40]
        output.append(self.layer5_img(output[-1]))  # [b,256,15,20]

        return output[2:]


class DepthEncoder_S(nn.Module):
    def __init__(self, input_channel=4):
        super(DepthEncoder_S, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # original
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            self.lrelu,
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            self.lrelu, )  # [1,8,240,320]
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            self.lrelu,
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            self.lrelu,
        )  # [1,16,120,160]
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            self.lrelu,
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            self.lrelu,
        )  # [1,32,60,80]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            self.lrelu,
            nn.Conv2d(128, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            self.lrelu,
        )  # [1,64,30,40]
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            self.lrelu,
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            self.lrelu,
        )  # [1,128,15,20]

    def forward(self, x):
        features = []
        features.append(self.conv0(x))
        features.append(self.conv1(features[-1]))
        features.append(self.conv2(features[-1]))
        features.append(self.conv3(features[-1]))
        features.append(self.conv4(features[-1]))

        return features


class CBAM(nn.Module):
    def __init__(self, channels, reduction=4):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x) * x
        max_pool = torch.max(ca, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(ca, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([max_pool, avg_pool], dim=1)) * ca
        return sa
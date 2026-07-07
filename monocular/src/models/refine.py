import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd
from torch.autograd import Function
import functools
import BpOps


class BpConvLocal(Function):
    @staticmethod
    def forward(ctx, input, weight):
        assert input.is_contiguous()
        assert weight.is_contiguous()
        ctx.save_for_backward(input, weight)
        output = BpOps.Conv2dLocal_F(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input, grad_weight = BpOps.Conv2dLocal_B(input, weight, grad_output)
        return grad_input, grad_weight

bpconvlocal = BpConvLocal.apply

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


class GenKernel_L(nn.Module):
    def __init__(self, in_channels, pk, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.conv = nn.Sequential(
            Basic2d(in_channels, pk * pk - 1, norm_layer=norm_layer, act=act),
        )

    def forward(self, fout):
        weight = self.conv(fout)
        weight_sum = torch.sum(weight.abs(), dim=1, keepdim=True)
        weight = torch.div(weight, weight_sum + self.eps)
        weight_mid = 1 - torch.sum(weight, dim=1, keepdim=True)
        weight_pre, weight_post = torch.split(weight, [weight.shape[1] // 2, weight.shape[1] // 2], dim=1)
        weight = torch.cat([weight_pre, weight_mid, weight_post], dim=1).contiguous()
        return weight


class CSPN(nn.Module):
    """
    implementation of CSPN++
    """
    def __init__(self, in_channels, pt, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.pt = pt
        self.weight3x3 = GenKernel_L(in_channels, 3, norm_layer=norm_layer, act=act, eps=eps)
        self.weight5x5 = GenKernel_L(in_channels, 5, norm_layer=norm_layer, act=act, eps=eps)
        self.weight7x7 = GenKernel_L(in_channels, 7, norm_layer=norm_layer, act=act, eps=eps)

        # self.convmask = nn.Sequential(
        #     Basic2d(in_channels, 3, norm_layer=None, act=nn.Sigmoid),
        # )
        self.convck = nn.Sequential(
            Basic2d(in_channels, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )
        self.convct = nn.Sequential(
            Basic2d(in_channels + 3, 3, norm_layer=None, act=functools.partial(nn.Softmax, dim=1)),
        )
        self.sigma_spatial_3 = nn.Parameter(torch.tensor(1.0))
        self.sigma_spatial_5 = nn.Parameter(torch.tensor(1.0))
        self.sigma_spatial_7 = nn.Parameter(torch.tensor(1.0))

    def get_spatial_kernel(self, diameter, sigma):
        dist_range = torch.linspace(-1, 1, diameter, device=sigma.device)
        x, y = torch.meshgrid(dist_range, dist_range, indexing='ij')
        patch = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        return torch.exp(- patch.square().sum(0) / (2 * sigma ** 2)) \
            .reshape(1, diameter * diameter, 1, 1)

    # @custom_fwd(cast_inputs=torch.float32)
    def forward(self, fout, hn, h0):
        spatial3x3 = self.get_spatial_kernel(3, self.sigma_spatial_3)
        spatial5x5 = self.get_spatial_kernel(5, self.sigma_spatial_5)
        spatial7x7 = self.get_spatial_kernel(7, self.sigma_spatial_7)

        weight3x3 = self.weight3x3(fout)  # [b,9,h,w]
        weight5x5 = self.weight5x5(fout)  # [b,25,h,w]
        weight7x7 = self.weight7x7(fout)  # [b,49,h,w]

        weight3x3 = weight3x3 * spatial3x3
        weight5x5 = weight5x5 * spatial5x5
        weight7x7 = weight7x7 * spatial7x7

        weight3x3 /= weight3x3.sum(1, keepdim=True).clamp(1e-7)
        weight5x5 /= weight5x5.sum(1, keepdim=True).clamp(1e-7)
        weight7x7 /= weight7x7.sum(1, keepdim=True).clamp(1e-7)

        # mask3x3, mask5x5, mask7x7 = torch.split(self.convmask(fout) * (h0 > 1e-3).float(), 1, dim=1)  # all [b,1,h,w]
        conf3x3, conf5x5, conf7x7 = torch.split(self.convck(fout), 1, dim=1)  # all [b,1,h,w]

        hn3x3 = hn5x5 = hn7x7 = hn
        hns = [hn, ]
        for i in range(self.pt):
            hn3x3 = bpconvlocal(hn3x3, weight3x3)
            hn5x5 = bpconvlocal(hn5x5, weight5x5)
            hn7x7 = bpconvlocal(hn7x7, weight7x7)
            # hn3x3 = (1. - mask3x3) * bpconvlocal(hn3x3, weight3x3) + mask3x3 * h0
            # hn5x5 = (1. - mask5x5) * bpconvlocal(hn5x5, weight5x5) + mask5x5 * h0
            # hn7x7 = (1. - mask7x7) * bpconvlocal(hn7x7, weight7x7) + mask7x7 * h0
            if i == self.pt // 2 - 1:
                hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7)
                # hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5)
        hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5 + conf7x7 * hn7x7)
        # hns.append(conf3x3 * hn3x3 + conf5x5 * hn5x5)
        hns = torch.cat(hns, dim=1)
        wt = self.convct(torch.cat([fout, hns], dim=1))
        hn = torch.sum(wt * hns, dim=1, keepdim=True)
        return hn


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class Up_S(nn.Module):
    def __init__(self, inchannel):
        super(Up_S, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.layer1 = PixelShufflePack(inchannel, 8, 2, upsample_kernel=3)  # [1,32,480,640]

    def forward(self, mde_feat, unet_out):
        if mde_feat is None:
            return self.layer1(unet_out)
        else:
            fused_feat = torch.cat((mde_feat, unet_out), dim=1)
            out = self.layer1(fused_feat)
            return out


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)  # Channel-wise attention
        x = x * self.sa(x)  # Spatial attention
        return x


class Up_Cbam(nn.Module):
    def __init__(self, inchannel):
        super(Up_Cbam, self).__init__()
        self.attention = CBAM(inchannel)
        self.pixel_shuffle = PixelShufflePack(inchannel, 16, 2, upsample_kernel=3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, mde_feat, unet_out):
        fused_feat = torch.cat((mde_feat, unet_out), dim=1)
        attended_feat = self.attention(fused_feat)
        out = self.pixel_shuffle(attended_feat)
        return out

class Up(nn.Module):
    def __init__(self, inchannel):
        super(Up, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.layer1 = PixelShufflePack(inchannel, 16, 2, upsample_kernel=3)  # [1,32,480,640]
        # special for midas
        # self.conv = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
    def forward(self, mde_feat, unet_out):
        if mde_feat is None:
            return self.layer1(unet_out)
        else:
            # special for midas
            # mde_feat = self.conv(mde_feat)
            fused_feat = torch.cat((mde_feat, unet_out), dim=1)
            out = self.layer1(fused_feat)
            return out

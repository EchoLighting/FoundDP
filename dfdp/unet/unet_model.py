""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .swin_reg import *
from torch.cuda.amp import autocast

class UNet(nn.Module):
    def __init__(self, n_channels=6, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (ResDoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))

        self.swin = SwinIR(in_chans=1024,
                    window_size=8,  depths=[2, 2, 6, 2],
                    embed_dim=384, num_heads=[3, 6, 12, 24])
        
        # self.down5_add = (Down(1024, 2048))
        # self.up0_add = (Up(2048, 1024, bilinear))

        self.up1 = (Up(1024, 512, bilinear))
        self.up2 = (Up(512, 256, bilinear))
        self.up3 = (Up(256, 128, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.out_depth = (OutConv(64, 1))
        self.out_aif = (OutConv(64, 3))
        # self.sig = nn.Sigmoid()

    def forward(self, left, right):
        stack_rgb = torch.concat([left, right], 1)
        depth, aif = self.fit(stack_rgb)
        return depth, aif

    def forward_self(self, input_dict):
        stack_rgb = input_dict['stack_rgb_img']
        outputs = self.fit(stack_rgb)
        losses, outputs = self.compute_loss(outputs, input_dict)
        return losses, outputs
    
    from torch import amp
    @amp.autocast(device_type="cuda")
    def fit(self,x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.swin(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        depth = self.out_depth(x)
        aif = self.out_aif(x)

        return depth, aif

        outputs = dict()
        outputs['pred_depth'] = depth
        outputs['pred_depth_process'] = 1/(depth+1e-9)
        outputs['pred_AiF_img'] = aif
        return outputs

    def compute_loss(self, outputs, input_dict):
        losses = dict()
        l2 = nn.MSELoss(reduction='mean')
        l1 = nn.SmoothL1Loss(reduction='mean')

        d_real = outputs['pred_depth']
        d_process = outputs['pred_depth_process']
        gt_d, gt_AiF = input_dict['depth'], input_dict['AiF_img']
        gt_d_processs = 1/(gt_d+1e-9)

        mask = gt_d > 0
        mask.detach_()
        losses['depth'] = l1(d_process[mask],gt_d_processs[mask])
        losses['depth_ori'] = l1(d_real[mask],gt_d[mask])

        # AiF = outputs['pred_AiF_img']
        # losses['AiF'] = l2(AiF, gt_AiF)

        # smoothness
        abs_fn = lambda x: x**2
        edge_constant = 50.
        img_gx, img_gy = self.image_grads(gt_AiF)
        weights_x = torch.exp(-torch.mean(
            abs_fn(edge_constant * img_gx), axis=1, keepdims=True))
        weights_y = torch.exp(-torch.mean(
            abs_fn(edge_constant * img_gy), axis=1, keepdims=True))
        d_gx, d_gy = self.image_grads(d_real)
        losses['smooth_ori'] = (
            torch.mean(weights_x * self.robust_l1(d_gx)) +
            torch.mean(weights_y * self.robust_l1(d_gy))) / 2.
        
        losses['total'] = losses['depth_ori'] + 1e-1 * losses['smooth_ori'] 

        return losses, outputs

    def inference(self, input_dict):
        stack_rgb = input_dict['stack_rgb_img']
        outputs = self.fit(stack_rgb)
        return outputs

    def image_grads(self, image_batch, stride=1):
        image_batch_gh = image_batch[..., stride:, :] - image_batch[
            ..., :-stride, :]
        image_batch_gw = image_batch[..., stride:] - image_batch[..., :-stride]
        return image_batch_gh, image_batch_gw

    def robust_l1(self, x):
        """Robust L1 metric."""
        return (x**2 + 0.001**2)**0.5


class ResBlock(nn.Module):
    """
    残差块（Residual Block）
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ResBackbone(nn.Module):
    """
    通用骨干网络
    输入: [B, C, W, H]
    输出: [B, C, W, H]（保持输入输出尺寸一致）
    """
    def __init__(self, in_channels, num_blocks=4):
        super(ResBackbone, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 堆叠多个残差块
        self.resblocks = nn.Sequential(
            *[ResBlock(in_channels) for _ in range(num_blocks)]
        )
        
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.resblocks(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out
    
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation, dilation = dilation, bias=False), nn.BatchNorm2d(out_planes))

class UnetBackbone(nn.Module):
    """
    通用骨干网络
    输入: [B, C, W, H]
    输出: [B, C, W, H]（保持输入输出尺寸一致）
    """
    def __init__(self, n_channels=3, bilinear=False):
        super(UnetBackbone, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (ResDoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        self.down4 = (Down(512, 1024))

        self.up1 = (Up(1024, 512, bilinear))
        self.up2 = (Up(512, 256,  bilinear))
        self.up3 = (Up(256, 128, bilinear))
        self.up4 = (Up(128, 64, bilinear))

        self.branch1 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))


        self.branch3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(64, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))
        self.conv1 =  ResDoubleConv(128, 128)

    from torch import amp
    @amp.autocast(device_type="cuda")
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        output_branch1 = self.branch1(x)
        output_branch1 = F.interpolate(output_branch1, (x.size()[2],x.size()[3]), mode='bilinear',align_corners=True)
        output_branch3 = self.branch3(x)
        output_branch3 = F.interpolate(output_branch3, (x.size()[2],x.size()[3]), mode='bilinear',align_corners=True)
        output_feature = torch.cat((output_branch1, output_branch3,  x), 1)

        out = self.conv1(output_feature)
        # out = self.conv1(x)
        return out
    
class DisplacementSimilarity(nn.Module):
    def __init__(self, ks=21):
        super(DisplacementSimilarity, self).__init__()
        self.ks=ks
        self.max_disp = 2 * ks - 1
        # self.softmax = nn.Softmin(dim=1)
        
    def forward(self, A, B):
        B_size, C, H, W = A.size()
        padded_A = F.pad(A, (self.ks - 1, self.ks - 1, 0, 0), mode="constant", value=0)
        similarity_map = []

        for shift in range(self.max_disp):
            shifted_A = padded_A[:, :, :, shift:shift + W]  

            similarity = (shifted_A - B).sum(dim=1)  # 点乘并在通道维度求和
            similarity_map.append(similarity)

        similarity_map = torch.stack(similarity_map, dim=1)  # 在宽度方向拼接
        # x = self.softmax(similarity_map)
        
        # disp = torch.reshape(torch.arange(0, self.max_disp, device=torch.cuda.current_device(), dtype=torch.float32),[1,self.max_disp,1,1])
        # disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        # similarity_map = x * disp

        return similarity_map

class SimilarityNeck(nn.Module):
    def __init__(self, ks=21, out_channels=128):
        super(SimilarityNeck, self).__init__()
        self.conv_disp = ResDoubleConv(2 * ks - 1, 128)
        self.out_conv = ResDoubleConv(384, 256)

    def forward(self, similarity_map,xl,yr):
        # 转换为 [B, C', W, H] 格式
        disp = self.conv_disp(similarity_map)
        cat = torch.concat([disp,xl,yr],dim=1)
        out = self.out_conv(cat)
        return out

class ShiftUNet(nn.Module): #效果是太细了，感官不够舒服，需要一个统一架构
    def __init__(self):
        super().__init__()
        self.backbone = UnetBackbone()
        self.disp = DisplacementSimilarity(ks=33)
        self.neck = SimilarityNeck(ks=33)
        self.outconv = OutConv(256,1)

    def forward(self, xl, yr):  
        xl = self.backbone(xl)
        yr = self.backbone(yr)
        disp = self.disp(xl, yr)
        neck = self.neck(disp,xl,yr)
        depth = self.outconv(neck)
        aif = None
        return depth, aif


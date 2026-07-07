import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=False)
    )

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
#        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=False)
        return x

class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat
        
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=1, padding=1) # stride 2->1

        if self.concat: 
            self.conv2 = BasicConv(out_channels*2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x, rem):
        # (4, 64, 1, 48, 48)
        x = self.up2(x)
        # x (4, 64, 2, 96, 96) rem (4, 48, 2, 96, 96)
        x = self.conv1(x) 
        # print(x.size())

        assert(x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x

class Disp(nn.Module):
    def __init__(self, maxdisp=12):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)

    def forward(self, x):
        x = F.interpolate(x, [self.maxdisp, x.size()[3]*4, x.size()[4]*4], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        # x = torch.clamp(x, -30, 30)
        x = self.softmax(x)
        x = self.disparity(x)
        return x
    
class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = torch.reshape(torch.arange(-self.maxdisp//2, self.maxdisp//2, device=torch.cuda.current_device()),[1,self.maxdisp,1,1])
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            # x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            out = torch.sum(x * disp, 1,keepdim=True)
        return out


class ConfidenceHead(nn.Module):
    def __init__(self, in_ch_feat=32, in_ch_prob=2, mid=32):
        super(ConfidenceHead, self).__init__()
        self.conv = nn.Sequential(
            convbn(in_ch_feat + in_ch_prob, mid, 3, 1, 1),
            nn.Conv2d(mid, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, feat, var_prob_map, peak_map):
        # feat: [B,C,H,W], var_prob_map: [B,1,H,W], peak_map: [B,1,H,W]
        x = torch.cat([feat, var_prob_map, peak_map], dim=1)
        x = torch.nan_to_num(x)
        conf = self.conv(x)
        return conf  # in (0,1)


class MatchingModified(nn.Module):
    def __init__(self):
        super(MatchingModified, self).__init__()
        self.start =  nn.Sequential(
            BasicConv(64, 32, is_3d=True, kernel_size=3, padding=1),
            BasicConv(32, 48, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(48, 64, is_3d=True, kernel_size=3, padding=1))
        self.conv1a = nn.Sequential(
            BasicConv(64, 64, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(64, 64, is_3d=True, kernel_size=3, padding=1))
        self.deconv1a = Conv2x(64, 64, is_3d=True, deconv=False)
        # final 3D -> produce cost volume logits
        self.end = nn.Sequential(
            BasicConv(64, 64, is_3d=True, kernel_size=4, padding=1, stride=2, deconv=True),
            BasicConv(64, 1, is_3d=True, kernel_size=3, padding=1, stride=1, bn=False, relu=False)
        )
        # small 2D head to predict log_sigma from decoded 2D features 
        self.logsigma_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, cost3d):
        x = self.start(cost3d)
        rem0 = x
        x = self.conv1a(x)
        x = self.deconv1a(x, rem0)
        x = self.end(x)   # [B,1,D,H,W]  (3D)
        return x  # logits for disparity volume (squeezed channel later)

class Stereonet_DP(nn.Module):
    def __init__(self, maxdisp=12, channel = 128, mchannel = 32, patchsize = 192):
        super(Stereonet_DP, self).__init__()
        self.maxdisp = maxdisp
        self.channel = channel
        self.mchannel = mchannel
        self.patchsize = patchsize
        self.feature = Feature()
        self.matching = MatchingModified()
        self.softmax = nn.Softmin(dim=1)
        self.disparity = Disp(maxdisp=self.maxdisp)
        # confidence head to fuse feature & prob-var
        self.confidence_head = ConfidenceHead(in_ch_feat=32, in_ch_prob=2, mid=32)


        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                if hasattr(m, 'weight'):
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)

    def build_cost_volume(self, xl_feat, yr_feat):
        B, C, H, W = xl_feat.size()
        maxdisp = self.maxdisp
        cost = xl_feat.new_zeros(B, C * 2, maxdisp, H, W)
        for i in range(maxdisp):
            gap = i - maxdisp // 2
            if gap < 0:
                cost[:, :C, i, :, :gap] = xl_feat[:, :, :, :gap]
                cost[:, C:, i, :, :gap] = yr_feat[:, :, :, -gap:]
            elif gap == 0:
                cost[:, :C, i, :, :] = xl_feat
                cost[:, C:, i, :, :] = yr_feat
            else:
                cost[:, :C, i, :, gap:] = xl_feat[:, :, :, gap:]
                cost[:, C:, i, :, gap:] = yr_feat[:, :, :, :-gap]
        return cost  # [B, 2C, D, H, W]

    def forward(self, xl, yr):
        feat_l = self.feature(xl)  # [B,32,H',W']
        feat_r = self.feature(yr)
        B, C, Hf, Wf = feat_l.shape

        cost3d = self.build_cost_volume(feat_l, feat_r)  # [B, 2C, D, H', W']
        logits3d = self.matching(cost3d)   
        depth = self.disparity(logits3d)
        logits = torch.squeeze(F.interpolate(logits3d, [self.maxdisp, Hf*4, Wf*4], mode='trilinear', align_corners=False), 1)
        prob = self.softmax(logits)  # probability over disparities

        # compute theoretical variance from prob volume
        disp_range = torch.arange(0, self.maxdisp, device=prob.device, dtype=prob.dtype).view(1, -1, 1, 1)
        var_prob = torch.sum(prob * (disp_range - depth) ** 2, dim=1, keepdim=True) 
        var_prob_max = torch.max(var_prob).clamp(min=1e-6)
        var_prob_clamp = var_prob / var_prob_max

        # peak probability (sharpness)
        peak_prob, _ = torch.max(prob, dim=1, keepdim=True)  # [B,1,H',W']

        feat = F.interpolate(feat_l, [Hf*4, Wf*4], mode='bilinear', align_corners=False)
        conf_map = self.confidence_head(feat, var_prob_clamp, peak_prob)  # [B,1,H',W']

        return depth, conf_map, {'disp': depth.unsqueeze(1), 'prob': prob, 'var_prob': var_prob, 'peak_prob': peak_prob}

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 64, kernel_size=3, stride=1, padding=1),
            BasicConv(64, 64, kernel_size=3, stride=2, padding=1))
        self.layer1 = nn.Sequential(
            BasicConv(64, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            BasicConv(128, 128, kernel_size=3, stride=1, padding=8,dilation=8),
            BasicConv(128, 128, kernel_size=3, stride=2, padding=1))
        # self.start = nn.Sequential(
        #     BasicConv(3, 32, kernel_size=3, padding=1),
        #     BasicConv(32, 64, kernel_size=3, stride=3, padding=1))
        # self.layer1 = nn.Sequential(
        #     BasicConv(64, 128, kernel_size=3, stride=1, padding=4, dilation=4),
        #     BasicConv(128, 128, kernel_size=3, stride=1, padding=8,dilation=8))

        self.branch1 = nn.Sequential(nn.AvgPool2d((20, 20), stride=(20, 20)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=False))


        self.branch3 = nn.Sequential(nn.AvgPool2d((10, 10), stride=(10, 10)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=False))

        self.end = nn.Sequential(
            BasicConv(192, 96, kernel_size=3, stride=1, padding=1),
            BasicConv(96, 32, kernel_size=1, bn=False, relu=False, padding=0))

    def forward(self, x):
        # print(x.size())
        x = self.start(x)
        # print('1',x.size())

        x = self.layer1(x)
        # print('2',x.size())


        output_branch1 = self.branch1(x)
        output_branch1 = F.interpolate(output_branch1, (x.size()[2],x.size()[3]),mode='bilinear',align_corners=True)
        output_branch3 = self.branch3(x)
        output_branch3 = F.interpolate(output_branch3, (x.size()[2],x.size()[3]),mode='bilinear',align_corners=True)
              
        output_feature = torch.cat((output_branch1, output_branch3,  x), 1)
        output_feature = self.end(output_feature)
        # print('2',output_feature.size())

        return output_feature
    

class Stereonet_3D(nn.Module):
    def __init__(self, maxdisp=12, channel = 128, mchannel = 32, patchsize = 192):
        super(Stereonet_3D, self).__init__()
        self.maxdisp = maxdisp
        self.channel = channel
        self.mchannel = mchannel
        self.patchsize = patchsize
        self.feature = Feature()
        # self.features = Feature()
        self.matching = Matching()
        self.disp = Disp(self.maxdisp)
        self.softmax = nn.Softmin(dim=1)
        self.confidence_head = ConfidenceHeadAdvanced(in_ch_feat=32, in_ch_prob=2, mid=32)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xl, yr):  
        x = self.feature(xl)    # B 32 H W
        y = self.feature(yr)    # B 32 H W

        B, C, H, W = x.size()
        cost = torch.zeros(B, C * 2, self.maxdisp, H, W).type_as(x)  # [B, 64, D, 1/4H, 1/4W]
        for i in range(self.maxdisp):
            gap = i-self.maxdisp//2
            if gap < 0:
                cost[:, :C, i, :, :gap] = x[:, :, :, :gap]
                cost[:, C:, i, :, :gap] = y[:, :, :, -gap:]
            elif gap == 0:
                cost[:, :C, i, :, :] = x
                cost[:, C:, i, :, :] = y
            if gap > 0:
                cost[:, :C, i, :, gap:] = x[:, :, :, gap:]
                cost[:, C:, i, :, gap:] = y[:, :, :, :-gap]

        cost = self.matching(cost)   
        depth = self.disp(cost)
        # logits = torch.squeeze(F.interpolate(cost, [self.maxdisp, H*4, W*4], mode='trilinear', align_corners=False), 1)
        # prob = self.softmax(logits)  # probability over disparities

        # # compute theoretical variance from prob volume
        # disp_range = torch.arange(0, self.maxdisp, device=prob.device, dtype=prob.dtype).view(1, -1, 1, 1)
        # var_prob = torch.sum(prob * (disp_range - depth) ** 2, dim=1, keepdim=True) 
        # var_prob_max = torch.max(var_prob).clamp(min=1e-6)
        # var_prob_clamp = var_prob / var_prob_max

        # # peak probability (sharpness)
        # peak_prob, _ = torch.max(prob, dim=1, keepdim=True)  # [B,1,H',W']

        # feat = F.interpolate(x, [H*4, W*4], mode='bilinear', align_corners=False)
        # conf_map = self.confidence_head(feat, var_prob_clamp, peak_prob)  # [B,1,H',W']

        return depth
    
    @torch.no_grad()
    def infer_image(self, l, r):
        depth = self.forward(l, r)
        return depth

class Matching(nn.Module):
    def __init__(self):
        super(Matching, self).__init__()
        self.start =  nn.Sequential(
            BasicConv(64, 32, is_3d=True, kernel_size=3, padding=1),
            BasicConv(32, 48, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(48, 64, is_3d=True, kernel_size=3, padding=1))
        self.conv1a = nn.Sequential(
            BasicConv(64, 64, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(64, 64, is_3d=True, kernel_size=3, padding=1))
        # self.conv2a = nn.Sequential(
        #     BasicConv(64, 96, is_3d=True, kernel_size=3, stride=2, padding=1),
        #     BasicConv(96, 96, is_3d=True, kernel_size=3, padding=1))
        # self.conv3a = nn.Sequential(
        #     BasicConv(96, 128, is_3d=True, kernel_size=3, stride=2, padding=1),
        #     BasicConv(128, 128, is_3d=True, kernel_size=3, padding=1))
        # self.deconv3a = Conv2x(128, 96, is_3d=True, deconv=True)
        # self.deconv2a = Conv2x(96, 64, is_3d=True, deconv=True)
        self.deconv1a = Conv2x(64, 64, is_3d=True, deconv=False) # True -> False
        self.end = nn.Sequential(
            BasicConv(64, 64, is_3d=True, kernel_size=4, padding=1, stride=2, deconv=True),
            BasicConv(64, 1, is_3d=True, kernel_size=3, padding=1, stride=1, bn=False, relu=False))

    def forward(self, x):
        x = self.start(x)
        rem0 = x
        x = self.conv1a(x)
        #rem1 = x
        #x = self.conv2a(x)
        #rem2 = x
        #x = self.conv3a(x)
        #pdb.set_trace()
        #x = self.deconv3a(x, rem2)
        # = self.deconv2a(x, rem1)
        # print(x.size(), rem0.size())
        x = self.deconv1a(x, rem0) #(4 64 4 48 48,  4 48 2 96 96)
        x = self.end(x)
        return x

class DepthFusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1)
        )

    def forward(self, I_rel, I_metric, conf_map):
        x = torch.cat([I_rel, I_metric, conf_map], dim=1)
        return self.decoder(self.encoder(x))


def to_log_depth(x, eps=1e-6):
    # x: depth in meters, assume positive; returns log(depth)
    return torch.log(x + eps)

def from_log_depth(x):
    return torch.exp(x)

def compute_gradients(x):
    """
    Simple sobel-like gradients (abs). x: [B,1,H,W]
    returns gx, gy: same shape
    """
    # horizontal gradient: right - left
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]  # [B,1,H,W-1]
    gx = F.pad(gx, (0,1,0,0))  # pad to original size
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]  # [B,1,H-1,W]
    gy = F.pad(gy, (0,0,0,1))
    return gx, gy

def grad_magnitude(x):
    gx, gy = compute_gradients(x)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)

# -------- small U-Net style fusion network --------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=ks//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=ks, stride=1, padding=ks//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class DepthFusionC2Plus(nn.Module):
    """
    C2+ fusion: takes I_rel, I_metric, conf_map and outputs refined metric depth.
    Operates in log-depth domain internally (configurable).
    """
    def __init__(self, in_channels=4, base_ch=64, use_rgb=False, use_log=True):
        """
        in_channels: number of channels concatenated (if rgb used, adjust)
        base_ch: base channels
        use_rgb: whether rgb is concatenated (set False if not)
        use_log: use log-depth internally (recommended)
        """
        super().__init__()
        self.use_rgb = use_rgb
        self.use_log = use_log

        # encoder
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch*2, base_ch*4)

        # bottleneck
        self.bottleneck = ConvBlock(base_ch*4, base_ch*4)

        # decoder
        self.up2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_ch*2, base_ch)

        # output residual head
        self.res_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch//2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch//2, 1, kernel_size=1)
        )

        # optional small gate to scale DP input feature
        self.conf_proj = nn.Sequential(
            nn.Conv2d(1, base_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, I_rel, I_metric, conf_map, rgb=None):
        """
        Expect shapes [B,1,H,W] for depths & conf. rgb optionally [B,3,H,W]
        Returns depth_pred in same domain as inputs (if use_log True, inputs should be log-depth).
        """
        # prepare inputs
        # compute grad of I_rel and include as channel
        gx, gy = compute_gradients(I_rel)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)  # [B,1,H,W]

        # gate DP metric by conf (soft gating)
        dp_input = I_metric * conf_map  # suppress low-conf DP signal

        # concat channels: [I_rel, grad_mag, dp_input, conf]
        if self.use_rgb and (rgb is not None):
            x = torch.cat([I_rel, grad_mag, dp_input, conf_map, rgb], dim=1)
        else:
            x = torch.cat([I_rel, grad_mag, dp_input, conf_map], dim=1)

        # UNet forward
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)

        b = self.bottleneck(e3)

        u2 = self.up2(b)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        res = self.res_head(d1)  # residual in same domain as I_rel (log or inv)
        # initial guess D_init: blend I_metric (gated) and scaled I_rel
        # use learned small gate to scale dp contribution per spatial location
        dp_gate = self.conf_proj(conf_map)  # [B,base_ch,H,W] -> but final multiply use mean along channel? use scalar map:
        # produce scalar gate map
        gate_map = torch.mean(dp_gate, dim=1, keepdim=True)  # [B,1,H,W] in (0,1)

        D_init = gate_map * (I_metric) + (1.0 - gate_map) * I_rel
        depth_out = D_init + res  # residual correction

        return depth_out, {'D_init': D_init, 'res': res, 'gate_map': gate_map}



def conv3x3(in_planes, out_planes, stride=1, bias=True):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

class ConfidenceHeadAdvanced(nn.Module):
    """
    Enhanced Confidence Head with structural & disparity cues
    """
    def __init__(self, in_ch_feat=32, in_ch_prob=1, mid=64, use_grad=True, use_localvar=True):
        super(ConfidenceHeadAdvanced, self).__init__()
        self.use_grad = use_grad
        self.use_localvar = use_localvar

        # in_channels = in_ch_feat + in_ch_prob
        # if use_grad:
        #     in_channels += 1
        # if use_localvar:
        #     in_channels += 1

        self.conv = nn.Sequential(
            conv3x3(35, mid),
            nn.ReLU(inplace=True),
            conv3x3(mid, mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, feat, var_prob_map, peak_map, image=None):
        x_list = [feat, var_prob_map, peak_map]

        # gradient magnitude
        if self.use_grad and image is not None:
            gray = torch.mean(image, dim=1, keepdim=True)
            grad_x = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0,1,0,0))
            grad_y = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0,0,0,1))
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
            grad_mag = grad_mag / (grad_mag.max() + 1e-6)
            x_list.append(grad_mag)

        # local variance of feature
        if self.use_localvar:
            local_mean = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
            local_var = torch.mean((feat - local_mean)**2, dim=1, keepdim=True)
            local_var = local_var / (local_var.max() + 1e-6)
            x_list.append(local_var)

        x = torch.cat(x_list, dim=1)
        x = torch.nan_to_num(x)
        conf = self.conv(x)

        x = (x - x.min()) / (x.max() - x.min())
        return conf

class ConfidenceNetLite(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 32  # small but expressive

        self.enc1 = nn.Sequential(
            nn.Conv2d(10, ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 2, 1),   # downsample
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch, ch, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Sequential(
            nn.Conv2d(ch, 1, 1),
            # nn.Sigmoid()
        )

    def forward(self, disp, xl, yr, tex, light, dp):
        x = torch.cat([disp, xl, yr, tex, light, dp], dim=1)
        # x = disp
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d  = self.dec(e2)

        fuse = d + e1     # residual skip
        conf = self.out(fuse)
        return conf

    def infer_image(self, disp, xl, yr, tex, light, dp):
        with torch.no_grad():
            return self.forward(disp, xl, yr, tex, light, dp)



class ConfidenceNetV2(nn.Module):
    """
    Lightweight RAFT-style encoder + U-Net decoder confidence network.
    Input channels default = 7 (inv_depth + rgb_l + rgb_r)
    Output: logits (B,1,H,W)
    """
    def __init__(self, in_ch=7, base_ch=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)         # same size
        self.enc2 = ConvBlock(base_ch, base_ch * 2)   # down1
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)  # down2

        self.pool = nn.AvgPool2d(2, 2)

        # bottleneck
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 4)

        # decoder
        self.up1 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec2 = ConvBlock(base_ch * 2, base_ch)

        # output logits
        self.out_conv = nn.Conv2d(base_ch, 1, kernel_size=1)

        # init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inv_depth_norm: torch.Tensor, rgb_l: torch.Tensor, rgb_r: torch.Tensor) -> torch.Tensor:
        """
        inv_depth_norm: (B,1,H,W)
        rgb_l, rgb_r: (B,3,H,W)
        returns logits: (B,1,H,W)  (raw logits, no sigmoid)
        """
        x = torch.cat([inv_depth_norm, rgb_l, rgb_r], dim=1)  # (B,7,H,W)
        # x = torch.cat([rgb_l, rgb_r], dim=1)  # (B,7,H,W)
        e1 = self.enc1(x)         # (B,base,H,W)
        e2 = self.pool(e1)
        e2 = self.enc2(e2)        # (B,base*2,H/2,W/2)
        e3 = self.pool(e2)
        e3 = self.enc3(e3)        # (B,base*4,H/4,W/4)

        b = self.bottleneck(e3)

        d1 = self.up1(b)          # (B,base*2,H/2,W/2)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)         # (B,base,H,W)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        logits = self.out_conv(d2)  # (B,1,H,W) logits
        return logits

    def infer_image(self, inv_depth_norm: torch.Tensor, rgb_l: torch.Tensor, rgb_r: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(inv_depth_norm, rgb_l, rgb_r)
        
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Shallow CNN encoder (local features)
# ---------------------------------------------------------
class LocalEncoder(nn.Module):
    def __init__(self, in_ch=7, out_ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)  # [B,64,H,W]


# ---------------------------------------------------------
# Patch embedding for ViT
# ---------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=64, embed_dim=96, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch, patch)  # downsample 4×
        self.patch = patch

    def forward(self, x):
        x = self.proj(x)     # [B,embed_dim,H/4,W/4]
        B, C, H, W = x.shape
        return x, (H, W)


# ---------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim=96, num_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        # x: [B,HW,C]
        h = x
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = h + attn_out          # residual 1

        h = x
        x2 = self.norm2(x)
        x = h + self.mlp(x2)      # residual 2
        return x


# ---------------------------------------------------------
# Simple Decoder
# ---------------------------------------------------------
class SimpleDecoder(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim//2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, 1, 3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)  # logits


# ---------------------------------------------------------
# Main Network: ConfidenceViT
# ---------------------------------------------------------
class ConfidenceViT(nn.Module):
    """
    Inputs:
        inv_depth: [B,1,H,W]
        rgb_l:     [B,3,H,W]
        rgb_r:     [B,3,H,W]

        Concatenated input: [B,7,H,W]

    Output:
        conf_logits: [B,1,H,W]
    """

    def __init__(self, embed_dim=48, depth=4):
        super().__init__()

        # 1. Local CNN
        self.local = LocalEncoder(in_ch=18, out_ch=64)

        # 2. Patch embedding
        self.patch = PatchEmbed(in_ch=64, embed_dim=embed_dim, patch=8)

        # 3. Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=4, mlp_ratio=4)
            for _ in range(depth)
        ])

        # 4. Decoder
        self.decoder = SimpleDecoder(dim=embed_dim)

    def forward(self, inv_depth, xl, yr, diff_lr, grad_l, grad_r, grad_inv, var):

        x = torch.cat([inv_depth, xl, yr, diff_lr, grad_l, grad_r, grad_inv, var], dim=1)  # [B,7,H,W]
        # x = torch.cat([rgb_l, rgb_r], dim=1)  # [B,6,H,W]
        # x = torch.cat([inv_depth, tex, light, dp], dim=1)

        # 1. Local feature extractor
        feat = self.local(x)   # [B,64,H,W]

        # 2. Patch embedding
        tokens, (H, W) = self.patch(feat)   # [B,C,H/4,W/4]
        B, C, H2, W2 = tokens.shape

        # reshape to [B,HW,C] for transformer
        x = tokens.flatten(2).transpose(1, 2)

        # 3. Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # back to image shape
        x = x.transpose(1, 2).reshape(B, C, H2, W2)

        # 4. Decode + upsample
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        conf_logits = self.decoder(x)

        conf_logits = F.leaky_relu(conf_logits, negative_slope=0.1)
        # conf_logits = F.interpolate(conf_logits, size=inv_depth.shape[2:], mode='bilinear', align_corners=False)

        return conf_logits
    
    def infer_image(self, inv_depth, xl, yr, diff_lr, grad_l, grad_r, grad_inv, var):
        with torch.no_grad():
            return self.forward(inv_depth, xl, yr, diff_lr, grad_l, grad_r, grad_inv, var)

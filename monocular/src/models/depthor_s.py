import torch
import torch.nn as nn
from .decoder import Decoder_S, DepthRegression_S
from .refine import CSPN, Up_S
from .encoder import ImageEncoder_S, DepthEncoder_S
from src.utils.set_mde import compute_rel_depth, compute_metric_depth, set_depthanything, set_depthanything_metric, transforms_cfg


class Depthor(nn.Module):
    def __init__(self, n_bins=100, min_val=0.1, max_val=10, norm='linear'):
        super(Depthor, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.img_encoder = ImageEncoder_S()
        self.SpEncoder = DepthEncoder_S(input_channel=3)
        self.depth_head = DepthRegression_S(64, dim_out=n_bins//2, norm=norm)
        self.decoder = Decoder_S(num_classes=64)
        self.conv_out = nn.Sequential(nn.Conv2d(64, n_bins//2, kernel_size=1, stride=1, padding=0),
                                      nn.Softmax(dim=1))
        self.align_mde = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),)
        self.up = Up_S(inchannel=32 + 64)
        self.refine = CSPN(in_channels=8, pt=6)
        self._reset_parameters()

        self.depth_anything = set_depthanything(encoder='vits')
        # self.depth_anything = set_depthanything_metric(encoder='vits')
        for param in self.depth_anything.parameters():
            param.requires_grad = False
        # self.depth_anything_metric = set_depthanything_metric(encoder='vitl')

    def _reset_parameters(self):
        modules = [self.depth_head, self.decoder, self.conv_out, self.SpEncoder, self.img_encoder, self.refine, self.up, self.align_mde]
        for s in modules:
            for m in s.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_data):
        x = input_data['image']  # [b, 3, 480, 640]
        x = (x - self._mean) / self._std  # normalize
        sparse_depth = input_data['sparse_depth']  # [b, 1, 480, 640]
        B, _, H, W = x.shape

        cfg = transforms_cfg["zju_s"]
        mde_feat, rel_depth, inv_depth = compute_rel_depth(self.depth_anything, x, **cfg)

        sparse_depth_all = sparse_depth

        depth_features = self.SpEncoder(torch.cat((inv_depth, rel_depth, sparse_depth_all), dim=1))

        img_features = self.img_encoder(x)
        unet_out = self.decoder(img_features, depth_features, )

        bin_widths_normed, range_attention_maps = self.depth_head(unet_out)
        out = self.conv_out(range_attention_maps)
        bin_widths = (self.max_val - self.min_val) * bin_widths_normed
        bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_val)
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)
        depth_0 = torch.sum(out * centers, dim=1, keepdim=True)

        depth_0 = nn.functional.interpolate(depth_0, [H, W], mode='bilinear', align_corners=True)

        # refine with mde+unet
        mde_feat = self.align_mde(mde_feat)
        cspn_feat = self.up(mde_feat, unet_out)
        final = self.refine(cspn_feat, depth_0, sparse_depth)

        return depth_0, final

    def get_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.SpEncoder, self.img_encoder, self.depth_head, self.conv_out, self.refine, self.up, self.align_mde]
        for m in modules:
            yield from m.parameters()

    def set_extra_param(self, device):
        self.register_buffer('_mean', torch.tensor(
            [0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor(
            [0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))


if __name__ == '__main__':
    model = Depthor.build(100)
    x = torch.rand(2, 3, 480, 640)
    bins, pred = model(x)
    print(bins.shape, pred.shape)

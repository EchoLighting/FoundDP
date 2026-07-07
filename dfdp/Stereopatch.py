import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=False),
    )


class BasicConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deconv=False,
        is_3d=False,
        bn=True,
        relu=True,
        **kwargs,
    ):
        super().__init__()
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


class Feature(nn.Module):
    """Feature extractor for small input patch (p=50)."""

    def __init__(self):
        super().__init__()
        self.start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 64, kernel_size=3, stride=1, padding=1),
            BasicConv(64, 64, kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            BasicConv(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv(96, 128, kernel_size=3, stride=1, padding=1),
        )

        # Adaptive context branches avoid fixed pool size issues on small patches.
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=False),
        )
        self.branch3 = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            convbn(128, 32, 1, 1, 0, 1),
            nn.ReLU(inplace=False),
        )

        self.end = nn.Sequential(
            BasicConv(192, 96, kernel_size=3, stride=1, padding=1),
            BasicConv(96, 32, kernel_size=1, bn=False, relu=False, padding=0),
        )

    def forward(self, x):
        x = self.start(x)
        x = self.layer1(x)

        b1 = F.interpolate(self.branch1(x), x.shape[2:], mode="bilinear", align_corners=True)
        b3 = F.interpolate(self.branch3(x), x.shape[2:], mode="bilinear", align_corners=True)
        return self.end(torch.cat([b1, b3, x], dim=1))


class Matching3D(nn.Module):
    def __init__(self, in_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            BasicConv(in_ch, 48, is_3d=True, kernel_size=3, padding=1),
            BasicConv(48, 32, is_3d=True, kernel_size=3, padding=1),
            BasicConv(32, 16, is_3d=True, kernel_size=3, padding=1),
            BasicConv(16, 1, is_3d=True, kernel_size=3, padding=1, bn=False, relu=False),
        )

    def forward(self, x):
        return self.net(x)


class Stereopatch(nn.Module):
    """Predict center-point depth from left/right patches: [B,3,p,p] -> [B,1]."""

    def __init__(self, maxdisp=24):
        super().__init__()
        self.maxdisp = maxdisp
        self.feature = Feature()
        self.matching = Matching3D(in_ch=64)

        # Learnable temperature for softmin over disparity costs (helps probability sharpness & calibration).
        self.log_tau = nn.Parameter(torch.tensor(0.0))

        # Disparity support for soft-argmin style center regression.
        disp_values = torch.arange(-maxdisp // 2, maxdisp // 2, dtype=torch.float32)
        self.register_buffer("disp_values", disp_values, persistent=False)

        in_dim = maxdisp + 64 + 4  # center_logits + center_feat + [disp_mean, log_var, entropy, peak]
        self.center_trunk = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        self.depth_head = nn.Linear(64, 1)
        self.log_scale_head = nn.Linear(64, 1)

        # Learnable post-calibration for confidence only:
        #   s_conf = a * s_raw + b, with a>0 to preserve monotonicity.
        # This adjusts confidence scale without changing depth NLL (which uses s_raw).
        # We normalize so that raw_a=0 => a≈1.
        self._conf_a_raw = nn.Parameter(torch.tensor(0.0))
        self._conf_b = nn.Parameter(torch.tensor(0.0))

        # NOTE: confidence will be derived from predicted log_scale (probabilistic calibration).
        # Keep this module for backward compatibility / ablations if needed.
        self.conf_fuse = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _build_cost_volume(self, xl_feat, yr_feat):
        b, c, h, w = xl_feat.shape
        d = self.maxdisp
        cost = torch.zeros(b, c * 2, d, h, w, device=xl_feat.device, dtype=xl_feat.dtype)

        for i in range(d):
            gap = i - d // 2
            if gap == 0:
                cost[:, :c, i] = xl_feat
                cost[:, c:, i] = yr_feat
            elif gap < 0:
                cost[:, :c, i, :, :gap] = xl_feat[:, :, :, :gap]
                cost[:, c:, i, :, :gap] = yr_feat[:, :, :, -gap:]
            else:
                cost[:, :c, i, :, gap:] = xl_feat[:, :, :, gap:]
                cost[:, c:, i, :, gap:] = yr_feat[:, :, :, :-gap]
        return cost

    def forward(
        self,
        xl,
        yr,
        return_conf: bool = False,
        return_aux: bool = False,
        conf_rel_thresh: Optional[float] = None,
        log_scale_min: float = -6.0,
        log_scale_max: float = 3.0,
    ):
        xl_feat = self.feature(xl)
        yr_feat = self.feature(yr)

        cost = self._build_cost_volume(xl_feat, yr_feat)
        logits_3d = self.matching(cost).squeeze(1)  # [B, D, Hf, Wf]

        h_mid = logits_3d.shape[2] // 2
        w_mid = logits_3d.shape[3] // 2
        center_logits = logits_3d[:, :, h_mid, w_mid]  # [B, D]

        # Disparity distribution statistics (stereo confidence common cues: peak/entropy/variance).
        # Treat center_logits as costs and convert to probability with softmin.
        tau = F.softplus(self.log_tau) + 1e-3
        center_prob = torch.softmax(-center_logits / tau, dim=1)
        disp = self.disp_values.to(center_prob.dtype).view(1, -1)
        disp_mean = torch.sum(center_prob * disp, dim=1, keepdim=True)
        disp_var = torch.sum(center_prob * (disp - disp_mean) ** 2, dim=1, keepdim=True)
        disp_log_var = torch.log(disp_var + 1e-6)
        disp_entropy = -torch.sum(center_prob * torch.log(center_prob + 1e-12), dim=1, keepdim=True)
        disp_peak = torch.amax(center_prob, dim=1, keepdim=True)

        xl_center = xl_feat[:, :, h_mid, w_mid]
        yr_center = yr_feat[:, :, h_mid, w_mid]
        center_feat = torch.cat([xl_center, yr_center], dim=1)  # [B, 64]

        head_in = torch.cat(
            [center_logits, center_feat, disp_mean, disp_log_var, disp_entropy, disp_peak],
            dim=1,
        )
        trunk = self.center_trunk(head_in)
        depth = self.depth_head(trunk)
        depth = F.softplus(depth) + 1e-6

        log_scale = self.log_scale_head(trunk)

        # Confidence as calibrated probability: P(rel_err <= r0).
        # Under a Laplace model on relative error with scale b=exp(s):
        #   P(|e_rel| <= r0) = 1 - exp(-r0 / b) = 1 - exp(-r0 * exp(-s)).
        # This is strictly monotone in s, and can be trained via BCE using the event label.
        s_raw = torch.clamp(log_scale, min=log_scale_min, max=log_scale_max)

        # a>0 ensures conf is strictly monotone w.r.t. s_raw
        conf_a = F.softplus(self._conf_a_raw) / 0.6931471805599453
        conf_b = self._conf_b
        s = torch.clamp(conf_a * s_raw + conf_b, min=log_scale_min, max=log_scale_max)
        if conf_rel_thresh is None:
            # Default (option B): probability of being within 10% relative error.
            conf_rel_thresh = 0.1
        r0 = float(conf_rel_thresh)
        conf = 1.0 - torch.exp(-r0 * torch.exp(-s))
        conf = conf.clamp(0.0, 1.0)

        if not return_conf:
            return depth

        if return_aux:
            aux = {
                "disp_mean": disp_mean,
                "disp_log_var": disp_log_var,
                "disp_entropy": disp_entropy,
                "disp_peak": disp_peak,
                "log_scale": log_scale,
                "conf_a": conf_a.detach(),
                "conf_b": conf_b.detach(),
                "tau": tau.detach(),
            }
            return depth, conf, aux
        return depth, conf

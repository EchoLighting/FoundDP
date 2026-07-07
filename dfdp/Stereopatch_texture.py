import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from .Stereopatch import Feature, Matching3D


def _center_window_mean(x, win: int = 5):
    """
    x: [B,C,H,W]
    return: [B,1]
    """
    b, c, h, w = x.shape
    wh = min(win, h)
    ww = min(win, w)

    h0 = max(h // 2 - wh // 2, 0)
    w0 = max(w // 2 - ww // 2, 0)
    h1 = h0 + wh
    w1 = w0 + ww

    patch = x[:, :, h0:h1, w0:w1]
    return patch.mean(dim=(1, 2, 3), keepdim=False).view(b, 1)


def _center_window_abs_mean(x, win: int = 5):
    return _center_window_mean(torch.abs(x), win=win)


def _warp_right_to_left_x(right, disp_x, padding_mode="border"):
    """
    Warp right image/feature to left coordinate by horizontal disparity.

    right:  [B,C,H,W]
    disp_x: [B,1] or [B,1,1,1], in pixels of `right` resolution.

    Convention follows your cost-volume construction:
        left(x) matches right(x - disp)
    """
    b, c, h, w = right.shape
    device = right.device
    dtype = right.dtype

    if disp_x.dim() == 4:
        disp_x = disp_x.view(b, 1)
    elif disp_x.dim() == 2:
        pass
    else:
        disp_x = disp_x.view(b, 1)

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )

    xx = xx.unsqueeze(0).expand(b, -1, -1)
    yy = yy.unsqueeze(0).expand(b, -1, -1)

    x_sample = xx - disp_x.view(b, 1, 1)
    y_sample = yy

    if w > 1:
        x_norm = 2.0 * x_sample / (w - 1) - 1.0
    else:
        x_norm = torch.zeros_like(x_sample)

    if h > 1:
        y_norm = 2.0 * y_sample / (h - 1) - 1.0
    else:
        y_norm = torch.zeros_like(y_sample)

    grid = torch.stack([x_norm, y_norm], dim=-1)  # [B,H,W,2]

    warped = F.grid_sample(
        right,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    return warped


def _center_texture_variance(x, win: int = 9):
    """
    Simple local texture cue.
    x: [B,3,H,W]
    return: [B,1]
    """
    gray = x.mean(dim=1, keepdim=True)
    b, c, h, w = gray.shape

    wh = min(win, h)
    ww = min(win, w)

    h0 = max(h // 2 - wh // 2, 0)
    w0 = max(w // 2 - ww // 2, 0)
    h1 = h0 + wh
    w1 = w0 + ww

    patch = gray[:, :, h0:h1, w0:w1]
    var = patch.var(dim=(1, 2, 3), unbiased=False, keepdim=False).view(b, 1)
    return var


def _center_grad_energy(x, win: int = 9):
    """
    Simple gradient energy cue.
    x: [B,3,H,W]
    return: [B,1]
    """
    gray = x.mean(dim=1, keepdim=True)

    gx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    gy = gray[:, :, 1:, :] - gray[:, :, :-1, :]

    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))

    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return _center_window_mean(grad, win=win)


class StereopatchV2(nn.Module):
    """
    Predict center-point depth and confidence from left/right DP patches.

    Main changes versus original:
    1. Predict log-depth mean, then map to depth range [depth_min, depth_max].
    2. log_scale is the Laplace scale of log-depth error.
    3. Confidence is fused from:
       - probabilistic confidence from log_scale
       - stereo/DP reliability confidence from cost/consistency/texture cues
    """

    def __init__(
        self,
        maxdisp=24,
        depth_min=0.1,
        depth_max=10.0,
        default_conf_rel_thresh=0.02,
    ):
        super().__init__()

        self.maxdisp = maxdisp
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.log_depth_min = math.log(self.depth_min)
        self.log_depth_max = math.log(self.depth_max)
        self.default_conf_rel_thresh = float(default_conf_rel_thresh)

        self.feature = Feature()
        self.matching = Matching3D(in_ch=64)

        # Learnable temperature for softmin over disparity costs.
        self.log_tau = nn.Parameter(torch.tensor(0.0))

        disp_values = torch.arange(-maxdisp // 2, maxdisp // 2, dtype=torch.float32)
        self.register_buffer("disp_values", disp_values, persistent=False)

        # Existing cues:
        #   disp_mean, disp_log_var, disp_entropy, disp_peak
        # New cost cues:
        #   prob_margin, cost_margin
        cost_cue_dim = 6

        # center_logits_norm: [B,maxdisp]
        # center_feat: xl_center + yr_center = [B,64]
        in_dim = maxdisp + 64 + cost_cue_dim

        self.center_trunk = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # Predict raw log-depth location.
        self.log_depth_head = nn.Linear(64, 1)

        # Predict log Laplace scale for log-depth residual.
        self.log_scale_head = nn.Linear(64, 1)

        # Confidence calibration for probabilistic confidence only.
        # s_conf = a * s_raw + b, a > 0.
        self._conf_a_raw = nn.Parameter(torch.tensor(0.0))
        self._conf_b = nn.Parameter(torch.tensor(0.0))

        # Stereo/DP reliability branch.
        #
        # Inputs:
        # trunk: 64
        # mu_log_depth: 1
        # log_scale: 1
        # cost cues: 6
        # photo_err: 1
        # feat_err: 1
        # texture: 1
        # grad_energy: 1
        # dp_diff: 1
        conf_stereo_dim = 64 + 1 + 1 + 6 + 1 + 1 + 1 + 1 + 1

        self.conf_stereo_head = nn.Sequential(
            nn.Linear(conf_stereo_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        # Optional learned fusion.
        #
        # Inputs:
        # conf_prob: 1
        # conf_stereo: 1
        # disp_entropy, disp_peak, prob_margin, cost_margin: 4
        # photo_err, feat_err, texture, grad_energy, dp_diff: 5
        self.conf_fuse = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

        self._init_weights()

        # A reasonable initial scale for log-depth error.
        # exp(-3) ~= 0.05, roughly 5% log-depth scale.
        nn.init.constant_(self.log_scale_head.bias, -3.0)

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

        cost = torch.zeros(
            b,
            c * 2,
            d,
            h,
            w,
            device=xl_feat.device,
            dtype=xl_feat.dtype,
        )

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

    def _prob_conf_from_log_scale(
        self,
        log_scale,
        conf_rel_thresh,
        log_scale_min,
        log_scale_max,
    ):
        """
        Convert log-depth Laplace scale into:
            P(|log z_pred - log z_gt| <= log(1 + rel_thresh))
        """
        s_raw = torch.clamp(log_scale, min=log_scale_min, max=log_scale_max)

        conf_a = F.softplus(self._conf_a_raw) / 0.6931471805599453
        conf_b = self._conf_b

        s_conf = torch.clamp(
            conf_a * s_raw + conf_b,
            min=log_scale_min,
            max=log_scale_max,
        )

        delta = math.log1p(float(conf_rel_thresh))
        conf_prob = 1.0 - torch.exp(-delta * torch.exp(-s_conf))
        conf_prob = conf_prob.clamp(1e-4, 1.0 - 1e-4)

        return conf_prob, s_conf, conf_a, conf_b

    def forward(
        self,
        xl,
        yr,
        return_conf: bool = False,
        return_aux: bool = False,
        conf_rel_thresh: Optional[float] = None,
        log_scale_min: float = -6.0,
        log_scale_max: float = 3.0,
        use_learned_fusion: bool = False,
    ):
        if conf_rel_thresh is None:
            conf_rel_thresh = self.default_conf_rel_thresh

        xl_feat = self.feature(xl)
        yr_feat = self.feature(yr)

        cost = self._build_cost_volume(xl_feat, yr_feat)
        logits_3d = self.matching(cost).squeeze(1)  # [B,D,Hf,Wf]

        h_mid = logits_3d.shape[2] // 2
        w_mid = logits_3d.shape[3] // 2

        center_logits = logits_3d[:, :, h_mid, w_mid]  # [B,D]

        # Normalize center cost curve before feeding to MLP.
        center_logits_norm = (
            center_logits - center_logits.mean(dim=1, keepdim=True)
        ) / (center_logits.std(dim=1, keepdim=True, unbiased=False) + 1e-6)

        # Softmin cost distribution.
        tau = F.softplus(self.log_tau) + 1e-3
        center_prob = torch.softmax(-center_logits / tau, dim=1)

        disp = self.disp_values.to(center_prob.dtype).view(1, -1)

        disp_mean = torch.sum(center_prob * disp, dim=1, keepdim=True)
        disp_var = torch.sum(center_prob * (disp - disp_mean) ** 2, dim=1, keepdim=True)
        disp_log_var = torch.log(disp_var + 1e-6)

        disp_entropy = -torch.sum(
            center_prob * torch.log(center_prob + 1e-12),
            dim=1,
            keepdim=True,
        )
        disp_entropy = disp_entropy / math.log(float(self.maxdisp))

        disp_peak = torch.amax(center_prob, dim=1, keepdim=True)

        # Probability margin: top1 - top2.
        prob_top2 = torch.topk(center_prob, k=2, dim=1, largest=True).values
        prob_margin = prob_top2[:, 0:1] - prob_top2[:, 1:2]

        # Cost margin: second-best cost - best cost.
        # Since we use softmin, lower center_logits means better match.
        cost_top2 = torch.topk(center_logits, k=2, dim=1, largest=False).values
        cost_margin = cost_top2[:, 1:2] - cost_top2[:, 0:1]

        xl_center = xl_feat[:, :, h_mid, w_mid]
        yr_center = yr_feat[:, :, h_mid, w_mid]
        center_feat = torch.cat([xl_center, yr_center], dim=1)  # [B,64]

        cost_cues = torch.cat(
            [
                disp_mean,
                disp_log_var,
                disp_entropy,
                disp_peak,
                prob_margin,
                cost_margin,
            ],
            dim=1,
        )

        head_in = torch.cat(
            [
                center_logits_norm,
                center_feat,
                cost_cues,
            ],
            dim=1,
        )

        trunk = self.center_trunk(head_in)

        # Predict log-depth, then constrain depth to [depth_min, depth_max].
        raw_log_depth = self.log_depth_head(trunk)

        mu_log_depth = self.log_depth_min + (
            self.log_depth_max - self.log_depth_min
        ) * torch.sigmoid(raw_log_depth)

        depth = torch.exp(mu_log_depth)

        # log_scale is for log-depth residual.
        log_scale = self.log_scale_head(trunk)

        if not return_conf:
            return depth

        # Probabilistic confidence from log-scale.
        conf_prob, s_conf, conf_a, conf_b = self._prob_conf_from_log_scale(
            log_scale=log_scale,
            conf_rel_thresh=conf_rel_thresh,
            log_scale_min=log_scale_min,
            log_scale_max=log_scale_max,
        )

        # Feature consistency:
        # disp_mean is in feature-pixel unit.
        yr_feat_warp = _warp_right_to_left_x(yr_feat, disp_mean)
        feat_err_map = torch.abs(xl_feat - yr_feat_warp)
        feat_err = _center_window_abs_mean(feat_err_map, win=5)

        # Photometric consistency:
        # convert feature disparity to raw-pixel disparity.
        if xl_feat.shape[-1] > 1:
            scale_x = (xl.shape[-1] - 1) / float(xl_feat.shape[-1] - 1)
        else:
            scale_x = 1.0

        raw_disp_mean = disp_mean * scale_x

        yr_warp = _warp_right_to_left_x(yr, raw_disp_mean)
        photo_err_map = torch.abs(xl - yr_warp)
        photo_err = _center_window_abs_mean(photo_err_map, win=9)

        # Texture / gradient / raw DP difference cues.
        texture = _center_texture_variance(xl, win=9)
        grad_energy = _center_grad_energy(xl, win=9)
        dp_diff = _center_window_abs_mean(xl - yr, win=9)

        conf_stereo_in = torch.cat(
            [
                trunk,
                mu_log_depth,
                log_scale,
                cost_cues,
                photo_err,
                feat_err,
                texture,
                grad_energy,
                dp_diff,
            ],
            dim=1,
        )

        conf_stereo_logit = self.conf_stereo_head(conf_stereo_in)
        conf_stereo = torch.sigmoid(conf_stereo_logit).clamp(1e-4, 1.0 - 1e-4)

        if use_learned_fusion:
            fuse_in = torch.cat(
                [
                    conf_prob,
                    conf_stereo,
                    disp_entropy,
                    disp_peak,
                    prob_margin,
                    cost_margin,
                    photo_err,
                    feat_err,
                    texture,
                    grad_energy,
                    dp_diff,
                ],
                dim=1,
            )
            conf = torch.sigmoid(self.conf_fuse(fuse_in))
            conf = conf.clamp(1e-4, 1.0 - 1e-4)
        else:
            # More conservative and stable default:
            # confidence is high only when both uncertainty and stereo reliability agree.
            conf = torch.sqrt(conf_prob * conf_stereo)
            conf = conf.clamp(1e-4, 1.0 - 1e-4)

        if return_aux:
            aux = {
                "mu_log_depth": mu_log_depth,
                "raw_log_depth": raw_log_depth,
                "log_scale": log_scale,
                "s_conf": s_conf,

                "conf_prob": conf_prob,
                "conf_stereo": conf_stereo,
                "conf_stereo_logit": conf_stereo_logit,

                "disp_mean": disp_mean,
                "disp_log_var": disp_log_var,
                "disp_entropy": disp_entropy,
                "disp_peak": disp_peak,
                "prob_margin": prob_margin,
                "cost_margin": cost_margin,

                "photo_err": photo_err,
                "feat_err": feat_err,
                "texture": texture,
                "grad_energy": grad_energy,
                "dp_diff": dp_diff,

                "tau": tau.detach(),
                "conf_a": conf_a.detach(),
                "conf_b": conf_b.detach(),
            }
            return depth, conf, aux
            # return depth, conf_stereo, aux

        return depth, conf
        # return depth, conf_stereo
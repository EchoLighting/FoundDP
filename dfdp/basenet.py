""" Full assembly of the parts to form the complete network """

from .unet.unet_parts import *
from .unet.swin_reg import *
from torch.cuda.amp import autocast
from .dddnet.dddnet import YRStereonet_3D
from .unet.unet_model import UNet,ShiftUNet
class BaseNet(nn.Module):
    def __init__(self,):
        super(BaseNet, self).__init__()
        self.net = YRStereonet_3D()
        # self.net = ShiftUNet()
        # self.net = UNet()

    def forward(self, input_dict):
        stack_rgb = input_dict['stack_rgb_img']
        left, right = stack_rgb[:,0:3,:,:], stack_rgb[:,3:,:,:]

        outputs = dict()
        depth, aif = self.net(left, right)
        outputs['pred_depth_process'] = 1/(depth+1e-9)
        outputs['pred_depth'] = depth
        outputs['pred_AiF_img'] = aif
        losses, outputs = self.compute_loss(outputs, input_dict)
        return losses, outputs
    
    from torch import amp
    @amp.autocast(device_type="cuda")
    def fit(self,stack_rgb):
        # left, right = stack_rgb[:,0:3,:200,:300], stack_rgb[:,3:,:200,:300]
        left, right = stack_rgb[:,0:3,:,:], stack_rgb[:,3:,:,:]
        outputs = dict()
        depth, aif = self.net(left, right)
        outputs['pred_depth_process'] = 1/(depth+1e-9)
        outputs['pred_depth'] = depth
        return outputs
    
    def inference(self, input_dict):
        stack_rgb = input_dict['stack_rgb_img']
        outputs = self.fit(stack_rgb)
        outputs['pred_AiF_img'] = input_dict['AiF_img']
        return outputs
    
    def compute_loss(self, outputs, input_dict):
        losses = dict()
        l2 = nn.MSELoss(reduction='mean')
        l1 = nn.SmoothL1Loss(reduction='mean')

        gt_d, gt_AiF = input_dict['depth'], input_dict['AiF_img']
        gt_d_process = 1/(gt_d+1e-9)
        d_real, d_process, AiF =outputs['pred_depth'], outputs['pred_depth_process'], outputs['pred_AiF_img']

        # norm
        mask = gt_d > 1e-9
        mask.detach_()
        losses['depth'] = l1(d_process[mask],gt_d_process[mask])
        losses['depth_ori'] = l1(d_real[mask],gt_d[mask])

        # # AiF loss
        # losses['AiF'] = l2(AiF, gt_AiF) if AiF is not None else 0

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
        # d_gx1, d_gy1 = self.image_grads(d_process)
        # losses['smooth'] = (
        #     torch.mean(weights_x * self.robust_l1(d_gx1)) +
        #     torch.mean(weights_y * self.robust_l1(d_gy1))) / 2.
        
        # losses['total'] = losses['depth']
        losses['total'] = losses['depth_ori'] +0.1*losses['smooth_ori']

        return losses, outputs

    def image_grads(self, image_batch, stride=1):
        image_batch_gh = image_batch[..., stride:, :] - image_batch[
            ..., :-stride, :]
        image_batch_gw = image_batch[..., stride:] - image_batch[..., :-stride]
        return image_batch_gh, image_batch_gw

    def robust_l1(self, x):
        """Robust L1 metric."""
        return (x**2 + 0.001**2)**0.5

""" Full assembly of the parts to form the complete network """

from .unet.unet_parts import *
from .unet.swin_reg import *
from torch.cuda.amp import autocast
from .dddnet.dddnet import YRStereonet_3D, Mydeblur, UNetBlur, Myreblur
from .unet.unet_model import UNet,ShiftUNet
from pytorch_msssim import SSIM

class Basenet_Both(nn.Module):
    def __init__(self,):
        super(Basenet_Both, self).__init__()
        self.dfdp_net = YRStereonet_3D()
        self.deblur_net = Mydeblur()
        self.render_net = Myreblur()
        # self.deblur_net = UNetBlur(7,1,3)
        # self.render_net = UNetBlur(4,3,3)

        self.loss_l1_ssim = L1_SSIM_Loss(channel=3)
        
    from torch import amp
    @amp.autocast(device_type="cuda")
    def forward(self, input_dict):
        stack_rgb, gt_aif = input_dict['stack_rgb_img'], input_dict['AiF_img']
        left, right = stack_rgb[:,0:3,:,:], stack_rgb[:,3:,:,:]

        outputs = dict()
        depth_est, aif_est = self.dfdp_net(left, right)
        depth_fix, aif_fix = self.deblur_net(left, right, depth_est)
        pred_render_l, pred_render_r = self.render_net(gt_aif, input_dict['depth'])

        outputs['pred_depth_est'] = depth_est
        outputs['pred_depth_fix'] = depth_fix
        outputs['pred_aif'] = aif_fix
        outputs['pred_render_l'] = pred_render_l
        outputs['pred_render_r'] = pred_render_r

        losses, outputs = self.compute_loss(outputs, input_dict)
        return losses, outputs
    
    # @autocast()
    def fit(self,input_dict):
        stack_rgb, gt_aif = input_dict['stack_rgb_img'], input_dict['AiF_img']
        left, right = stack_rgb[:,0:3,:,:], stack_rgb[:,3:,:,:]

        outputs = dict()
        depth_est, aif_est = self.dfdp_net(left, right)
        depth_fix, aif_fix = self.deblur_net(left, right, depth_est)
        pred_render_l, pred_render_r = self.render_net(gt_aif, input_dict['depth'])

        outputs['pred_depth_est'] = depth_est
        outputs['pred_depth_fix'] = depth_fix
        outputs['pred_aif'] = aif_fix
        outputs['pred_render_l'] = pred_render_l
        outputs['pred_render_r'] = pred_render_r
        return outputs
    
    def inference(self, input_dict):
        outputs = self.fit(input_dict)
        return outputs
    
    def compute_loss(self, outputs, input_dict):
        losses = dict()
        l2 = nn.MSELoss(reduction='mean')
        l1 = nn.SmoothL1Loss(reduction='mean')

        gt_d, gt_AiF = input_dict['depth'], input_dict['AiF_img']
        pred_depth_est, pred_depth_fix, pred_AiF = outputs['pred_depth_est'], outputs['pred_depth_fix'], outputs['pred_aif']
        pred_render_l, pred_render_r = outputs['pred_render_l'], outputs['pred_render_r']
        stack_rgb_img = input_dict['stack_rgb_img']
        rt_render_l, rt_render_r = stack_rgb_img[:,0:3,:,:], stack_rgb_img[:,3:,:,:]

        # norm
        mask = gt_d > 1e-9
        mask.detach_()
        losses['depth_est'] = l1(pred_depth_est[mask],gt_d[mask])
        losses['depth_fix'] = l1(pred_depth_fix[mask],gt_d[mask])

        # AiF loss
        losses['aif'] = self.loss_l1_ssim(pred_AiF, gt_AiF)

        # render loss
        losses['render_l'] = self.loss_l1_ssim(pred_render_l, rt_render_l)
        losses['render_r'] = self.loss_l1_ssim(pred_render_r, rt_render_r)
        
        # losses['total'] = losses['depth']
        losses['total'] = losses['depth_est'] + \
              losses['depth_fix'] + losses['aif'] + \
              losses['render_l'] + losses['render_r']
        # losses['total'] = losses['render']
        return losses, outputs

    def image_grads(self, image_batch, stride=1):
        image_batch_gh = image_batch[..., stride:, :] - image_batch[
            ..., :-stride, :]
        image_batch_gw = image_batch[..., stride:] - image_batch[..., :-stride]
        return image_batch_gh, image_batch_gw

    def robust_l1(self, x):
        """Robust L1 metric."""
        return (x**2 + 0.001**2)**0.5

    def scale(self, input):
        scale_value = 0.5
        out = (input - scale_value)*2
        return out
    
    def inverse_scale(self, input):
        scale_value = 0.5
        out = input/2 + scale_value
        return out
    
# L1 + SSIM
class L1_SSIM_Loss(nn.Module):
    def __init__(self, channel=3, alpha=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIM(
            data_range=1.0, 
            win_size=11, 
            channel=channel, 
            nonnegative_ssim=True   
        )
        self.alpha = alpha

    def forward(self, pred, gt):
        ssim_loss = 1 - self.ssim(pred, gt)
        return self.l1(pred, gt) + self.alpha * ssim_loss
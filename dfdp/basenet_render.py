""" Full assembly of the parts to form the complete network """

from .unet.unet_parts import *
from .unet.swin_reg import *
from torch.cuda.amp import autocast
from .dddnet.dddnet import YRStereonet_3D, Mydeblur, UNetBlur, Myreblur
from .unet.unet_model import UNet,ShiftUNet
from pytorch_msssim import SSIM
import numpy as np
import xlsxwriter
import copy


class Basenet_Render(nn.Module):
    def __init__(self, train_mode='render'):
        super(Basenet_Render, self).__init__()
        self.train_mode=train_mode

        self.render_net = Myreblur()
        self.dfdp_net = YRStereonet_3D()
        self.deblur_net = Mydeblur()
        
    from torch import amp
    @amp.autocast(device_type="cuda")
    def forward(self, input_dict):
        if self.train_mode == 'render':
            losses, outputs = self.render_train(input_dict)
        elif self.train_mode == 'cycle':
            losses, outputs = self.cycle_train(input_dict)
        elif self.train_mode == 'direct':
            losses, outputs = self.direct_train(input_dict)
        return losses, outputs
    
    def render_train(self, input_dict, train=True):
        stack_rgb, gt_aif = self.scale(input_dict['stack_rgb_img']), self.scale(input_dict['AiF_img'])
        rt_render_l, rt_render_r = stack_rgb[:,0:3,:,:], stack_rgb[:,3:,:,:]
        gt_depth = self.linear(input_dict['depth'])

        pred_render_l, pred_render_r = self.render_net(gt_aif, gt_depth)

        losses = None
        if train is True:
            results, gts = dict(), dict()
            results['pred_render_l'] = pred_render_l
            results['pred_render_r'] = pred_render_r
            gts['gt_depth'] = gt_depth
            gts['rt_render_l'], gts['rt_render_r'] = rt_render_l, rt_render_r
            losses = self.compute_render_loss(results, gts)   

        outputs=dict()
        outputs['gt_depth'], outputs['gt_aif'] = self.inverse_linear(gt_depth), self.inverse_scale(gt_aif)
        outputs['rt_render_l'], outputs['rt_render_r'] = self.inverse_scale(rt_render_l), self.inverse_scale(rt_render_r)
        outputs['pred_render_l'], outputs['pred_render_r'] = self.inverse_scale(pred_render_l), self.inverse_scale(pred_render_r)
        outputs['pred_depth_est'] = None
        outputs['pred_depth_fix'], outputs['pred_aif'] = None, None
        outputs['reblur_render_l'], outputs['reblur_render_r'] = None, None
        return losses, outputs

    def cycle_train(self, input_dict, train=True):
        gt_aif = self.scale(input_dict['AiF_img'])
        gt_depth = self.linear(input_dict['depth'])

        with torch.no_grad():
            pred_render_l, pred_render_r = self.render_net(gt_aif, gt_depth)
            pred_render_l, pred_render_r = self.augmentation(pred_render_l, pred_render_r)
        depth_est, aif_est = self.dfdp_net(pred_render_l, pred_render_r)
        depth_fix, aif_fix = self.deblur_net(pred_render_l, pred_render_r, depth_est)
        reblur_render_l, reblur_render_r = self.render_net(aif_fix, depth_fix)

        losses = None
        if train is True:
            results, gts = dict(), dict()
            gts['gt_depth'], gts['gt_aif'] = gt_depth, gt_aif
            gts['pred_render_l'], gts['pred_render_r'] = pred_render_l, pred_render_r
            results['pred_depth_est'], results['pred_depth_fix'], results['pred_aif'] = depth_est, depth_fix, aif_fix
            results['reblur_render_l'], results['reblur_render_r'] = reblur_render_l, reblur_render_r
            losses = self.compute_cycle_loss(results, gts)

        outputs=dict()
        outputs['gt_depth'], outputs['gt_aif'] = self.inverse_linear(gt_depth), self.inverse_scale(gt_aif)
        outputs['rt_render_l'], outputs['rt_render_r'] = None, None
        outputs['pred_render_l'], outputs['pred_render_r'] = self.inverse_scale(pred_render_l), self.inverse_scale(pred_render_r)
        outputs['pred_depth_est'] = self.inverse_linear(depth_est.to(torch.float32))
        outputs['pred_depth_fix'], outputs['pred_aif'] = self.inverse_linear(depth_fix.to(torch.float32)), self.inverse_scale(aif_fix)
        outputs['reblur_render_l'], outputs['reblur_render_r'] = self.inverse_scale(reblur_render_l), self.inverse_scale(reblur_render_r)
        return losses, outputs
    
    def direct_train(self, input_dict, train=True):
        stack_rgb, gt_aif = self.scale(input_dict['stack_rgb_img']), self.scale(input_dict['AiF_img'])
        stack_rgb, gt_aif = input_dict['stack_rgb_img'], input_dict['AiF_img']
        rt_render_l, rt_render_r = stack_rgb[:,0:3,:,:], stack_rgb[:,3:,:,:]
        gt_depth = self.linear(input_dict['depth'])

        depth_est, aif_est = self.dfdp_net(rt_render_l, rt_render_r)
        depth_fix, aif_fix = self.deblur_net(rt_render_l, rt_render_r, depth_est)
        # depth_est, depth_fix = depth2disp(depth_est, d_max=depth_est.max()), depth2disp(depth_fix, d_max=depth_fix.max())

        losses = None
        if train is True:
            results, gts = dict(), dict()
            gts['gt_depth'], gts['gt_aif'] = gt_depth, gt_aif
            gts['rt_render_l'], gts['rt_render_r'] = rt_render_l, rt_render_r
            results['pred_depth_est'], results['pred_depth_fix'], results['pred_aif'] = depth_est, depth_fix, aif_fix
            # losses = self.compute_direct_loss(results, gts)

        outputs=dict()
        outputs['gt_depth'], outputs['gt_aif'] = self.inverse_linear(gt_depth,mask=True), self.inverse_scale(gt_aif)
        outputs['rt_render_l'], outputs['rt_render_r'] = self.inverse_scale(rt_render_l), self.inverse_scale(rt_render_r)
        outputs['pred_render_l'], outputs['pred_render_r'] = None, None
        outputs['pred_depth_est'] = self.inverse_linear(depth_est.to(torch.float32))
        # outputs['pred_depth_est'] = depth_est.to(torch.float32)
        outputs['pred_depth_fix'], outputs['pred_aif'] = self.inverse_linear(depth_fix.to(torch.float32)), self.inverse_scale(aif_fix)
        # outputs['pred_depth_fix'], outputs['pred_aif'] = depth_fix.to(torch.float32), aif_fix
        outputs['reblur_render_l'], outputs['reblur_render_r'] = None, None
        return losses, outputs

    def cycle_test(self, input_dict, train=False):
        gt_depth, gt_aif = self.linear(input_dict['depth']), self.scale(input_dict['AiF_img'])
        stack_rgb = self.scale(input_dict['stack_rgb_img'])
        gt_l, gt_r = stack_rgb[:,0:3,:,:], stack_rgb[:,3:,:,:]

        depth_est, aif_est = self.dfdp_net(gt_l, gt_r)
        depth_fix, aif_fix = self.deblur_net(gt_l, gt_r, depth_est)
        reblur_render_l, reblur_render_r = self.render_net(aif_fix, depth_fix)

        losses = None
        if train is True:
            results, gts = dict(), dict()
            gts['gt_depth'], gts['gt_aif'] = gt_depth, gt_aif
            gts['gt_l'], gts['gt_r'] = gt_l, gt_r
            results['pred_depth_est'], results['pred_depth_fix'], results['pred_aif'] = depth_est, depth_fix, aif_fix
            results['reblur_render_l'], results['reblur_render_r'] = reblur_render_l, reblur_render_r
            losses = self.compute_cycle_test_loss(results, gts)

        outputs=dict()
        outputs['gt_depth'], outputs['gt_aif'] = self.inverse_linear(gt_depth), self.inverse_scale(gt_aif)
        outputs['rt_render_l'], outputs['rt_render_r'] = None, None
        outputs['gt_l'], outputs['gt_r'] = self.inverse_scale(gt_l), self.inverse_scale(gt_r)
        outputs['pred_depth_est'] = self.inverse_linear(depth_est.to(torch.float32), mask=False)
        outputs['pred_depth_fix'], outputs['pred_aif'] = self.inverse_linear(depth_fix.to(torch.float32), mask=False), self.inverse_scale(aif_fix)
        outputs['reblur_render_l'], outputs['reblur_render_r'] = self.inverse_scale(reblur_render_l), self.inverse_scale(reblur_render_r)
        return losses, outputs
    
    def inference(self, input_dict, realdata=False):
        if realdata is True:
            losses, outputs = self.cycle_test(input_dict, train=False)
        elif self.train_mode == 'cycle':
            losses, outputs = self.cycle_train(input_dict, train=False)
        elif self.train_mode == 'render':
            losses, outputs = self.render_train(input_dict, train=False)
        elif self.train_mode == 'direct':
            losses, outputs = self.direct_train(input_dict, train=False)
        return outputs
    
    def compute_render_loss(self, results, gts):
        losses = dict()
        l2 = nn.MSELoss(reduction='mean')
        l1 = nn.SmoothL1Loss(reduction='mean')

        pred_render_l, pred_render_r = results['pred_render_l'], results['pred_render_r']
        rt_render_l, rt_render_r = gts['rt_render_l'], gts['rt_render_r']

        # render loss
        losses['render_l'] = l1(pred_render_l, rt_render_l)
        losses['render_r'] = l1(pred_render_r, rt_render_r)

        losses['total'] = losses['render_l'] + losses['render_r']
        return losses 

    def compute_cycle_loss(self, results, gts):
        losses = dict()
        l2 = nn.MSELoss(reduction='mean')
        l1 = nn.SmoothL1Loss(reduction='mean')

        gt_depth, gt_aif = gts['gt_depth'], gts['gt_aif']
        pred_render_l, pred_render_r = gts['pred_render_l'], gts['pred_render_r']
        pred_depth_est, pred_depth_fix, pred_aif = results['pred_depth_est'], results['pred_depth_fix'], results['pred_aif']
        reblur_render_l, reblur_render_r = results['reblur_render_l'], results['reblur_render_r']

        # norm
        losses['depth_est'] = l1(pred_depth_est[self.mask], gt_depth[self.mask])
        losses['depth_fix'] = l1(pred_depth_fix[self.mask], gt_depth[self.mask])
        # AiF loss
        losses['aif'] = l1(pred_aif, gt_aif)
        # render loss
        losses['reblur_l'] = l1(pred_render_l, reblur_render_l)
        losses['reblur_r'] = l1(pred_render_r, reblur_render_r)

        # smoothness
        abs_fn = lambda x: x**2
        edge_constant = 100.
        img_gx = self.image_grads(gt_aif,stride=2)
        weights_x = torch.exp(-torch.mean(abs_fn(edge_constant * img_gx), axis=1, keepdims=True))
        d_gx = self.image_grads(pred_depth_fix,stride=2)
        losses['smooth'] = torch.mean(weights_x * self.robust_l1(d_gx))

        losses['total'] = losses['depth_est']*2 + losses['depth_fix'] + \
            losses['aif'] + losses['reblur_l'] + losses['reblur_r'] + \
            losses['smooth']*0.1
        return losses 

    def compute_cycle_test_loss(self, results, gts):
        losses = dict()
        l2 = nn.MSELoss(reduction='mean')
        l1 = nn.SmoothL1Loss(reduction='mean')

        # gt_depth, gt_aif = gts['gt_depth'], gts['gt_aif']
        gt_l, gt_r = gts['gt_l'], gts['gt_r']
        pred_depth_est, pred_depth_fix, pred_aif = results['pred_depth_est'], results['pred_depth_fix'], results['pred_aif']
        reblur_render_l, reblur_render_r = results['reblur_render_l'], results['reblur_render_r']

        # norm
        # losses['depth_est'] = l1(pred_depth_est[self.mask], gt_depth[self.mask])
        # losses['depth_fix'] = l1(pred_depth_fix[self.mask], gt_depth[self.mask])
        # AiF loss
        # losses['aif'] = l1(pred_aif, gt_aif)
        # render loss
        losses['reblur_l'] = l1(gt_l, reblur_render_l)
        losses['reblur_r'] = l1(gt_r, reblur_render_r)

        losses['total'] = losses['aif'] + losses['reblur_l'] + losses['reblur_r']
        return losses 

    def compute_direct_loss(self, results, gts):
        losses = dict()
        l2 = nn.MSELoss(reduction='mean')
        l1 = nn.SmoothL1Loss(reduction='mean')

        gt_depth, gt_aif = gts['gt_depth'], gts['gt_aif']
        rt_render_l, rt_render_r = gts['rt_render_l'], gts['rt_render_r']
        pred_depth_est, pred_depth_fix, pred_aif = results['pred_depth_est'], results['pred_depth_fix'], results['pred_aif']
        pred_disp_est, pred_disp_fix = self.depth2disp(pred_depth_est, d_max=pred_depth_est.max()), self.depth2disp(pred_depth_fix, d_max=pred_depth_fix.max())

        # norm
        losses['depth_est'] = l1(pred_depth_est[self.mask], gt_depth[self.mask])
        losses['depth_fix'] = l1(pred_depth_fix[self.mask], gt_depth[self.mask])
        # AiF loss
        losses['aif'] = l1(pred_aif, gt_aif)

        losses['total'] = losses['depth_est']*2 + losses['depth_fix'] + losses['aif']
        return losses 

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
    
    def linear(self, depth):
        self.mask = depth > 0
        self.mask.detach_()
        depth[self.mask] = torch.log(depth[self.mask])
        return depth

    def inverse_linear(self, depth, mask=False):
        if mask == True:
            depth[self.mask] = torch.exp(depth[self.mask])
        else:
            depth = torch.exp(depth)
        return depth
    
    def image_grads(self, image_batch, stride=1):
        image_batch_gw = image_batch[..., stride:] - image_batch[..., :-stride]
        return image_batch_gw

    def noise(self, render_l, render_r):
        N, C, H, W = render_l.shape
        device=render_l.device
        noise_range = 0.05*np.random.rand() #because of scale
        noise_map_l = torch.randn_like(render_l, device=device) * noise_range
        noise_map_r = torch.randn_like(render_r, device=device) * noise_range

        range1, range2 = (np.random.rand()/2), (np.random.rand()/2+0.5)
        weight_l = torch.linspace(range1,range2,W, device=device)
        weight_l = weight_l.repeat(N,C,H,1)
        weight_r = torch.flip(weight_l,[-1])

        noise_l, noise_r = noise_map_l*weight_l, noise_map_r*weight_r
        render_l, render_r = render_l+noise_l, render_r+noise_r
        return render_l, render_r
    
    def gamma(self, render_l, render_r):
        gamma_l = 0.1*np.random.rand()+0.95
        gamma_r = 0.1*np.random.rand()+0.95
        render_l, render_r = render_l**gamma_l, render_r**gamma_r
        return render_l, render_r

    def augmentation(self, render_l, render_r):
        render_l, render_r = self.inverse_scale(render_l), self.inverse_scale(render_r)
        render_l, render_r = torch.clip(render_l, 0, 1.0), torch.clip(render_r, 0, 1.0)
        render_l, render_r = self.gamma(render_l, render_r)
        render_l, render_r = self.noise(render_l, render_r)
        render_l, render_r = torch.clip(render_l, 0, 1.0), torch.clip(render_r, 0, 1.0)
        render_l, render_r = self.scale(render_l), self.scale(render_r)
        return render_l, render_r

    def depth2disp(self, depth, d_max=20, d_min=0.2):
        """Convert depth to disparity (inverse depth)"""
        a = 1/d_max
        b = 1/d_min - 1/d_max
        disp = (1/depth-a)/b    
        return disp

# L1 + SSIM
class L1_SSIM_Loss(nn.Module):
    def __init__(self, channel=3, alpha=0.1):
        super().__init__()
        self.l1 = nn.SmoothL1Loss()
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
    

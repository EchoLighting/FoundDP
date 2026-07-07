import os
import sys
sys.dont_write_bytecode = True
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
ddp = False ##CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=gpu  --master_port=12357 FoundDP_train.py
if ddp == False:
    os.environ["CUDA_VISIBLE_DEVICES"]='3'
import logging

import yaml
import time
import cv2 as cv
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from dfdp.dual_pixel_disparity.utils.metrics import *
from dfdp.basenet_render import Basenet_Render
from dfdp.dddnet.dddnet import Mydeblur
from deeplens.utils import set_seed, set_logger
from deeplens.psfnet import *
from dfdp.depth_conf import Stereonet_3D, DepthFusionC2Plus
from dfdp import get_lens, get_dataset, select_focus_dist
from dfdp import *
from monocular.depth_anything_v2.dpt import DepthAnythingV2
from monocular.founddp.founddp import FoundDP
from dfdp.depth_conf import compute_gradients
from dfdp.dual_pixel_disparity.utils.metrics import *

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.CRITICAL)

def ddp_setup():
    os.environ["MASTER_ADDR"] = "localhost"
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def config():
    with open('configs/dfdp_by_sdirt_rf50mm.yml') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    
    # Device
    num_gpus = torch.cuda.device_count()
    args['num_gpus'] = num_gpus
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    args['device'] = device
    logging.info(f'Using {num_gpus} GPUs')

    # Result folder
    result_dir = f'./results/' + datetime.now().strftime("%m%d-%H%M%S") + '-FoundDP_train'
    args['results_dir'] = result_dir
    os.makedirs(result_dir, exist_ok=True)
    logging.info(f'Result folder: {result_dir}')
    
    # Random seed
    set_seed(123456)
    torch.set_default_dtype(torch.float32)
    
    return args

def train(args):
    if ddp : ddp_setup()
    id = 0 if ddp is False else int(os.environ['LOCAL_RANK'])
    if id == 0:  # Logger
        set_logger(args['results_dir'])

    device = args['device']
    train_mode = args.get('train_mode', 'dfdp')  # 'dfdp', 'deblur', 'founddp'
    data_mode = args.get('data_mode', 'simulation')  # 'simulation' or 'real'

    # Lens
    train_lens, test_lens = get_lens(args)

    # Models
    dfdp_net = Stereonet_3D()
    dfdp_net = dfdp_net.to(device)
    if ddp:
        dfdp_net = DDP(dfdp_net, device_ids=[id], find_unused_parameters=True)
    else:
        dfdp_net = nn.DataParallel(dfdp_net)
    
    if 'dfdpnet_pretrained' in args['train'].keys():
        net_dict = dfdp_net.state_dict()
        pretrain_dict = torch.load(args['train']['dfdpnet_pretrained'], weights_only=True)
        update_dict = {}
        for k, v in pretrain_dict.items():
            k1 = k
            if k1 in net_dict and net_dict[k1].shape == pretrain_dict[k].shape:
                update_dict[k1] = v
        net_dict.update(update_dict)
        dfdp_net.load_state_dict(net_dict)

    torch.cuda.empty_cache()
    dfdp_net = dfdp_net.to(device)

    deblur_net = Mydeblur()
    deblur_net = deblur_net.to(device)
    if ddp:
        deblur_net = DDP(deblur_net, device_ids=[id], find_unused_parameters=True)
    else:
        deblur_net = nn.DataParallel(deblur_net)
    
    if 'ours_deblur_pretrained' in args['train'].keys():
        net_dict = deblur_net.state_dict()
        pretrain_dict = torch.load(args['train']['ours_deblur_pretrained'], weights_only=True)
        update_dict = {}
        for k, v in pretrain_dict.items():
            k1 = k
            if k1 in net_dict and net_dict[k1].shape == pretrain_dict[k].shape:
                update_dict[k1] = v
        net_dict.update(update_dict)
        deblur_net.load_state_dict(net_dict)

    torch.cuda.empty_cache()
    deblur_net = deblur_net.to(device)

    model = None
    if train_mode == 'founddp':
        model = FoundDP.from_pretrained('./ckpt/founddp.ckpt').to(device)
        if 'ours_pretrained' in args['train'].keys():
            net_dict = model.state_dict()
            pretrain_dict = torch.load(args['train']['ours_pretrained'], map_location='cpu', weights_only=True)
            update_dict = {}
            for k, v in pretrain_dict.items():
                k1 = k
                if k1 in net_dict and net_dict[k1].shape == pretrain_dict[k].shape:
                    update_dict[k1] = v
            net_dict.update(update_dict)
            model.load_state_dict(net_dict)
        torch.cuda.empty_cache()
        model = model.to(device)

    # Dataset
    train_set, val_set = get_dataset(args)
    val_loader = DataLoader(val_set, batch_size=1)
    print(f'Totally {len(train_set)} images for training, {len(val_set)} images for test.')
    if ddp == False:
        train_loader = DataLoader(train_set, batch_size=args['bs'], num_workers=4, pin_memory=True, shuffle=True, drop_last=True)
    else:
        train_loader = DataLoader(dataset=train_set, batch_size=args['bs'], shuffle=False, pin_memory=True, drop_last=True, sampler=DistributedSampler(train_set, drop_last=True))

    # Optimizer
    if train_mode == 'dfdp':
        params_to_optimize = dfdp_net.parameters()
    elif train_mode == 'deblur':
        params_to_optimize = deblur_net.parameters()
    elif train_mode == 'founddp':
        params_to_optimize = model.parameters()
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    optimizer = optim.AdamW(params_to_optimize, lr=float(args['lr']))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['epochs'] * len(train_set), eta_min=0)
    scaler = torch.amp.GradScaler('cuda')

    args['mse_Middlebury_FS_min'] = 100
    args['acc1_Middlebury_FS_max'] = 0.0
    args['ai1_max'] = 1.0

    epoch = 0

    # Training
    for epoch in range(args['epochs'] + 1):
        if ddp: train_loader.sampler.set_epoch(epoch)

        # Evaluation
        if epoch % 1 == 0 and id == 0 and epoch > 0:
            with torch.no_grad():
                validate(dfdp_net, deblur_net, model, test_lens, val_loader, epoch, len(val_set), 'Middlebury_FS', args, train_mode, data_mode)

        # Training
        if train_mode == 'dfdp':
            dfdp_net.train()
        elif train_mode == 'deblur':
            deblur_net.train()
        elif train_mode == 'founddp':
            model.train()

        for sample in tqdm(train_loader, dynamic_ncols=True):
            if data_mode == 'simulation':
                # Input data
                aif, depth_ori = sample
                aif = aif.to(device)
                depth = depth_ori.to(device)
                depth = depth * 1.0 - test_lens.d_sensor / 1e3

                # Render focal stack
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        focus_dists = select_focus_dist(depth, args['n_stack'], mode='linear')
                        focus_dists -= train_lens.d_sensor / 1e3
                        focal_stack = []
                        for i in range(args['bs']):
                            foc_dist = focus_dists[i:i+1, 0]
                            defocus_img = train_lens.render(aif[i:i+1], depth=-depth[i:i+1]*1e3, foc_dist=-foc_dist*1e3, train=True)
                            focal_stack.append(defocus_img)
                            torch.cuda.empty_cache()
                        focal_stack = torch.cat(focal_stack, dim=0)
                        xl, yr = focal_stack[:, 0:3, :, :], focal_stack[:, 3:, :, :]
                    torch.cuda.empty_cache()
                
            elif data_mode == 'real':
                img_l, img_r, disp, img_b, depth = sample
                xl = img_l.to(device)
                yr = img_r.to(device)
                # disp_ori = disp.to(device)
                depth = depth.to(device)
                aif = img_b.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                if train_mode == 'dfdp':
                    pred_depth = dfdp_net(xl, yr)
                elif train_mode == 'deblur':
                    pred_depth = dfdp_net(xl, yr)
                    pred_depth, _ = deblur_net(xl, yr, pred_depth)
                elif train_mode == 'founddp':
                    with torch.no_grad():
                        d_metrix = dfdp_net.module.infer_image(xl, yr)
                        d_metrix, _ = deblur_net(xl, yr, d_metrix)
                        d_metrix = d_metrix.float()
                    final_depth = model(aif, d_metrix)
                    pred_depth = final_depth

                Si = nn.SmoothL1Loss()
                with torch.no_grad():
                    depth = map_depth_to_range(depth)
                    mask = (depth > 1.0) & (depth < 10.0)
                pred_depth = map_depth_to_range(pred_depth)
                loss = Si(torch.log10(pred_depth[mask]), torch.log10(depth[mask]))

                assert torch.isnan(loss).sum() == 0, print(loss)

            scaler.scale(loss).backward()
            if train_mode == 'dfdp':
                clip_grad_norm_(parameters=dfdp_net.parameters(), max_norm=1.0)
            elif train_mode == 'deblur':
                clip_grad_norm_(parameters=deblur_net.parameters(), max_norm=1.0)
            elif train_mode == 'founddp':
                clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
    
    print("finish")

@torch.no_grad()
def validate(net_F, net_M, net_P, test_lens: PSFNet, valid_dataloader, epoch, num_val, scene, args, train_mode, data_mode):
    if train_mode == 'dfdp':
        net_F.eval()
    elif train_mode == 'deblur':
        net_M.eval()
    elif train_mode == 'founddp':
        net_P.eval()

    result_img_dir = f'{args["results_dir"]}/results/'
    os.makedirs(result_img_dir, exist_ok=True)
    device = args['device']

    if train_mode == 'dfdp':
        torch.save(net_F.state_dict(), f'{args["results_dir"]}/dfdp_net_last.pth')
    elif train_mode == 'deblur':
        torch.save(net_M.state_dict(), f'{args["results_dir"]}/deblur_net_last.pth')
    elif train_mode == 'founddp':
        torch.save(net_P.state_dict(), f'{args["results_dir"]}/founddp_net_last.pth')

    Avg_abs_rel = 0.0
    Avg_sq_rel = 0.0
    Avg_mse = 0.0
    Avg_mae = 0.0
    Avg_rmse = 0.0
    Avg_rmse_log = 0.0
    Avg_accuracy_1 = 0.0
    Avg_accuracy_2 = 0.0
    Avg_accuracy_3 = 0.0
    ai_1 = 0
    ai_2 = 0
    sp = 0
    ai1_ct = 0
    ai_2_ct = 0
    sp_ct = 0
    Avg_accuracy_1_ct = 0.0
    Avg_accuracy_2_ct = 0.0
    Avg_accuracy_3_ct = 0.0
    val_time = 0.0

    for idx, samples in enumerate(tqdm(valid_dataloader, desc="valid")):
        if data_mode == 'simulation':
            aif, depth_ori = samples
            aif = aif.to(device)
            gt_depth = depth_ori.to(device)
            focus_dists = select_focus_dist(gt_depth, args['n_stack'], mode='linear')
            focus_dists -= test_lens.d_sensor / 1e3
            focal_stack = []
            for i in range(args['bs']):
                foc_dist = focus_dists[i:i+1, 0]
                defocus_img = test_lens.render(aif[i:i+1], depth=-gt_depth[i:i+1]*1e3, foc_dist=-foc_dist*1e3, train=True)
                focal_stack.append(defocus_img)
                torch.cuda.empty_cache()
            focal_stack = torch.cat(focal_stack, dim=0)
            xl, yr = focal_stack[:, 0:3, :, :], focal_stack[:, 3:, :, :]
            torch.cuda.empty_cache()

        elif data_mode == 'real':
            img_l, img_r, disp, img_b, depth = samples
            xl = img_l.to(device)
            yr = img_r.to(device)
            # disp_ori = disp.to(device)
            gt_depth = depth.to(device)
            aif = img_b.to(device)

        start = time.time()
        with torch.no_grad():
            if train_mode == 'dfdp':
                pred_depth = net_F.module.infer_image(xl, yr)
            elif train_mode == 'deblur':
                pred_depth = net_F.module.infer_image(xl, yr)
                pred_depth, _ = net_M(xl, yr, pred_depth)
            elif train_mode == 'founddp':
                d_metrix = net_F.module.infer_image(xl, yr)
                d_metrix, _ = net_M(xl, yr, d_metrix)
                d_metrix = d_metrix.float()
                pred_depth = net_P.predict(aif, d_metrix)

        val_time = val_time + (time.time() - start)
        gt_depth = map_depth_to_range(gt_depth)
        pred_depth = map_depth_to_range(pred_depth)

        gt_disp_ = np.squeeze(gt_depth.data.cpu().numpy())
        pred_disp_ = np.squeeze(pred_depth.data.cpu().numpy())

        test_mask = (gt_disp_ > 1.0) & (gt_disp_ < 10.0)

        Avg_abs_rel = Avg_abs_rel + mask_abs_rel(pred_disp_, gt_disp_, test_mask)
        Avg_sq_rel = Avg_sq_rel + mask_sq_rel(pred_disp_, gt_disp_, test_mask)
        Avg_mse = Avg_mse + mask_mse(pred_disp_, gt_disp_, test_mask)
        Avg_mae = Avg_mae + mask_mae(pred_disp_, gt_disp_, test_mask)
        Avg_rmse = Avg_rmse + mask_rmse(pred_disp_, gt_disp_, test_mask)
        Avg_rmse_log = Avg_rmse_log + mask_rmse_log(pred_disp_, gt_disp_, test_mask)
        Avg_accuracy_1 = Avg_accuracy_1 + mask_accuracy_k(pred_disp_, gt_disp_, 1, test_mask)
        Avg_accuracy_2 = Avg_accuracy_2 + mask_accuracy_k(pred_disp_, gt_disp_, 2, test_mask)
        Avg_accuracy_3 = Avg_accuracy_3 + mask_accuracy_k(pred_disp_, gt_disp_, 3, test_mask)
        ai1, _ = affine_invariant_1(pred_disp_, gt_disp_, confidence_map=test_mask)
        ai_1 = ai_1 + ai1
        ai2, _ = affine_invariant_2(pred_disp_, gt_disp_, confidence_map=test_mask)
        ai_2 = ai_2 + ai2
        sp = sp + 1 - abs(spearman_correlation(pred_disp_, gt_disp_, W=test_mask))

        depth_max = gt_disp_.max()
        save_image(aif, f'{result_img_dir}/{scene}_img{idx}_A_gt_aif.png', normalize=False)
        colorbar(gt_disp_, depth_max, f'{result_img_dir}/{scene}_img{idx}_B_gt_disp.png')
        save_image(xl, f'{result_img_dir}/{scene}_img{idx}_C_gt_l.png', normalize=False)
        save_image(yr, f'{result_img_dir}/{scene}_img{idx}_D_gt_r.png', normalize=False)
        colorbar(pred_disp_, depth_max, f'{result_img_dir}/{scene}_img{idx}_F_pred_disp_with_bar.png')

        mask = weak_texture_mask_gradient(np.squeeze((255*aif).data.cpu().numpy().astype(np.uint8)).transpose(1, 2, 0))
        test_mask = ((gt_disp_ * mask) > 1.0) & ((gt_disp_ * mask) < 10.0)
        if test_mask.sum() == 0:
            continue
        ai1, _ = affine_invariant_1(pred_disp_, gt_disp_, confidence_map=test_mask)
        ai1_ct = ai1_ct + ai1
        ai2, _ = affine_invariant_2(pred_disp_, gt_disp_, confidence_map=test_mask)
        ai_2_ct = ai_2_ct + ai2
        sp_ct = sp_ct + 1 - abs(spearman_correlation(pred_disp_, gt_disp_, W=test_mask))
        Avg_accuracy_1_ct = Avg_accuracy_1_ct + mask_accuracy_k(pred_disp_, gt_disp_, 1, test_mask)
        Avg_accuracy_2_ct = Avg_accuracy_2_ct + mask_accuracy_k(pred_disp_, gt_disp_, 2, test_mask)
        Avg_accuracy_3_ct = Avg_accuracy_3_ct + mask_accuracy_k(pred_disp_, gt_disp_, 3, test_mask)

    logging.info(f"Avg_mse({epoch}): {Avg_mse / num_val}, {Avg_mae / num_val}")
    logging.info(f"Avg_acc({epoch}): {Avg_accuracy_1 / num_val}, {Avg_accuracy_2 / num_val}, {Avg_accuracy_3 / num_val}")
    logging.info(f"Avg_ai1({epoch}): {ai_1 / num_val}, Avg_ai2({epoch}): {ai_2 / num_val}, Avg_1-spcc({epoch}): {sp / num_val}")
    logging.info(f"In weak texture regions: Avg_ai1({epoch}): {ai1_ct / num_val}, Avg_ai2({epoch}): {ai_2_ct / num_val}, Avg_1-spcc({epoch}): {sp_ct / num_val}")
    logging.info(f"In weak texture regions: Avg_acc({epoch}): {Avg_accuracy_1_ct / num_val}, {Avg_accuracy_2_ct / num_val}, {Avg_accuracy_3_ct / num_val}")

    if Avg_mse / num_val < args[f'mse_{scene}_min']:
        args[f'mse_{scene}_min'] = Avg_mse / num_val
        if train_mode == 'dfdp':
            torch.save(net_F.state_dict(), f'{args["results_dir"]}/val_depth_{scene}_dfdp_net_best.pth')
        elif train_mode == 'deblur':
            torch.save(net_M.state_dict(), f'{args["results_dir"]}/val_depth_{scene}_deblur_net_best.pth')
        elif train_mode == 'founddp':
            torch.save(net_P.state_dict(), f'{args["results_dir"]}/val_depth_{scene}_founddp_net_best.pth')
    if Avg_accuracy_1 / num_val > args[f'acc1_{scene}_max']:
        args[f'acc1_{scene}_max'] = Avg_accuracy_1 / num_val
        if train_mode == 'dfdp':
            torch.save(net_F.state_dict(), f'{args["results_dir"]}/val_depth_{scene}_dfdp_net_best_acc1.pth')
        elif train_mode == 'deblur':
            torch.save(net_M.state_dict(), f'{args["results_dir"]}/val_depth_{scene}_deblur_net_best_acc1.pth')
        elif train_mode == 'founddp':
            torch.save(net_P.state_dict(), f'{args["results_dir"]}/val_depth_{scene}_founddp_net_best_acc1.pth')
    if ai_1 / num_val < args['ai1_max']:
        args['ai1_max'] = ai_1 / num_val
        if train_mode == 'dfdp':
            torch.save(net_F.state_dict(), f'{args["results_dir"]}/val_depth_{scene}_dfdp_net_best_ai1.pth')
        elif train_mode == 'deblur':
            torch.save(net_M.state_dict(), f'{args["results_dir"]}/val_depth_{scene}_deblur_net_best_ai1.pth')
        elif train_mode == 'founddp':
            torch.save(net_P.state_dict(), f'{args["results_dir"]}/val_depth_{scene}_founddp_net_best_ai1.pth')

def colorbar(img, vmax, dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(img, cmap="jet", vmin=1.0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.06)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Meter', fontsize=24)
    cbar.ax.tick_params(labelsize=24)
    plt.savefig(dir, bbox_inches='tight', dpi=300)
    plt.close(fig)

def map_depth_to_range(depth, mask=None, d_min=1.0, d_max=10.0, eps=1e-6):
    out = torch.zeros_like(depth)
    if mask is None:
        valid = torch.ones_like(depth, dtype=torch.bool)
    else:
        valid = mask
    if not torch.any(valid):
        return out
    depth_valid = depth[valid]
    src_min = depth_valid.min()
    src_max = depth_valid.max()
    out[valid] = (depth[valid] - src_min) / (src_max - src_min + eps) * (d_max - d_min) + d_min

    return out

def weak_texture_mask_gradient(
    img_rgb,
    grad_ksize=3,
    smooth_ksize=15,
    grad_thresh=10,
    morph_ksize=20
):
    """
    Generate weak-texture mask based on local gradient energy.

    Args:
        img_rgb (np.ndarray): HxWx3 RGB image (uint8 or float).
        grad_ksize (int): Sobel kernel size.
        smooth_ksize (int): Window size for local averaging.
        grad_thresh (float): Threshold for weak texture.
        morph_ksize (int): Kernel size for morphological ops.

    Returns:
        weak_mask (np.ndarray): HxW binary mask, 1 = weak texture.
        grad_energy (np.ndarray): Gradient magnitude map (float).
    """

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=grad_ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=grad_ksize)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_energy = cv2.GaussianBlur(
        grad_mag,
        (smooth_ksize, smooth_ksize),
        sigmaX=0
    )
    weak_mask = (grad_energy < grad_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize)
    )
    weak_mask = cv2.morphologyEx(weak_mask, cv2.MORPH_OPEN, kernel)
    weak_mask = cv2.morphologyEx(weak_mask, cv2.MORPH_CLOSE, kernel)

    return weak_mask

if __name__ == "__main__":
    args = config()
    # Set train_mode in args or modify config file
    args['train_mode'] = 'dfdp'  # Change to 'deblur' or 'founddp' as needed
    args['data_mode'] = 'simulation'  # Change to 'real' if using real data
    train(args)
    destroy_process_group()
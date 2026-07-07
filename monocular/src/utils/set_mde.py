import torch
import torchvision.transforms as T


def set_depthanything(encoder='vits'):
    """
    Initialize the DepthAnythingV2 model with a specified encoder type.

    Args:
        encoder (str): Encoder type. Options are 'vitl', 'vitb', or 'vits'.

    Returns:
        DepthAnythingV2: Initialized model.
    """
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    }

    if encoder not in model_configs:
        raise ValueError(f"Invalid encoder type: {encoder}. Choose from 'vitl', 'vitb', or 'vits'.")

    from src.models.depth_anything_v2.dpt import DepthAnythingV2
    config = model_configs[encoder]

    depth_anything = DepthAnythingV2(**config)
    # checkpoint_path = f"checkpoints/depth_anything_v2_{encoder}.pth"

    checkpoint_path = f"/home/xjj/code/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth"
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    return depth_anything


def set_depthanything_metric(encoder='vits', dataset='hypersim', max_depth=20):
    """
    Initialize the DepthAnythingV2 model with metric configuration.

    Args:
        encoder (str): Encoder type. Options are 'vitl', 'vitb', or 'vits'.
        dataset (str): Dataset type. Options are 'hypersim' or 'vkitti'.
        max_depth (int): Maximum depth value. 20 for indoor model, 80 for outdoor model.

    Returns:
        DepthAnythingV2: Initialized model.
    """
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    }

    if encoder not in model_configs:
        raise ValueError(f"Invalid encoder type: {encoder}. Choose from 'vitl', 'vitb', or 'vits'.")
    if dataset not in ['hypersim', 'vkitti']:
        raise ValueError(f"Invalid dataset type: {dataset}. Choose from 'hypersim' or 'vkitti'.")

    from src.models.depth_anything_v2_metric.dpt import DepthAnythingV2
    config = {**model_configs[encoder], 'max_depth': max_depth}

    depth_anything = DepthAnythingV2(**config)
    checkpoint_path = f"checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth"
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    return depth_anything


transforms_cfg = {
    "zju_s": {
        "trans": T.Compose([T.Resize((252, 322))]),
        "feat_trans": T.Compose([T.Resize((240, 320))]),
        "inv_trans": T.Compose([T.Resize((480, 640))]),
    },
    "zju_l":{
        "trans": T.Compose([T.Resize((490, 630))]),
        "feat_trans": T.Compose([T.Resize((240, 320))]),
        "inv_trans": T.Compose([T.Resize((480, 640))]),
    },
    "real": {
        "trans": T.Compose([T.Resize((924, 700))]),
        "feat_trans": T.Compose([T.Resize((464, 352))]),
        "inv_trans": T.Compose([T.Resize((928, 704))]),
    }
}

def compute_rel_depth(model, x,
                      trans=T.Compose([T.Resize((252, 322))]),
                      feat_trans=T.Compose([T.Resize((240, 320))]),
                      inv_trans=T.Compose([T.Resize((480, 640))])):
    img = trans(x)
    with torch.no_grad():
        model.eval()
        feat, inv_depth = model(img)

    inv_depth = inv_trans(inv_depth)
    feat = feat_trans(feat)
    rel_depth = 1 / (inv_depth + 0.1)

    rel_max = torch.amax(rel_depth, dim=(1, 2), keepdim=True)
    rel_min = torch.amin(rel_depth, dim=(1, 2), keepdim=True)
    inv_max = torch.amax(inv_depth, dim=(1, 2), keepdim=True)
    inv_min = torch.amin(inv_depth, dim=(1, 2), keepdim=True)

    rel_depth = (rel_depth - rel_min) / (rel_max - rel_min)
    inv_depth = (inv_depth - inv_min) / (inv_max - inv_min)

    return feat, rel_depth.unsqueeze(1), inv_depth.unsqueeze(1)


def compute_metric_depth(model, x,
                      trans=T.Compose([T.Resize((252, 322))]),
                      feat_trans=T.Compose([T.Resize((240, 320))]),
                      inv_trans=T.Compose([T.Resize((480, 640))])):
    img = trans(x)
    with torch.no_grad():
        model.eval()
        feat, rel_depth = model(img)

    rel_depth = inv_trans(rel_depth)
    feat = feat_trans(feat)
    inv_depth = 1 / (rel_depth + 0.1)

    rel_max = torch.amax(rel_depth, dim=(1, 2), keepdim=True)
    rel_min = torch.amin(rel_depth, dim=(1, 2), keepdim=True)
    inv_max = torch.amax(inv_depth, dim=(1, 2), keepdim=True)
    inv_min = torch.amin(inv_depth, dim=(1, 2), keepdim=True)

    rel_depth = (rel_depth - rel_min) / (rel_max - rel_min)
    inv_depth = (inv_depth - inv_min) / (inv_max - inv_min)

    return feat, rel_depth.unsqueeze(1), inv_depth.unsqueeze(1)
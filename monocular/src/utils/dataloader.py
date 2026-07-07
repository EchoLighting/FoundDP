import torch
import torch.nn
import random
import numpy as np
import cv2
import torch.nn.functional as F


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)


def get_region_empty(depth, region_height, region_width):
    c, h, w = depth.shape
    random_height = random.randint(2, 6) * region_height  # for zju
    random_width = random.randint(2, 6) * region_width    # for zju
    start_x = random.randint(0, h - random_height)
    start_y = random.randint(0, w - random_width)
    depth[:, start_x:start_x + random_height, start_y:start_y + random_width] = 0
    return depth


def get_rotate_translate(image, angle, offset):
    # image: C, H, W
    assert image.ndim == 3, "Input image must have shape (C, H, W)"
    c, h, w = image.shape

    # convert to H,W,C for OpenCV
    image = np.transpose(image, (1, 2, 0))  # (H, W, C)
    # C==1
    image = image.squeeze()

    min_rotation_angle = -angle
    max_rotation_angle = angle
    min_translation_offset = -offset
    max_translation_offset = offset

    # rotate
    rotation_angle = random.uniform(min_rotation_angle, max_rotation_angle)
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    # translate
    translation_offset_x = random.uniform(min_translation_offset, max_translation_offset)
    translation_offset_y = random.uniform(min_translation_offset, max_translation_offset)
    translation_matrix = np.float32([[1, 0, translation_offset_x], [0, 1, translation_offset_y]])
    translated_image = cv2.warpAffine(rotated_image, translation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    # C==1
    if translated_image.ndim == 2:
        translated_image = np.expand_dims(translated_image, axis=-1)  # (H, W, 1)

    # convert to C, H, W
    translated_image = np.transpose(translated_image, (2, 0, 1))  # (C, H, W)

    return translated_image



def get_sparse_depth_real(depth, hsv, front_depth=None):
    c, h, w = depth.shape

    # # for 480*640
    # zone_num = [30, 40]
    # x_l, x_r = np.random.randint(15, 65), np.random.randint(50, 65)
    # y_u, y_p = np.random.randint(20, 35), np.random.randint(5, 45)

    # for 928*704
    zone_num = [40, 30]
    y_u, y_p = np.random.randint(15, 93), np.random.randint(70, 93)
    x_l, x_r = np.random.randint(26, 46), np.random.randint(1, 59)

    x_d = np.linspace(0+x_l, w-x_r, zone_num[1])[None].repeat(zone_num[0], axis=0).ravel()
    y_d = np.linspace(0+y_u, h-y_p, zone_num[0])[..., None].repeat(zone_num[1], axis=1).ravel()

    x_noise = np.random.randint(-1, 2, size=x_d.shape)
    y_noise = np.random.randint(-1, 2, size=y_d.shape)
    x_d = x_d + x_noise
    y_d = y_d + y_noise

    idx_mask = (x_d < w) & (x_d > 0) & (y_d < h) & (y_d > 0)
    x_d = x_d[idx_mask].astype('int')
    y_d = y_d[idx_mask].astype('int')

    sparse_depth = np.zeros_like(depth)
    mask = np.zeros_like(depth)
    mask[:, y_d, x_d] = 1
    index = np.nonzero(get_rotate_translate(mask, 1.5, 0))
    sparse_depth[index] = depth[index]

    large_random_map, small_random_map = np.random.rand(1, h, w), np.random.rand(1, h, w)
    binomial_map = np.random.binomial(1, 0.5, h * w).reshape(1, h, w)
    mask = ((sparse_depth > 8.1) | ((sparse_depth < 8.1) & (binomial_map > 0) & (small_random_map < 0.1)))
    sparse_depth[mask] = 0

    v_channel = hsv[2, :, :].numpy()
    rgb_random_map = np.random.rand(1, h, w)
    mask = (sparse_depth > 0) & (v_channel[None] < 40/255) & (rgb_random_map > 0.2)
    sparse_depth[mask] = 0

    # random error
    error_random_map = np.random.rand(1, h, w)
    mask = (sparse_depth > 0) & (error_random_map < 0.03)
    random_factor = np.random.uniform(0.5, 1.5, size=sparse_depth[mask].shape)
    sparse_depth[mask] = random_factor * sparse_depth[mask] + np.random.rand(1)
    sparse_depth[mask] = np.clip(sparse_depth[mask], 0, 8.1)

    # region shift
    if front_depth != None:
        depth_value = front_depth
    else:
        depth_value = np.quantile(depth, np.random.uniform(0.2, 0.4))

    direction = np.random.choice([-1, 1], 2, replace=True)
    translation = direction * np.array([(h-y_p-y_u)/zone_num[0], (w-x_r-x_l)/zone_num[1]]) * np.random.randint(0, 3, 2)
    mask = sparse_depth > depth_value
    warp_region = sparse_depth * mask
    warp_region = \
          cv2.warpAffine(warp_region.reshape(h, w, 1), np.array([[1, 0, translation[0]], [0, 1, translation[1]]]), (w, h), flags=cv2.INTER_NEAREST)[None]
    sparse_depth = sparse_depth * ~mask + warp_region

    return torch.from_numpy(sparse_depth)


def get_sparse_depth_nyu(depth, max_depth):
    c, h, w = depth.shape  # 1, 480, 640

    zone_num = [30, 40]
    x_l, x_r = np.random.randint(15, 65), np.random.randint(50, 65)
    y_u, y_p = np.random.randint(20, 35), np.random.randint(5, 45)

    x_d = np.linspace(0+x_l, w-x_r, zone_num[1])[None].repeat(zone_num[0], axis=0).ravel()
    y_d = np.linspace(0+y_u, h-y_p, zone_num[0])[..., None].repeat(zone_num[1], axis=1).ravel()

    x_noise = np.random.randint(-1, 2, size=x_d.shape)
    y_noise = np.random.randint(-1, 2, size=y_d.shape)
    x_d = x_d + x_noise
    y_d = y_d + y_noise

    idx_mask = (x_d < w) & (x_d > 0) & (y_d < h) & (y_d > 0)
    x_d = x_d[idx_mask].astype('int')
    y_d = y_d[idx_mask].astype('int')

    sparse_depth = np.zeros_like(depth)
    mask = np.zeros_like(depth)
    mask[:, y_d, x_d] = 1
    index = np.nonzero(get_rotate_translate(mask, 1.5, 0))
    sparse_depth[index] = depth[index]

    return torch.from_numpy(sparse_depth)


def get_sparse_depth_zju(depth, max_depth):
    c, h, w = depth.shape  # 1, 480, 640
    zone_num = [8, 8]

    # for zju
    x_l, x_r = np.random.randint(75, 95), np.random.randint(95, 115)
    y_u, y_p = np.random.randint(-40, -15), np.random.randint(55, 70)

    x_d = np.linspace(0+x_l, w-x_r, zone_num[1])[None].repeat(zone_num[0], axis=0).ravel()
    y_d = np.linspace(0+y_u, h-y_p, zone_num[0])[..., None].repeat(zone_num[1], axis=1).ravel()
    y_d = np.clip(y_d, 0, h)

    x_noise = np.random.randint(-1, 2, size=x_d.shape)
    y_noise = np.random.randint(-1, 2, size=y_d.shape)
    x_d = x_d + x_noise
    y_d = y_d + y_noise

    idx_mask = (x_d < w) & (x_d > 0) & (y_d < h) & (y_d > 0)
    x_d = x_d[idx_mask].astype('int')
    y_d = y_d[idx_mask].astype('int')

    region_height, region_width = (h-y_p-y_u)//zone_num[0], (w-x_r-x_l)//zone_num[1]

    if np.random.rand() < 1:
        depth = get_region_empty(depth, region_height, region_width)

    sparse_depth = np.zeros_like(depth)
    mask = np.zeros_like(depth)
    mask[:, y_d, x_d] = 1
    index = np.nonzero(get_rotate_translate(mask, 1.5, 0))
    sparse_depth[index] = depth[index]

    # random error point
    error_random_map = np.random.rand(1, h, w)
    error_value_map = np.random.rand(1, h, w) * max_depth
    mask = (sparse_depth > 0) & (error_random_map < 0.05)
    sparse_depth[mask] = error_value_map[mask]

    # random lose point
    lose_random_map = np.random.rand(1, h, w)
    mask = (sparse_depth > 0) & (lose_random_map < 0.05)
    sparse_depth[mask] = 0

    return torch.from_numpy(sparse_depth)


def dtof_to_sparse_depth(dtof, fr, mask):
    mean, var = dtof[:, 0], dtof[:, 1]

    center = ((fr[:, :2] + fr[:, 2:]) / 2).long()
    center = torch.stack([
        torch.clip(center[:, 0], 0, 480),
        torch.clip(center[:, 1], 0, 640)
    ], dim=1)

    sparse_depth = torch.zeros((1, 480, 640))

    valid_mask = mask > 0
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]  # 获取有效索引
    sparse_depth[:, center[valid_indices, 0], center[valid_indices, 1]] = mean[valid_indices]

    return sparse_depth


def get_sparse_depth_mipi(dep):
    # Generate Grid points
    channel, img_h, img_w = dep.shape
    assert channel == 1

    stride = np.random.randint(40, 60)

    dist_coef = np.random.rand() * 4e-5 + 1e-5
    noise = np.random.rand() * 0.3

    x_odd, y_odd = np.meshgrid(np.arange(stride // 2, img_h, stride * 2), np.arange(stride // 2, img_w, stride))
    x_even, y_even = np.meshgrid(np.arange(stride // 2 + stride, img_h, stride * 2),
                                 np.arange(stride, img_w, stride))
    x_u = np.concatenate((x_odd.ravel(), x_even.ravel()))
    y_u = np.concatenate((y_odd.ravel(), y_even.ravel()))
    x_c = img_h // 2 + np.random.rand() * 50 - 25
    y_c = img_w // 2 + np.random.rand() * 50 - 25
    x_u = x_u - x_c
    y_u = y_u - y_c

    # Distortion
    r_u = np.sqrt(x_u ** 2 + y_u ** 2)
    r_d = r_u + dist_coef * r_u ** 3
    num_d = r_d.size
    sin_theta = x_u / r_u
    cos_theta = y_u / r_u
    x_d = np.round(r_d * sin_theta + x_c + np.random.normal(0, noise, num_d))
    y_d = np.round(r_d * cos_theta + y_c + np.random.normal(0, noise, num_d))
    idx_mask = (x_d < img_h) & (x_d > 0) & (y_d < img_w) & (y_d > 0)
    x_d = x_d[idx_mask].astype('int')
    y_d = y_d[idx_mask].astype('int')

    spot_mask = np.zeros((img_h, img_w))
    spot_mask[x_d, y_d] = 1

    dep_sp = torch.zeros_like(dep)
    dep_sp[:, x_d, y_d] = dep[:, x_d, y_d]

    return dep_sp

import torch
import os
import random
import numpy as np
from copy import deepcopy
import h5py
import torch.utils.data.distributed
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0,1"
import cv2
from ..utils.dataloader import seed_worker, get_rotate_translate, get_sparse_depth_real, get_sparse_depth_zju, get_sparse_depth_mipi


def distance2depth(intWidth, intHeight, fltFocal, npyDistance):
    npyImageplaneX = np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth).reshape(1, intWidth).repeat(intHeight, 0).astype(np.float32)[:, :, None]
    npyImageplaneY = np.linspace((-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight).reshape(intHeight,1).repeat(intWidth, 1).astype(np.float32)[:, :, None]
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate([npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2)
    npyDepth = npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal
    return npyDepth


def compute_fake_depth(image_hsv, depth, diffuse, semantic):
    image_hsv_np = np.array(image_hsv)
    depth_np = np.array(depth)
    diffuse_np = np.array(diffuse)
    semantic_np = np.array(semantic)

    gray = image_hsv_np[:, :, 2]
    # FOR WINDOWS
    mask_windows = (semantic_np == 9) & (diffuse_np < 10) & (np.random.rand(*semantic_np.shape) < 0.95)
    depth_np[mask_windows] = 0

    # for transparent area
    mask_transparent = (semantic_np != 19) & (semantic_np != 9) & (semantic_np != 2) & (diffuse_np < 10) & (gray > 110) & (np.random.rand(*semantic_np.shape) < 0.95)

    random_factor = np.random.uniform(0.5, 1.5, size=depth_np[mask_transparent].shape)
    depth_np[mask_transparent] = random_factor * depth_np[mask_transparent] + np.random.rand(1)

    fake_depth = Image.fromarray(depth_np.astype('float32'), mode='F')
    return fake_depth


class HYPERSIM(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode)

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   worker_init_fn=seed_worker)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode)
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=11,
                                   pin_memory=False)


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = deepcopy(args)
        self.args.mode = mode

        if mode == 'online_eval':
            md = 'test'
        else:
            md = 'train'

        if md != 'test':
            fname = args.filenames_file
            with open(fname, 'r') as fh:
                pairs = []
                for line in fh:
                    line = line.rstrip()
                    words = line.split()
                    pairs.append((words[0], words[1]))
                self.sample_list = pairs
        elif md == 'test':
            fname = args.filenames_file_eval
            with open(fname, 'r') as fh:
                pairs = []
                for line in fh:
                    line = line.rstrip()
                    words = line.split()
                    pairs.append((words[0], words[1]))

                self.sample_list = pairs

        self.mode = mode
        self.transform = transform
        self.is_for_online_eval = is_for_online_eval


    def __getitem__(self, idx):
        if self.args.mode == 'online_eval':
            input, target = self.sample_list[idx]  # input = image target = depth_gt
            scene_name = input.split('/')[0]
            input = os.path.join(self.args.data_path_eval, input)
            target = os.path.join(self.args.data_path_eval, target)

        elif self.args.mode == 'train':
            input, target = self.sample_list[idx]
            scene_name = input.split('/')[0]
            input = os.path.join(self.args.data_path, input)
            target = os.path.join(self.args.data_path, target)

        semantic_path = target.replace('depth_meters', 'semantic')
        diffuse_path = input.replace('tonemap', 'diffuse_reflectance')

        # diffuse = cv2.resize(cv2.imread(diffuse_path), (704, 928)) # rev
        diffuse = cv2.resize(cv2.imread(diffuse_path), (640, 480))
        diffuse = cv2.cvtColor(diffuse, cv2.COLOR_BGR2GRAY)
        with h5py.File(semantic_path, 'r') as f:
            semantic = f['dataset'][:]
        # semantic = cv2.resize(semantic, (704, 928), interpolation=cv2.INTER_NEAREST) # rev
        semantic = cv2.resize(semantic, (640, 480), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(cv2.imread(input), cv2.COLOR_BGR2RGB)
        with h5py.File(target, 'r') as f:
            depth_gt = f['dataset'][:]

        depth_gt = np.array(depth_gt)
        depth_gt = distance2depth(intWidth=1024, intHeight=768, fltFocal=886.81, npyDistance=depth_gt)

        count_greater_than_6 = np.sum(depth_gt > 6)
        total_elements = depth_gt.size
        ratio = count_greater_than_6/total_elements
        if ratio > 0.6:
            depth_gt = depth_gt / 2

        depth_gt[depth_gt > self.args.max_depth] = self.args.max_depth
        depth_gt[depth_gt < self.args.min_depth] = self.args.min_depth
        depth_gt[np.isnan(depth_gt)] = self.args.max_depth

        image = Image.fromarray(image, mode='RGB')
        depth_gt = Image.fromarray(depth_gt.astype('float32'), mode='F')
        image_hsv = image.convert('HSV')

        # image = image.resize((704, 928))         # rev
        # depth_gt = depth_gt.resize((704, 928))   # rev
        # image_hsv = image_hsv.resize((704, 928)) # rev
        image = image.resize((640, 480))
        depth_gt = depth_gt.resize((640, 480))
        image_hsv = image_hsv.resize((640, 480))

        diffuse = Image.fromarray(diffuse)
        semantic = Image.fromarray(semantic)
        fake_depth = compute_fake_depth(image_hsv, depth_gt, diffuse, semantic)

        if self.mode == 'train':
            image = np.array(image) / 255.0
            image_hsv = np.array(image_hsv, dtype=np.float32) / 255.0

            image = torch.from_numpy(image.transpose(2, 0, 1))
            image_hsv = torch.from_numpy(image_hsv.transpose(2, 0, 1))

            depth_gt = np.array(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=0)

            fake_depth = np.array(fake_depth, dtype=np.float32)
            fake_depth = np.expand_dims(fake_depth, axis=0)

            sample = {'image': image, 'depth': depth_gt}

        else:
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1))

            image_hsv = np.array(image_hsv, dtype=np.float32) / 255.0
            image_hsv = torch.from_numpy(image_hsv.transpose(2, 0, 1))

            depth_gt = np.array(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=0)

            fake_depth = np.array(fake_depth, dtype=np.float32)
            fake_depth = np.expand_dims(fake_depth, axis=0)

            fname = self.sample_list[idx][0]

            image_path = fname[fname.rfind('/') + 1:].replace('h5', 'jpg')
            image_folder = fname[:fname.rfind('/')]

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': True,
                          'image_path': image_path, 'image_folder': image_folder}
            else:
                sample = {'image': image}

        sample['path'] = target

        # real dataset
        # sparse_depth = get_sparse_depth_real(torch.from_numpy(get_rotate_translate(fake_depth, 0.9, 12)).clone(), image_hsv)

        # zju degrade
        sparse_depth = get_sparse_depth_zju(torch.from_numpy(get_rotate_translate(fake_depth, 1.8, 24)).clone(), max_depth=self.args.max_depth)

        # mipi degrade
        # sparse_depth = get_sparse_depth_mipi(torch.from_numpy(depth_gt.transpose(2, 0, 1)).clone())

        sample['sparse_depth'] = sparse_depth

        return sample

    def __len__(self):
        return len(self.sample_list)

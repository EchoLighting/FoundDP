import h5py
import numpy as np
from copy import deepcopy
import torch
import torch.utils.data.distributed
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

from ..utils.dataloader import seed_worker, get_sparse_depth_nyu


def remove_border(image):
    width, height = image.size

    border = 15
    left = border
    top = border
    right = width - border
    bottom = height - border

    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize((width, height), Image.LANCZOS)

    return resized_image


class NYUV2(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode,)

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   worker_init_fn=seed_worker)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode,)
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=6,
                                   pin_memory=False)


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, is_for_online_eval=False):
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
                    # print(line)
                    # words = line.split()
                    pairs.append(line)
                self.sample_list = pairs
        elif md == 'test':
            fname = args.filenames_file_eval
            with open(fname, 'r') as fh:
                pairs = []
                for line in fh:
                    line = line.rstrip()
                    pairs.append(line)
                self.sample_list = pairs

        self.mode = mode
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        path_file = self.sample_list[idx]

        f = h5py.File(path_file, 'r')
        if 'raw' in f.keys():
            raw_h5 = f['raw'][:]
            dep_h5 = f['depth'][:]
            rgb_h5 = f['rgb'][:]
        else:
            rgb_h5 = f['rgb'][:].transpose(1, 2, 0)
            dep_h5 = f['depth'][:]

        dep_h5[dep_h5 > self.args.max_depth] = self.args.max_depth
        dep_h5[dep_h5 < self.args.min_depth] = self.args.min_depth

        image = Image.fromarray(rgb_h5, mode='RGB')
        depth_gt = Image.fromarray(dep_h5.astype('float32'), mode='F')
        # image = remove_border(image)
        # depth_gt = remove_border(depth_gt)

        # image = image.resize((320, 240))
        # depth_gt = depth_gt.resize((320, 240))

        # left = (320 - 304) / 2
        # top = (240 - 228) / 2
        # right = left + 304
        # bottom = top + 228
        #
        # image = image.crop((left, top, right, bottom))
        # depth_gt = depth_gt.crop((left, top, right, bottom))

        # image = image.crop((304, 228))
        # depth_gt = depth_gt.crop((304, 228))

        if self.mode == 'train':
            depth_gt = np.array(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=0)

            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1))

            sample = {'image': image, 'depth': depth_gt}

        else:
            depth_gt = np.array(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=0)

            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1))

            fname = self.sample_list[idx]
            image_path = fname[fname.rfind('/') + 1:].replace('h5', 'jpg')
            image_folder = fname[:fname.rfind('/')]

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': True, 'image_path': image_path, 'image_folder': image_folder}
            else:
                sample = {'image': image}

        if 'raw' in f.keys():
            raw_depth = Image.fromarray(raw_h5.astype('float32'), mode='F')
            raw_depth  = np.array(raw_depth, dtype=np.float32)
            raw_depth = np.expand_dims(raw_depth, axis=0)

            sparse_depth = get_sparse_depth_nyu(torch.from_numpy(raw_depth).clone(), max_depth=self.args.max_depth)
            sample['sparse_depth'] = sparse_depth
        else:
            sparse_depth = get_sparse_depth_nyu(torch.from_numpy(depth_gt).clone(), max_depth=self.args.max_depth)
            sample['sparse_depth'] = sparse_depth

        return sample

    def __len__(self):
        return len(self.sample_list)
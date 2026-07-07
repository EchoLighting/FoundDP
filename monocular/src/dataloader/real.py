import torch
import os
import random
import numpy as np
from copy import deepcopy
import torch.utils.data.distributed
from PIL import Image, ImageOps
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0,1"
import cv2
from ..utils.dataloader import seed_worker


class REAL(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(args, mode,)
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   worker_init_fn=seed_worker)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(args, mode,)
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=0,
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
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        if self.args.mode == 'online_eval':
            input, target = self.sample_list[idx]  # input = image target = depth_gt
            input = os.path.join(self.args.data_path_eval, input)
            target = os.path.join(self.args.data_path_eval, target)
        elif self.args.mode == 'train':
            input, target = self.sample_list[idx]
            input = os.path.join(self.args.data_path, input)
            target = os.path.join(self.args.data_path, target)

        image = cv2.imread(input)
        depth_gt = np.load(target)/1000

        # depth_gt[depth_gt > self.args.max_depth] = self.args.max_depth
        # depth_gt[depth_gt < self.args.min_depth] = self.args.min_depth

        image = Image.fromarray(image, mode='RGB')
        depth_gt = Image.fromarray(depth_gt.astype('float32'), mode='F')

        image = image.rotate(-90, expand=True)
        pad_width = (704 - image.size[0]) // 2
        pad_height = (928 - image.size[1]) // 2

        image = ImageOps.expand(image, border=(pad_width, pad_height), fill=(0, 0, 0))
        depth_gt = ImageOps.expand(depth_gt, border=(pad_width, pad_height), fill=0)

        if self.mode == 'train':
            depth_gt = np.array(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=0)

            image = np.array(image, dtype=np.float32) / 255.0

            sample = {'image': image, 'depth': depth_gt}

        else:
            depth_gt = np.array(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=0)

            image = np.array(image, dtype=np.float32) / 255.0

            fname = self.sample_list[idx][0]
            image_path = fname[fname.rfind('/') + 1:].replace('h5', 'jpg')
            image_folder = fname[:fname.rfind('/')]

            if self.mode == 'online_eval':
                sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': True,
                          'image_path': image_path, 'image_folder': image_folder}
            else:
                sample = {'image': image}

        sparse_depth = torch.from_numpy(depth_gt.transpose(2, 0, 1))
        sample['sparse_depth'] = sparse_depth
        return sample

    def __len__(self):
        return len(self.sample_list)
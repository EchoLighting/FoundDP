import os
import json
import random
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ..utils.dataloader import seed_worker, dtof_to_sparse_depth


class ZJU(object):
    def __init__(self, args, mode):
        if mode == 'train':
            self.dataset = DataLoadPreprocess(args, mode,)
            self.data = DataLoader(
                self.dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_threads,
                pin_memory=True
            )
        elif mode == 'online_eval':
            self.dataset = DataLoadPreprocess(args, mode='test',)
            self.data = DataLoader(
                self.dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                pin_memory=False
            )


class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None):
        self.args = args
        self.mode = mode
        data_file = args.filenames_file_eval
        with open(data_file, 'r') as f:
            data = json.load(f)
        self.sample_list = data[mode]
        self.histogram_gt = np.zeros(100)
        self.histogram_l5 = np.zeros(100)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        file_entry = self.sample_list[idx]
        h5_file_path = os.path.join(self.args.data_path_eval, file_entry["filename"])
        with h5py.File(h5_file_path, 'r') as f:
            image = np.array(f['rgb'])
            depth_gt = np.array(f['depth'])
            l5 = np.array(f['hist_data'])
            fr = np.array(f['fr'])
            mask = np.array(f['mask'])


        depth_gt[depth_gt > self.args.max_depth] = self.args.max_depth
        # depth_gt[depth_gt < self.args.min_depth] = self.args.min_depth
        depth_gt[np.isnan(depth_gt)] = self.args.max_depth

        image = Image.fromarray(image)
        depth_gt = Image.fromarray(depth_gt.astype('float32'), mode='F')

        if self.mode == 'train':
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1))

            depth_gt = np.array(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=0)

            sample = {'image': image, 'depth': depth_gt}

        else:
            image = np.array(image, dtype=np.float32) / 255.0
            image = torch.from_numpy(image.transpose(2, 0, 1))

            depth_gt = np.array(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=0)

            fname = self.sample_list[idx]['filename']

            image_path = fname[fname.rfind('/') + 1:].replace('h5', 'jpg')
            image_folder = fname[:fname.rfind('/')]

            sample = {'image': image, 'depth': depth_gt, 'has_valid_depth': True,
                      'image_path': image_path, 'image_folder': image_folder}

        sparse_depth = dtof_to_sparse_depth(torch.from_numpy(l5).float(), torch.from_numpy(fr), torch.from_numpy(mask))

        dtof = torch.from_numpy(l5).float()
        mask = torch.from_numpy(mask).float()

        sample['sparse_depth'] = sparse_depth
        sample['dtof'] = dtof[:,0].reshape(1,8,8) * mask.reshape(1,8,8)
        return sample
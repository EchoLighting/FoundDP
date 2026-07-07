import torch
from torch.utils.data import Dataset
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2 as cv
import numpy as np
from glob import glob
from torchvision import transforms
from skimage.morphology import disk, closing
import random
from scipy.ndimage.interpolation import rotate
import torch.nn.functional as F
from typing import Dict
from torch import Tensor
from pathlib import Path
import warnings
import imageio.v3 as iio
from typing import Union
from numpy import ndarray
import re
from PIL import Image
import copy
import cv2

# ================================
# Dataset
# ================================

class NYUData(Dataset):
    def __init__(self, rgb_path, resize=None, train=True, render=False):
        super(NYUData, self).__init__()

        self.rgb_path = rgb_path
        self.depth_path = rgb_path
        self.scenes = glob(f'{rgb_path}/*')
        self.resize = resize
        self.train = train
        self.render = render
        
        self.imgs = []
        self.depths = []
        for scene in self.scenes:
            imgs = sorted(glob(f'{scene}/*.jpg'))
            depths = sorted(glob(f'{scene}/*.png'))
            self.imgs += imgs
            self.depths += depths

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, antialias=True)
        ])
        self.scale = 25.5
        self.crop = 20

    def __len__(self):
        if self.train is True:
            return 2000
        elif self.render is True:
            return 2000
        else:
            return 50
        
    def __getitem__(self, idx):
        if self.train==True:
            idx = np.random.randint(0, high=len(self.imgs))
        try:
            aif_img = cv.cvtColor(cv.imread(self.imgs[idx]), cv.COLOR_BGR2RGB) / 255.
            depth = cv.imread(self.depths[idx], -1) / self.scale  #* 1e3 # convert to [mm]
            h,w,c = aif_img.shape
            aif_img = aif_img[self.crop:(h-self.crop),self.crop:(w-self.crop),:]
            depth = depth[self.crop:(h-self.crop),self.crop:(w-self.crop)]
            assert (depth[depth>0].any()==True)
        except:
            print(f"fail file {self.depths[idx]}")
            return self.__getitem__(idx+1)

        if self.train:
            aif_img, depth = AutoAgument(aif_img, depth)
        depth = depth_preprocess(depth)
        
        aif_img = self.transform(aif_img.astype('float32'))
        depth = self.transform(depth.astype('float32'))
        if self.render:
            return aif_img
        else:
            return [aif_img, depth]

class FlyingThings3D(Dataset):
    def __init__(self, dataset_dir, resize=None, train=False, fs_num=0, render=False):
        super(FlyingThings3D, self).__init__()

        self.dataset_dir = dataset_dir
        self.scenes = [scene.split('/')[-1] for scene in glob(f'{dataset_dir}/*')]
        self.resize = resize
        self.fs_num = fs_num
        self.train = train
        self.render = render

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, antialias=True)
        ])

    def __len__(self):
        # return 400
        lenth = len(self.scenes)
        if self.train is True:
            lenth = lenth
        elif self.render is True:
            lenth = lenth   
        else:
            lenth = 50
        return lenth

    def __getitem__(self, index):
        scene = self.scenes[index]
        dataset_dir = self.dataset_dir
        DEPTH_FACTOR = 20
        resize = [self.resize[1], self.resize[0]]

        depth = cv.resize(cv.imread(f'{dataset_dir}/{scene}/disp.exr', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH) / DEPTH_FACTOR, resize) #m
        
        if self.fs_num > 0:
            focused_imgs = []
            focal_dists = []
            full_focal_stack = sorted(glob(f'{dataset_dir}/{scene}/*.png'))[:-1]
            selected_imgs = random.sample(full_focal_stack, self.fs_num)
            for img_name in selected_imgs:
                focal_dists.append(float(img_name.split('/')[-1][:-4]) / DEPTH_FACTOR)
                focused_img = cv.resize(cv.imread(img_name).astype(np.float32)/255., resize)
                focused_imgs.append(focused_img)
            
            focal_stack = np.stack(focused_imgs, axis=-1)
            
            if self.train:
                focal_stack, depth = AutoAgument(focal_stack, depth)
                
            focal_stack = np.transpose(focal_stack, (3, 2, 0, 1))   # shape of (S, C, H, W)
            focal_stack = torch.from_numpy(focal_stack.astype('float32'))
            depth = torch.from_numpy(depth.astype('float32')).unsqueeze(0)
            focal_dists = torch.from_numpy(np.stack(focal_dists, axis=-1))  

            return [focal_stack, depth, focal_dists]

        else:
            aif_img = cv.cvtColor(cv.imread(f'{dataset_dir}/{scene}/AiF.png'), cv.COLOR_BGR2RGB) / 255.
            
            if self.train:
                aif_img, depth = AutoAgument(aif_img, depth)
            depth = depth_preprocess(depth)

            aif_img = self.transform(aif_img.astype('float32'))
            depth = self.transform(depth.astype('float32'))
            if self.render:
                return aif_img  
            else:
                return [aif_img, depth]

class Middlebury_FS(Dataset):
    def __init__(self, dataset_dir, resize=None, train=False, fs_num=0):
        super(Middlebury_FS).__init__()

        self.dataset_dir = dataset_dir
        self.scenes = [scene.split('/')[-1] for scene in glob(f'{dataset_dir}/*')]
        self.resize = resize
        self.fs_num = fs_num
        self.train = train

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, antialias=True)
        ])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        scene = self.scenes[index]
        DEPTH_FACTOR = 10   # convert disparity to depth
        resize = [self.resize[1], self.resize[0]]

        depth = cv.resize(cv.imread(f'{dataset_dir}/{scene}/disp.exr', cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH) / DEPTH_FACTOR, resize)

        if self.fs_num > 0:
            raise Exception('Untested.')
            focused_imgs = []
            focal_dists = []
            full_focal_stack = sorted(glob(f'{dataset_dir}/{scene}/*.png'))[:-1]
            for _ in range(self.fs_num):
                focused_img = random.choice(full_focal_stack)
                focal_dists.append(float(focused_img.split('/')[-1][:-4]) / DEPTH_FACTOR)
                focused_img = cv.resize(cv.imread(focused_img).astype(np.float32)/255., resize)
                focused_imgs.append(focused_img)
            
            focal_stack = np.stack(focused_imgs, axis=-1)
            
            if self.train:
                focal_stack, depth = AutoAgument(focal_stack, depth)
                
            focal_stack = np.transpose(focal_stack, (3, 2, 0, 1))
            focal_stack = torch.from_numpy(focal_stack.astype('float32'))
            depth = torch.from_numpy(depth.astype('float32')).unsqueeze(0)
            focal_dists = torch.from_numpy(np.stack(focal_dists, axis=-1))
            
            return [focal_stack, depth, focal_dists]

        else:
            aif_img = cv.cvtColor(cv.imread(f'{dataset_dir}/{scene}/AiF.png'), cv.COLOR_BGR2RGB) / 255.
            
            if self.train:
                aif_img, depth = AutoAgument(aif_img, depth)
            depth[depth<0] = 0
            aif_img = self.transform(aif_img.astype('float32'))
            depth = self.transform(depth.astype('float32'))

            return [aif_img, depth]


class Middlebury(Dataset):
    def __init__(self, dataset_dir, resize=None, train=False):
        super(Middlebury).__init__()

        self.dataset_dir = dataset_dir
        self.scenes = sorted([scene.split('/')[-1] for scene in glob(f'{dataset_dir}/*')])
        self.resize = resize
        self.train = train

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, antialias=True)
        ])

        self.test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        scene = self.scenes[index]
        resize = [self.resize[1], self.resize[0]]

        aif_img = cv.cvtColor(cv.imread(f'{dataset_dir}/{scene}/im0.png'), cv.COLOR_BGR2RGB) / 255.
        calib_data = self.read_calib(f'{dataset_dir}/{scene}/calib.txt')
        disparity = imread(f'{dataset_dir}/{scene}/disp0.pfm')
        disparity = np.abs(disparity)
        disparity[disparity == float('inf')] = 0
        depth = disparity + calib_data['doffs']
        depth = calib_data['baseline'] * calib_data['cam0']['f'] / depth
        depth /= 1e3
        depth = depth.numpy()

        depth = cv.resize(depth, resize)


        aif_img = self.train_transform(aif_img.astype('float32'))
        depth = self.train_transform(depth.astype('float32'))

        return [aif_img, depth]
    
    @staticmethod
    def read_calib(calib_path) -> Dict:
        calib_data = {}
        with open(calib_path, 'r') as f:
            lines = f.readlines()

        for line in lines[:2]:
            k, v = line.split('=')
            v = v[1:-1].split(';')
            f, _, cx = v[0].split()
            cy = v[1].split()[2]
            calib_data[k] = {'f': float(f), 'cx': float(cx), 'cy': float(cy)}

        for line in lines[2:]:
            k, v = line.split('=')
            v = v.strip()
            if v.isdigit():
                v = int(v)
            else:
                v = float(v)
            calib_data[k] = v

        return calib_data
    
    __all__ = [
        'imread',
        'makepath',
        'read_pfm',

        'Pathlike',
    ]

Pathlike = Union[str, Path]
    
def imread(path: Pathlike, norm: bool = True, *args, **kwargs) -> Tensor:
    global _exr_checked

    path = makepath(path)
    if path.suffix == '.pfm':
        img = read_pfm(path, *args, **kwargs)
    else:
        if path.suffix == 'exr' and not _exr_checked:
            if os.environ.get('OPENCV_IO_ENABLE_OPENEXR', '0') == '0':
                os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
                warnings.warn('OPENCV_IO_ENABLE_OPENEXR not set. It has been set to 1')
            _exr_checked = True

        img = iio.imread(path, *args, **kwargs)

    if img.dtype == np.uint16:
        img = img.astype(np.int32)
    elif img.dtype == np.uint32:
        img = img.astype(np.int64)

    if norm:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.
        elif img.dtype == np.uint32:
            img = img.astype(np.float32) / 4294967295.

    img = torch.from_numpy(img.copy())
    if len(img.shape) == 3:
        img = img.permute(2, 0, 1)
    return img
    
def makepath(path: Pathlike) -> Path:
    if not isinstance(path, Path):
        return Path(path)
    return path

def read_pfm(path: Pathlike, scale: bool = False) -> ndarray:
    with open(path, 'rb') as file:
        header = file.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise RuntimeError(f'Not a PFM file ({str(path)})')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise RuntimeError(f'Malformed PFM header ({str(path)})')

        s = float(file.readline().decode("ascii").rstrip())
        if s < 0:  # little-endian
            endian = '<'
            s = -s
        else:
            endian = '>'  # big-endian
        # omit scale in SceneFlow

        data = np.fromfile(file, endian + 'f')
    if scale:
        data *= s
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data


# ================================
# Data augmentation
# ================================
def AutoAgument(img, depth):
    """ Automatic data augmentation.

    Args:
        img: [H, W, 3] ndarray
        depth: [H, W] ndarray 
    """
    # Color jitter
    if np.random.rand() > 0.5:
        contrast = np.random.uniform(0.75, 1.25) 
        brightness = np.random.uniform(-0.25, 0.25)
        img = contrast * img + brightness
        img = np.clip(img, 0.0, 1.0)
    
    # Gamma
    if np.random.rand() > 0.5:
        gamma_down = np.random.uniform(0.5,1)
        gamma_up = np.random.uniform(1,2)
        gamma = gamma_up if np.random.rand()>0.5 else gamma_down
        img = img**gamma

    # Flip W
    if np.random.rand() > 0.5:
        img = np.flip(img, 1)
        depth = np.flip(depth, 1)

    # Flip H
    if np.random.rand() > 0.75:
        img = np.flip(img, 0)
        depth = np.flip(depth, 0)

    # # Rotate
    # if np.random.rand() > 0.75:
    #     degree = np.random.randint(0, 180)
    #     if len(img.shape) == 4:
    #         for i in range(img.shape[-1]):
    #             img[...,i] = rotate(img[..., i], degree, reshape=False)
    #     else:
    #         img = rotate(img, degree, reshape=False)
    #     depth = rotate(depth, degree, reshape=False)

    # Crop
    if np.random.rand()>0.5:
        limit = 20
        shift = np.random.randint(0,limit)
        h,w,c = img.shape
        img = img[shift:(h-(limit-shift)),shift:(w-(limit-shift)),:]
        depth = depth[shift:(h-(limit-shift)),shift:(w-(limit-shift))]

    # # depth_flat
    # if np.random.rand() > 0.5:
    #     target_depth = np.random.rand()*(d_max-d_min)+d_min
    #     scale = np.random.rand()*0.1 + 0.9
    #     depth[depth>0]  = depth[depth>0] +(target_depth-depth[depth>0])*scale
    
    # depth_shift
    if np.random.rand() > 0.5:
        times = np.random.uniform(0.25,1.25)
        depth = depth * times

    return img, depth

def depth_preprocess(depth):
    scale = 1.0
    depth = depth / scale
    depth_mark = depth*1.0
    depth = np.clip(depth, 0.25, 10)
    depth[depth_mark<=0] = 0
    return depth

class Canon_Depth_Set(Dataset):
    def __init__(self, dataset_dir, resize=None):
        super(Canon_Depth_Set, self).__init__()

        self.dataset_dir = dataset_dir
        scenes = glob(f'{dataset_dir}/*')
        self.scenes = sorted(scenes)
        # [scene.split('/')[-1] for scene in glob(f'{dataset_dir}/*')]
        self.resize = resize
        self.file_type = glob(f"{self.scenes[0]}/l.*")[0].split('.')[-1]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC)
        ])

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        dataset_dir = self.dataset_dir
        scene = self.scenes[index]
        DEPTH_FACTOR = 10
        resize = [self.resize[1], self.resize[0]]

        if os.path.exists(f'{scene}/d.png'):
            depth = cv.resize(cv.imread(f'{scene}/d.png',0)/255.0 * DEPTH_FACTOR, resize)
        else:
            depth = np.ones(resize, dtype=np.float64)*2.5
        l_img = cv.cvtColor(cv.imread(f'{scene}/l.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.
        r_img = cv.cvtColor(cv.imread(f'{scene}/r.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.

        l_img = self.transform(l_img.astype('float32'))
        r_img = self.transform(r_img.astype('float32'))
        img = torch.cat((l_img,r_img),dim=0)

        depth[depth<0] = 0
        depth[depth>=10] = 0
        depth = self.transform(depth.astype('float32'))
        return [img, depth]
    

class Canon_Flat2Depth_Set(Dataset):
    def __init__(self, dataset_dir, resize=None, train=True):
        super(Canon_Flat2Depth_Set, self).__init__()
        self.dataset_dir = dataset_dir
        img_paths = glob(f'{dataset_dir}/**/f4/l.*',recursive=True)
        self.file_type = img_paths[0].split('.')[-1]
        img_paths = sorted(img_paths)
        self.dis_l, self.imgp_l = [], []

        from os.path import dirname,basename
        for img_path in img_paths:
            dis_str = basename(dirname(dirname(img_path)))
            if "inf" in dis_str:
                continue
            dis = float(dis_str)
            dis_m = dis/1000.0
            self.dis_l.append(dis_m)
            imgp = dirname(dirname(img_path))
            self.imgp_l.append(imgp)

        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC)
        ])

    def __len__(self):
        return len(self.imgp_l)

    def __getitem__(self, index):
        dis_m, imgp = self.dis_l[index], self.imgp_l[index]
        resize = [self.resize[1], self.resize[0]]
        f4_l_img = cv.cvtColor(cv.imread(f'{imgp}/f4/l.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.
        f4_r_img = cv.cvtColor(cv.imread(f'{imgp}/f4/r.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.

        f4_l_img = self.transform(f4_l_img.astype('float32'))
        f4_r_img = self.transform(f4_r_img.astype('float32'))
        f4_img = torch.cat((f4_l_img, f4_r_img), dim=0)

        depth = np.ones(resize)*dis_m
        depth = self.transform(depth.astype('float32'))
        return [f4_img, depth]

class Canon_Flat_Set(Dataset):
    def __init__(self, dataset_dir, resize=None, train=True):
        super(Canon_Flat_Set, self).__init__()
        inf = 100000
        self.file_type = 'png'
        self.dataset_dir = dataset_dir
        img_paths = glob(f'{dataset_dir}/**/f4/l.{self.file_type}',recursive=True)
        img_paths = sorted(img_paths)
        self.dis_l, self.imgp_l = [], []

        from os.path import dirname,basename
        for img_path in img_paths:
            dis_str = basename(dirname(dirname(img_path)))
            dis = inf if "inf" in dis_str else float(dis_str)
            dis_m = dis/1000.0
            self.dis_l.append(dis_m)
            imgp = dirname(dirname(img_path))
            self.imgp_l.append(imgp)

        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC)
        ])

    def __len__(self):
        return len(self.imgp_l)

    def __getitem__(self, index):
        dis_m, imgp = self.dis_l[index], self.imgp_l[index]
        resize = [self.resize[1], self.resize[0]]
        f4_l_img = cv.cvtColor(cv.imread(f'{imgp}/f4/l.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.
        f4_r_img = cv.cvtColor(cv.imread(f'{imgp}/f4/r.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.
        f20_l_img = cv.cvtColor(cv.imread(f'{imgp}/f20/l.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.
        f20_r_img = cv.cvtColor(cv.imread(f'{imgp}/f20/r.{self.file_type}'), cv.COLOR_BGR2RGB) / 255.


        f4_l_img = self.transform(f4_l_img.astype('float32'))
        f4_r_img = self.transform(f4_r_img.astype('float32'))
        f20_l_img = self.transform(f20_l_img.astype('float32'))
        f20_r_img = self.transform(f20_r_img.astype('float32'))

        f4_img = torch.cat((f4_l_img, f4_r_img), dim=0)
        f20_img = torch.cat((f20_l_img, f20_r_img), dim=0)

        depth = np.ones(resize)*dis_m
        depth = self.transform(depth.astype('float32'))
        return [f4_img, f20_img, depth]
    

    


class Ours_F4(Dataset):
    def __init__(self, dataset_dir, resize=None, train=True, render=False):
        super(Ours_F4,self).__init__()
        self.dataset_dir = dataset_dir
        self.render = render
        img_series = [img.split('/')[-1].split('.')[0] for img in glob(f'{dataset_dir}/F20/*')]
        corrupt = []
        self.img_uses = []
        with open(f'{dataset_dir}/corrupt_F4.txt') as f:
            for line in f.readlines():
                line = line.strip('\n')
                corrupt.append(line)
        for series in img_series:
            if series in corrupt or series in self.img_uses:
                continue
            self.img_uses.append(series)

        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC)
        ])
        self.transform_d = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
            return len(self.img_uses)
        
    def __getitem__(self, index):
        img_use = self.img_uses[index]
        img_f4_l = cv.cvtColor(cv.imread(f'{self.dataset_dir}/F4/{img_use}_l.JPG'), cv.COLOR_BGR2RGB) / 255.
        img_f4_r = cv.cvtColor(cv.imread(f'{self.dataset_dir}/F4/{img_use}_r.JPG'), cv.COLOR_BGR2RGB) / 255.
        img_f20 = cv.cvtColor(cv.imread(f'{self.dataset_dir}/F20/{img_use}.jpg'), cv.COLOR_BGR2RGB) / 255.

        img_f4_l = self.transform(img_f4_l.astype('float32'))
        img_f4_r = self.transform(img_f4_r.astype('float32'))
        img_f20 = self.transform(img_f20.astype('float32'))

        depth = cv.imread(f'{self.dataset_dir}/Depth_F4/{img_use}.png', cv2.IMREAD_UNCHANGED)
        depth = (depth/1000.0).astype(np.float32)

        kernel = np.ones((5,5), dtype=np.float32)
        depth = cv.resize(depth, (768,512))

        min_neighbors = cv.erode(depth, kernel)
        depth[min_neighbors < 0.2] = 0
        depth[depth<0.2] = 0
        depth[depth>=20] = 0

        depth = self.transform_d(depth.astype('float32'))

        img_f4 = torch.cat((img_f4_l, img_f4_r), dim=0)
        
        if self.render:
            return img_f20
        else:
            return img_f4_l, img_f4_r, img_f4, img_f20, depth

class Ours_F8(Dataset):
    def __init__(self, dataset_dir, resize=None, train=True, render=False):
        super(Ours_F8,self).__init__()
        self.dataset_dir = dataset_dir
        self.render = render
        img_series = [img.split('/')[-1].split('.')[0] for img in glob(f'{dataset_dir}/F20/*')]
        corrupt = []
        self.img_uses = []
        with open(f'{dataset_dir}/corrupt_F8.txt') as f:
            for line in f.readlines():
                line = line.strip('\n')
                corrupt.append(line)
        for series in img_series:
            if series in corrupt or series in self.img_uses:
                continue
            self.img_uses.append(series)

        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC)
        ])
        self.transform_d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC)
        ])

    def __len__(self):
            return len(self.img_uses)
        
    def __getitem__(self, index):
        img_use = self.img_uses[index]
        img_f8_l = cv.cvtColor(cv.imread(f'{self.dataset_dir}/F8/{img_use}_l.JPG'), cv.COLOR_BGR2RGB) / 255.
        img_f8_r = cv.cvtColor(cv.imread(f'{self.dataset_dir}/F8/{img_use}_r.JPG'), cv.COLOR_BGR2RGB) / 255.
        img_f20 = cv.cvtColor(cv.imread(f'{self.dataset_dir}/F20/{img_use}.jpg'), cv.COLOR_BGR2RGB) / 255.

        img_f8_l = self.transform(img_f8_l.astype('float32'))
        img_f8_r = self.transform(img_f8_r.astype('float32'))
        img_f20 = self.transform(img_f20.astype('float32'))

        depth = cv.imread(f'{self.dataset_dir}/Depth_F8/{img_use}.png', cv2.IMREAD_UNCHANGED)
        depth = (depth/1000.0).astype(np.float32)

        kernel = np.ones((5,5), dtype=np.float32)
        depth = cv.resize(depth, (768,512))

        min_neighbors = cv.erode(depth, kernel)
        depth[min_neighbors < 0.2] = 0
        depth[depth<0.2] = 0
        depth[depth>=20] = 0

        depth = self.transform_d(depth.astype('float32'))

        img_f8 = torch.cat((img_f8_l, img_f8_r), dim=0)
        if self.render:
            return img_f20
        else:   
            return img_f8_l, img_f8_r, img_f8, img_f20, depth
    
class ICCP2020(Dataset):
    def __init__(self, dataset_dir, resize=None, train=True):
        super(ICCP2020,self).__init__()
        self.dataset_dir = dataset_dir
        self.img_series = [img.split('/')[-1].split('_')[0] for img in glob(f'{dataset_dir}/*')]
        self.img_series = list(set(self.img_series))
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC)
        ])
        self.transform_d = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
            return len(self.img_series)
    
    def __getitem__(self, index):
        img_use = self.img_series[index]
        img_b = cv.cvtColor(cv.imread(f'{self.dataset_dir}/{img_use}_B.jpg'), cv.COLOR_BGR2RGB) / 255.
        img_l = cv.cvtColor(cv.imread(f'{self.dataset_dir}/{img_use}_L.jpg'), cv.COLOR_BGR2RGB) / 255.
        img_r = cv.cvtColor(cv.imread(f'{self.dataset_dir}/{img_use}_R.jpg'), cv.COLOR_BGR2RGB) / 255.

        disp = Image.open(f'{self.dataset_dir}/{img_use}_D.TIF')
        disp = np.array(disp).astype(np.float32)

        img_b, img_l, img_r, disp = random_crop_same(img_b, img_l, img_r, disp, 2940, 4410)
        img_b = self.transform(img_b.astype('float32'))
        img_l = self.transform(img_l.astype('float32'))
        img_r = self.transform(img_r.astype('float32'))
        disp = self.transform((disp/disp.max()).astype('float32'))
        depth = disp2depth(disp)
        depth = map_depth_to_range(depth)
        
        return img_l, img_r, disp, img_b, depth

class QPD2K(Dataset):
    def __init__(self, dataset_dir, resize=None, train=True):
        """QPD2K dataset loader.

        Expected layout under `dataset_dir`:
            gt/0000.npz, 0001.npz, ...           (depth ground-truth)
            input/L/0000.png, 0001.png, ...      (left RGB)
            input/R/0000.png, 0001.png, ...      (right RGB)

        Notes:
        - Depth in npz can be stored under common keys like 'depth' or as 'arr_0'.
        - Depth unit is assumed meters; if values look like millimeters (e.g. max > 200),
          it will be converted to meters by dividing by 1000.
        """
        super(QPD2K, self).__init__()
        self.dataset_dir = dataset_dir
        self.resize = resize
        self.train = train

        self.gt_dir = Path(dataset_dir) / 'gt'
        self.left_dir = Path(dataset_dir) / 'input' / 'L'
        self.right_dir = Path(dataset_dir) / 'input' / 'R'

        gt_files = sorted(self.gt_dir.glob('*.npz'))
        sample_ids = []
        for p in gt_files:
            stem = p.stem
            if (self.left_dir / f'{stem}.png').exists() and (self.right_dir / f'{stem}.png').exists():
                sample_ids.append(stem)
        self.sample_ids = sample_ids

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC),
        ])

    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, index):
        sample_id = self.sample_ids[index]

        left_path = self.left_dir / f'{sample_id}.png'
        right_path = self.right_dir / f'{sample_id}.png'
        depth_path = self.gt_dir / f'{sample_id}.npz'

        img_l = cv.imread(str(left_path), cv.IMREAD_COLOR)
        img_r = cv.imread(str(right_path), cv.IMREAD_COLOR)
        if img_l is None or img_r is None:
            raise FileNotFoundError(f'Failed to read RGB images: {left_path} / {right_path}')

        img_l = cv.cvtColor(img_l, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_r = cv.cvtColor(img_r, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_b = (img_l + img_r) / 2.0

        depth = self._load_npz_depth(depth_path)
        # Make sure depth is HxW float32
        depth = np.asarray(depth).squeeze().astype(np.float32)
        depth = np.abs(depth)  # ensure non-negative
        if depth.ndim != 2:
            raise ValueError(f'Expected depth to be 2D after squeeze, got shape={depth.shape} from {depth_path}')

        # # Heuristic: convert millimeters to meters if values are too large.
        # if np.nanmax(depth) > 200.0:
        #     depth = depth / 1000.0

        # Align depth resolution to RGB if needed.
        H, W = img_l.shape[:2]
        if depth.shape[:2] != (H, W):
            depth = cv.resize(depth, (W, H), interpolation=cv.INTER_NEAREST)

        # # Crop (kept consistent with existing datasets; auto-clamp if images are smaller)
        # crop_h = min(2940, H)
        # crop_w = min(4410, W)
        # img_b, img_l, img_r, depth = random_crop_same(img_b, img_l, img_r, depth, crop_h, crop_w)

        # Depth validity filtering in meters
        depth = depth.astype(np.float32)
        depth[depth < 0.02] = 0
        depth[depth >= 20] = 0

        disp = depth2disp(depth, d_max=float(np.max(depth)) if np.any(depth > 0) else 20.0, d_min=0.2)

        # ToTensor + Resize
        img_b = self.transform(img_b.astype('float32'))
        img_l = self.transform(img_l.astype('float32'))
        img_r = self.transform(img_r.astype('float32'))
        disp = self.transform(disp.astype('float32'))
        depth = self.transform(depth.astype('float32'))

        return img_l, img_r, disp, img_b, depth

    @staticmethod
    def _load_npz_depth(path: Path) -> np.ndarray:
        data = np.load(str(path))
        try:
            if isinstance(data, np.lib.npyio.NpzFile):
                if 'depth' in data.files:
                    return data['depth']
                if 'arr_0' in data.files:
                    return data['arr_0']
                if len(data.files) == 1:
                    return data[data.files[0]]
                raise KeyError(
                    f'NPZ depth file has multiple arrays {data.files}; please name one as "depth"'
                )
            return data
        finally:
            close = getattr(data, 'close', None)
            if callable(close):
                close()

class QPD2Kpatch(Dataset):
    """Patch-level QPD2K dataset compatible with `collate_patch_samples`.

    It reads QPD2K scene files like:
        <dataset_dir>/gt/<sample>.npz
        <dataset_dir>/input/L/<sample>.png
        <dataset_dir>/input/R/<sample>.png

    And returns:
        l_patches: [N, 3, patch_size, patch_size]
        r_patches: [N, 3, patch_size, patch_size]
        depth_vals: [N, 1] (raw depth values, without range mapping)
    """

    def __init__(
        self,
        dataset_dir: str,
        patch_size: int = 50,
        resize=None,
        train: bool = True,
        train_points=(500, 1000),
        val_points: int = 50,
        split_ratio: float = 0.9,
        seed: int = 1234,
        raw_d_min: float = 0.02,
        raw_d_max: float = 20.0,
        depth_erode_ks: int = 0,
    ):
        super().__init__()

        self.dataset_dir = str(dataset_dir)
        self.patch_size = int(patch_size)
        self.margin = self.patch_size // 2
        self.resize = resize

        self.train = bool(train)
        self.train_points = train_points
        self.val_points = int(val_points)
        self.seed = int(seed)

        self.raw_d_min = float(raw_d_min)
        self.raw_d_max = float(raw_d_max)
        self.depth_erode_ks = int(depth_erode_ks)

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        self.gt_dir = Path(self.dataset_dir) / 'gt'
        self.left_dir = Path(self.dataset_dir) / 'input' / 'L'
        self.right_dir = Path(self.dataset_dir) / 'input' / 'R'

        gt_files = sorted(self.gt_dir.glob('*.npz'))
        all_sample_ids = []
        for p in gt_files:
            stem = p.stem
            if (self.left_dir / f'{stem}.png').exists() and (self.right_dir / f'{stem}.png').exists():
                all_sample_ids.append(stem)

        if len(all_sample_ids) == 0:
            raise RuntimeError(f"No valid scenes found under dataset_dir={self.dataset_dir}")

        split_idx = int(len(all_sample_ids) * float(split_ratio))
        split_idx = max(1, min(split_idx, len(all_sample_ids) - 1)) if len(all_sample_ids) > 1 else 1
        if self.train:
            self.sample_ids = all_sample_ids[:split_idx]
        else:
            self.sample_ids = all_sample_ids[split_idx:] if len(all_sample_ids) > 1 else all_sample_ids

    def __len__(self):
        return len(self.sample_ids)

    @staticmethod
    def _load_npz_depth(path: Path) -> np.ndarray:
        data = np.load(str(path))
        try:
            if isinstance(data, np.lib.npyio.NpzFile):
                if 'depth' in data.files:
                    return data['depth']
                if 'arr_0' in data.files:
                    return data['arr_0']
                if len(data.files) == 1:
                    return data[data.files[0]]
                raise KeyError(
                    f'NPZ depth file has multiple arrays {data.files}; please name one as "depth"'
                )
            return data
        finally:
            close = getattr(data, 'close', None)
            if callable(close):
                close()

    def _read_scene(self, sample_id: str):
        left_path = self.left_dir / f'{sample_id}.png'
        right_path = self.right_dir / f'{sample_id}.png'
        depth_path = self.gt_dir / f'{sample_id}.npz'

        img_l = cv.imread(str(left_path), cv.IMREAD_COLOR)
        img_r = cv.imread(str(right_path), cv.IMREAD_COLOR)
        if img_l is None or img_r is None:
            raise RuntimeError(f'Failed to read RGB images for sample: {sample_id}')

        img_l = cv.cvtColor(img_l, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_r = cv.cvtColor(img_r, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth = self._load_npz_depth(depth_path)
        depth = np.asarray(depth).squeeze().astype(np.float32)
        depth = np.abs(depth)
        if depth.ndim != 2:
            raise ValueError(f'Expected depth to be 2D after squeeze, got shape={depth.shape} from {depth_path}')

        H, W = img_l.shape[:2]
        if depth.shape[:2] != (H, W):
            depth = cv.resize(depth, (W, H), interpolation=cv.INTER_NEAREST)

        # Mask invalid depths.
        depth[(depth < self.raw_d_min) | (depth >= self.raw_d_max)] = 0.0

        if self.resize is not None:
            if isinstance(self.resize, (tuple, list)) and len(self.resize) == 2:
                rh, rw = int(self.resize[0]), int(self.resize[1])
                if rh > 0 and rw > 0:
                    img_l = cv.resize(img_l, (rw, rh), interpolation=cv.INTER_AREA)
                    img_r = cv.resize(img_r, (rw, rh), interpolation=cv.INTER_AREA)
                    depth = cv.resize(depth, (rw, rh), interpolation=cv.INTER_NEAREST)

        return img_l, img_r, depth

    def _get_candidate_points(self, depth: np.ndarray):
        h, w = depth.shape
        y0, y1 = self.margin, h - self.margin
        x0, x1 = self.margin, w - self.margin
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 2), dtype=np.int64)

        depth_use = depth
        if self.depth_erode_ks and self.depth_erode_ks > 1:
            k = int(self.depth_erode_ks)
            kernel = np.ones((k, k), dtype=np.float32)
            depth_use = cv.erode(depth_use, kernel)

        roi = depth_use[y0:y1, x0:x1]
        ys, xs = np.where(roi > 0)
        if ys.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        ys = ys.astype(np.int64) + y0
        xs = xs.astype(np.int64) + x0
        return np.stack([ys, xs], axis=1)

    def _sample_indices(self, num_candidates, num_points, rng):
        if num_candidates == 0:
            return np.zeros((0,), dtype=np.int64)
        replace = num_candidates < num_points
        return rng.choice(num_candidates, size=num_points, replace=replace)

    def _extract_patch(self, img, y, x):
        m = self.margin
        if self.patch_size % 2 == 0:
            return img[y - m : y + m, x - m : x + m]
        return img[y - m : y + m + 1, x - m : x + m + 1]

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        img_l, img_r, depth_raw = self._read_scene(sample_id)

        rng = np.random.default_rng(self.seed + index)
        candidates = self._get_candidate_points(depth_raw)

        if self.train:
            low, high = int(self.train_points[0]), int(self.train_points[1])
            num_points = int(rng.integers(low, high + 1))
        else:
            num_points = self.val_points

        sample_idx = self._sample_indices(len(candidates), num_points, rng)
        if len(sample_idx) == 0:
            lp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            rp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            dp = torch.zeros((1, 1), dtype=torch.float32)
            return lp, rp, dp

        points = candidates[sample_idx]
        l_patches = []
        r_patches = []
        d_vals = []
        for (y, x) in points:
            y = int(y)
            x = int(x)
            l_patch = self._extract_patch(img_l, y, x)
            r_patch = self._extract_patch(img_r, y, x)
            if l_patch.shape[:2] != (self.patch_size, self.patch_size):
                continue
            if r_patch.shape[:2] != (self.patch_size, self.patch_size):
                continue
                
            # 直接使用原始深度值，不需要 map_depth_to_range
            d = float(depth_raw[y, x])
            if not (d > 0.0):
                continue

            l_patches.append(np.transpose(l_patch, (2, 0, 1)))
            r_patches.append(np.transpose(r_patch, (2, 0, 1)))
            d_vals.append(d)

        if len(l_patches) == 0:
            lp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            rp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            dp = torch.zeros((1, 1), dtype=torch.float32)
            return lp, rp, dp

        l_patches = torch.from_numpy(np.ascontiguousarray(np.stack(l_patches, axis=0))).float()
        r_patches = torch.from_numpy(np.ascontiguousarray(np.stack(r_patches, axis=0))).float()
        d_vals = torch.from_numpy(np.ascontiguousarray(np.array(d_vals, dtype=np.float32))).unsqueeze(1)

        return l_patches, r_patches, d_vals

class LearningDP(Dataset):
    def __init__(self, dataset_dir, resize=None, train=True, render=False):
        super(LearningDP, self).__init__()
        self.dataset_dir = dataset_dir
        self.img_series = [img.split('/')[-1] for img in glob(f'{dataset_dir}/scaled_images/*')]
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC)
        ])
        self.render = render

    def __len__(self):
            return len(self.img_series)
    
    def __getitem__(self, index):
        img_use = self.img_series[index]
        img_b = cv.cvtColor(cv.imread(f'{self.dataset_dir}/scaled_images/{img_use}/result_scaled_image_center.jpg'), cv.COLOR_BGR2RGB) / 255.
        img_l = cv.cvtColor(cv.imread(f'{self.dataset_dir}/left_pd/{img_use}/result_leftPd_center.png'), cv.COLOR_BGR2RGB) / 255.
        img_r = cv.cvtColor(cv.imread(f'{self.dataset_dir}/right_pd/{img_use}/result_rightPd_center.png'), cv.COLOR_BGR2RGB) / 255.    
        img_l = cv.resize(img_l, (img_b.shape[1], img_b.shape[0]))
        img_r = cv.resize(img_r, (img_b.shape[1], img_b.shape[0]))
        depth = cv.imread(f'{self.dataset_dir}/merged_depth/{img_use}/result_merged_depth_center.png', cv2.IMREAD_UNCHANGED) / 255.   
        img_b, img_l, img_r, depth = random_crop_same(img_b, img_l, img_r, depth, np.int16(img_b.shape[1]*2/3), img_b.shape[1])
        img_b = self.transform(img_b.astype('float32'))
        img_l = self.transform(img_l.astype('float32'))
        img_r = self.transform(img_r.astype('float32'))
        # depth = self.transform((depth/255).astype('float32'))
        dmax, dmin = 100.0, 0.2
        depth = (dmax*dmin)/(dmax-(dmax-dmin)*depth)
        depth[depth<0.2] = 0
        depth[depth>=100] = 0
        
        disp = depth2disp(depth,d_max=depth.max(),d_min=dmin)
        disp = self.transform(disp.astype('float32'))
        depth = self.transform(depth.astype('float32'))
        depth = map_depth_to_range(depth)

        if self.render:
            return img_b
        else:
            return img_l, img_r, disp, img_b, depth

class LDPpatch(Dataset):
    """Patch-level LearningDP dataset compatible with `collate_patch_samples`.

    It reads LearningDP scene folders like:
        <dataset_dir>/scaled_images/<scene>/result_scaled_image_center.jpg
        <dataset_dir>/left_pd/<scene>/result_leftPd_center.png
        <dataset_dir>/right_pd/<scene>/result_rightPd_center.png
        <dataset_dir>/merged_depth/<scene>/result_merged_depth_center.png

    And returns:
        l_patches: [N, 3, patch_size, patch_size]
        r_patches: [N, 3, patch_size, patch_size]
        depth_vals: [N, 1]  (mapped to a target range via `map_depth_to_range`)
    """

    def __init__(
        self,
        dataset_dir: str,
        patch_size: int = 50,
        resize=None,
        train: bool = True,
        train_points=(500, 1000),
        val_points: int = 50,
        split_ratio: float = 0.9,
        seed: int = 1234,
        raw_d_min: float = 0.2,
        raw_d_max: float = 100.0,
        mapped_d_min: float = 1.0,
        mapped_d_max: float = 10.0,
        depth_erode_ks: int = 0,
    ):
        super().__init__()

        self.dataset_dir = str(dataset_dir)
        self.patch_size = int(patch_size)
        self.margin = self.patch_size // 2
        self.resize = resize

        self.train = bool(train)
        self.train_points = train_points
        self.val_points = int(val_points)
        self.seed = int(seed)

        self.raw_d_min = float(raw_d_min)
        self.raw_d_max = float(raw_d_max)
        self.mapped_d_min = float(mapped_d_min)
        self.mapped_d_max = float(mapped_d_max)
        self.depth_erode_ks = int(depth_erode_ks)

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        # ── 与 LearningDP 相同的 scene 枚举方式 ──────────────────────────────
        all_scenes = sorted(
            img.split('/')[-1]
            for img in glob(os.path.join(self.dataset_dir, 'scaled_images', '*'))
        )
        if len(all_scenes) == 0:
            raise RuntimeError(f"No scenes found under dataset_dir={self.dataset_dir}/scaled_images/")

        split_idx = int(len(all_scenes) * float(split_ratio))
        split_idx = max(1, min(split_idx, len(all_scenes) - 1)) if len(all_scenes) > 1 else 1
        if self.train:
            self.scenes = all_scenes[:split_idx]
        else:
            self.scenes = all_scenes[split_idx:] if len(all_scenes) > 1 else all_scenes

    def __len__(self):
        return len(self.scenes)

    def _read_scene(self, img_name: str):
        base = self.dataset_dir

        # ── 与 LearningDP 相同的路径约定 ─────────────────────────────────────
        img_b = cv.imread(f'{base}/scaled_images/{img_name}/result_scaled_image_center.jpg')
        img_l = cv.imread(f'{base}/left_pd/{img_name}/result_leftPd_center.png')
        img_r = cv.imread(f'{base}/right_pd/{img_name}/result_rightPd_center.png')
        dep   = cv.imread(f'{base}/merged_depth/{img_name}/result_merged_depth_center.png',
                          cv.IMREAD_UNCHANGED)

        if img_b is None or img_l is None or img_r is None or dep is None:
            raise RuntimeError(f"Invalid LearningDP sample: {img_name}")

        img_b = cv.cvtColor(img_b, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_l = cv.cvtColor(img_l, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_r = cv.cvtColor(img_r, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # img_l / img_r 可能与 img_b 尺寸不同，与 LearningDP 保持一致
        h, w = img_b.shape[:2]
        img_l = cv.resize(img_l, (w, h), interpolation=cv.INTER_AREA)
        img_r = cv.resize(img_r, (w, h), interpolation=cv.INTER_AREA)

        # ── 与 LearningDP 相同的深度解码公式 ──────────────────────────────────
        dep = dep.astype(np.float32) / 255.0
        dmax, dmin = self.raw_d_max, self.raw_d_min          # 100.0, 0.2
        dep = (dmax * dmin) / (dmax - (dmax - dmin) * dep)
        dep[(dep < self.raw_d_min) | (dep >= self.raw_d_max)] = 0.0

        # ── 可选 resize（OpenCV，与 DP5Kpatch 一致）──────────────────────────
        if self.resize is not None:
            if isinstance(self.resize, (tuple, list)) and len(self.resize) == 2:
                rh, rw = int(self.resize[0]), int(self.resize[1])
                if rh > 0 and rw > 0:
                    img_l = cv.resize(img_l, (rw, rh), interpolation=cv.INTER_AREA)
                    img_r = cv.resize(img_r, (rw, rh), interpolation=cv.INTER_AREA)
                    dep   = cv.resize(dep,   (rw, rh), interpolation=cv.INTER_NEAREST)

        return img_l, img_r, dep

    # ── 以下三个辅助方法与 DP5Kpatch 完全相同 ──────────────────────────────────

    def _get_candidate_points(self, depth: np.ndarray):
        h, w = depth.shape
        y0, y1 = self.margin, h - self.margin
        x0, x1 = self.margin, w - self.margin
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 2), dtype=np.int64)

        depth_use = depth
        if self.depth_erode_ks and self.depth_erode_ks > 1:
            k = int(self.depth_erode_ks)
            kernel = np.ones((k, k), dtype=np.float32)
            depth_use = cv.erode(depth_use, kernel)

        roi = depth_use[y0:y1, x0:x1]
        ys, xs = np.where(roi > 0)
        if ys.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        ys = ys.astype(np.int64) + y0
        xs = xs.astype(np.int64) + x0
        return np.stack([ys, xs], axis=1)

    def _sample_indices(self, num_candidates, num_points, rng):
        if num_candidates == 0:
            return np.zeros((0,), dtype=np.int64)
        replace = num_candidates < num_points
        return rng.choice(num_candidates, size=num_points, replace=replace)

    def _extract_patch(self, img, y, x):
        m = self.margin
        if self.patch_size % 2 == 0:
            return img[y - m : y + m,     x - m : x + m]
        return     img[y - m : y + m + 1, x - m : x + m + 1]

    def __getitem__(self, index: int):
        img_name = self.scenes[index]
        img_l, img_r, depth_raw = self._read_scene(img_name)

        rng = np.random.default_rng(self.seed + index)
        candidates = self._get_candidate_points(depth_raw)

        if self.train:
            low, high = int(self.train_points[0]), int(self.train_points[1])
            num_points = int(rng.integers(low, high + 1))
        else:
            num_points = self.val_points

        sample_idx = self._sample_indices(len(candidates), num_points, rng)
        if len(sample_idx) == 0:
            lp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            rp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            dp = torch.zeros((1, 1), dtype=torch.float32)
            return lp, rp, dp

        # depth_mapped = map_depth_to_range(
        #     torch.from_numpy(depth_raw).float(),
        #     d_min=self.mapped_d_min,
        #     d_max=self.mapped_d_max,
        # ).numpy()

        points = candidates[sample_idx]
        l_patches, r_patches, d_vals = [], [], []
        for (y, x) in points:
            y, x = int(y), int(x)
            l_patch = self._extract_patch(img_l, y, x)
            r_patch = self._extract_patch(img_r, y, x)
            if l_patch.shape[:2] != (self.patch_size, self.patch_size):
                continue
            if r_patch.shape[:2] != (self.patch_size, self.patch_size):
                continue
            d = float(depth_raw[y, x])
            if not (d > 0.0):
                continue

            l_patches.append(np.transpose(l_patch, (2, 0, 1)))
            r_patches.append(np.transpose(r_patch, (2, 0, 1)))
            d_vals.append(d)

        if len(l_patches) == 0:
            lp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            rp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            dp = torch.zeros((1, 1), dtype=torch.float32)
            return lp, rp, dp

        l_patches = torch.from_numpy(np.ascontiguousarray(np.stack(l_patches, axis=0))).float()
        r_patches = torch.from_numpy(np.ascontiguousarray(np.stack(r_patches, axis=0))).float()
        d_vals    = torch.from_numpy(np.ascontiguousarray(np.array(d_vals, dtype=np.float32))).unsqueeze(1)

        return l_patches, r_patches, d_vals
    
class DP5K(Dataset):
    def __init__(self, dataset_dir, resize=None, train=True):
        super(DP5K, self).__init__()
        self.dataset_dir = dataset_dir
        self.img_series = [img for img in glob(f'{dataset_dir}/*')]
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC)
        ])

    def __len__(self):
            return len(self.img_series)
    
    def __getitem__(self, index):
        img_use = self.img_series[index]
        img_b = cv.cvtColor(cv.imread(f'{img_use}/cam0/FN_22.png'), cv.COLOR_BGR2RGB) / 255.
        img_l = cv.cvtColor(cv.imread(f'{img_use}/cam0/FN_1.8_l.png'), cv.COLOR_BGR2RGB) / 255.
        img_r = cv.cvtColor(cv.imread(f'{img_use}/cam0/FN_1.8_r.png'), cv.COLOR_BGR2RGB) / 255.    
        depth = cv.imread(f'{img_use}/cam0/dep.png', cv.IMREAD_UNCHANGED) / 1000.0

        img_b = self.transform(img_b.astype('float32'))
        img_l = self.transform(img_l.astype('float32'))
        img_r = self.transform(img_r.astype('float32'))
        # depth = self.transform(depth.astype('float32'))

        depth[depth<0.2] = 0
        depth[depth>=20] = 0
        
        disp = depth2disp(depth,d_max=depth.max())
        disp = self.transform(disp.astype('float32'))
        depth = self.transform(depth.astype('float32'))
        depth = map_depth_to_range(depth)
        
        return img_l, img_r, disp, img_b, depth


class DP5Kpatch(Dataset):
    """Patch-level DP5K dataset compatible with `collate_patch_samples`.

    It reads DP5K scene folders like:
        <scene>/cam0/FN_1.8_l.png
        <scene>/cam0/FN_1.8_r.png
        <scene>/cam0/dep.png

    And returns:
        l_patches: [N,3,patch,patch]
        r_patches: [N,3,patch,patch]
        depth_vals: [N,1] (mapped to a target range via `map_depth_to_range`)
    """

    def __init__(
        self,
        dataset_dir: str,
        patch_size: int = 50,
        resize=None,
        train: bool = True,
        train_points=(500, 1000),
        val_points: int = 50,
        split_ratio: float = 0.9,
        seed: int = 1234,
        raw_d_min: float = 0.2,
        raw_d_max: float = 20.0,
        mapped_d_min: float = 1.0,
        mapped_d_max: float = 10.0,
        depth_erode_ks: int = 0,
    ):
        super().__init__()

        self.dataset_dir = str(dataset_dir)
        self.patch_size = int(patch_size)
        self.margin = self.patch_size // 2
        self.resize = resize

        self.train = bool(train)
        self.train_points = train_points
        self.val_points = int(val_points)
        self.seed = int(seed)

        self.raw_d_min = float(raw_d_min)
        self.raw_d_max = float(raw_d_max)
        self.mapped_d_min = float(mapped_d_min)
        self.mapped_d_max = float(mapped_d_max)
        self.depth_erode_ks = int(depth_erode_ks)

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        all_scenes = sorted(glob(os.path.join(self.dataset_dir, "*")))
        all_scenes = [p for p in all_scenes if os.path.isdir(p)]
        if len(all_scenes) == 0:
            raise RuntimeError(f"No scenes found under dataset_dir={self.dataset_dir}")

        split_idx = int(len(all_scenes) * float(split_ratio))
        split_idx = max(1, min(split_idx, len(all_scenes) - 1)) if len(all_scenes) > 1 else 1
        if self.train:
            self.scenes = all_scenes[:split_idx]
        else:
            self.scenes = all_scenes[split_idx:] if len(all_scenes) > 1 else all_scenes

    def __len__(self):
        return len(self.scenes)

    def _read_scene(self, scene_dir: str):
        cam0 = os.path.join(scene_dir, "cam0")
        l_path = os.path.join(cam0, "FN_1.8_l.png")
        r_path = os.path.join(cam0, "FN_1.8_r.png")
        d_path = os.path.join(cam0, "dep.png")

        img_l = cv.imread(l_path, cv.IMREAD_COLOR)
        img_r = cv.imread(r_path, cv.IMREAD_COLOR)
        dep = cv.imread(d_path, cv.IMREAD_UNCHANGED)
        if img_l is None or img_r is None or dep is None:
            raise RuntimeError(f"Invalid DP5K sample in {scene_dir}")

        img_l = cv.cvtColor(img_l, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_r = cv.cvtColor(img_r, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        dep = dep.astype(np.float32) / 1000.0  # mm -> m

        # Mask invalid depths.
        dep[(dep < self.raw_d_min) | (dep >= self.raw_d_max)] = 0.0

        if self.resize is not None:
            # torchvision's Resize expects (H,W); OpenCV uses (W,H)
            if isinstance(self.resize, (tuple, list)) and len(self.resize) == 2:
                h, w = int(self.resize[0]), int(self.resize[1])
                if h > 0 and w > 0:
                    img_l = cv.resize(img_l, (w, h), interpolation=cv.INTER_AREA)
                    img_r = cv.resize(img_r, (w, h), interpolation=cv.INTER_AREA)
                    dep = cv.resize(dep, (w, h), interpolation=cv.INTER_NEAREST)
        return img_l, img_r, dep

    def _get_candidate_points(self, depth: np.ndarray):
        h, w = depth.shape
        y0, y1 = self.margin, h - self.margin
        x0, x1 = self.margin, w - self.margin
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 2), dtype=np.int64)

        depth_use = depth
        if self.depth_erode_ks and self.depth_erode_ks > 1:
            k = int(self.depth_erode_ks)
            kernel = np.ones((k, k), dtype=np.float32)
            depth_use = cv.erode(depth_use, kernel)

        roi = depth_use[y0:y1, x0:x1]
        ys, xs = np.where(roi > 0)
        if ys.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        ys = ys.astype(np.int64) + y0
        xs = xs.astype(np.int64) + x0
        return np.stack([ys, xs], axis=1)

    def _sample_indices(self, num_candidates, num_points, rng):
        if num_candidates == 0:
            return np.zeros((0,), dtype=np.int64)
        replace = num_candidates < num_points
        return rng.choice(num_candidates, size=num_points, replace=replace)

    def _extract_patch(self, img, y, x):
        m = self.margin
        if self.patch_size % 2 == 0:
            return img[y - m : y + m, x - m : x + m]
        return img[y - m : y + m + 1, x - m : x + m + 1]

    def __getitem__(self, index: int):
        scene_dir = self.scenes[index]
        img_l, img_r, depth_raw = self._read_scene(scene_dir)

        rng = np.random.default_rng(self.seed + index)
        candidates = self._get_candidate_points(depth_raw)

        if self.train:
            low, high = int(self.train_points[0]), int(self.train_points[1])
            num_points = int(rng.integers(low, high + 1))
        else:
            num_points = self.val_points

        sample_idx = self._sample_indices(len(candidates), num_points, rng)
        if len(sample_idx) == 0:
            lp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            rp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            dp = torch.zeros((1, 1), dtype=torch.float32)
            return lp, rp, dp

        # Map depth (per-scene) to the target range before sampling point values.
        # depth_mapped = map_depth_to_range(
        #     torch.from_numpy(depth_raw).float(),
        #     d_min=self.mapped_d_min,
        #     d_max=self.mapped_d_max,
        # ).numpy()

        points = candidates[sample_idx]
        l_patches = []
        r_patches = []
        d_vals = []
        for (y, x) in points:
            y = int(y)
            x = int(x)
            l_patch = self._extract_patch(img_l, y, x)
            r_patch = self._extract_patch(img_r, y, x)
            if l_patch.shape[:2] != (self.patch_size, self.patch_size):
                continue
            if r_patch.shape[:2] != (self.patch_size, self.patch_size):
                continue
            d = float(depth_raw[y, x])
            if not (d > 0.0):
                continue

            l_patches.append(np.transpose(l_patch, (2, 0, 1)))
            r_patches.append(np.transpose(r_patch, (2, 0, 1)))
            d_vals.append(d)

        if len(l_patches) == 0:
            lp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            rp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            dp = torch.zeros((1, 1), dtype=torch.float32)
            return lp, rp, dp

        l_patches = torch.from_numpy(np.ascontiguousarray(np.stack(l_patches, axis=0))).float()
        r_patches = torch.from_numpy(np.ascontiguousarray(np.stack(r_patches, axis=0))).float()
        d_vals = torch.from_numpy(np.ascontiguousarray(np.array(d_vals, dtype=np.float32))).unsqueeze(1)

        return l_patches, r_patches, d_vals


class DPpatch(Dataset):
    """Patch-level stereo dataset for center-point depth regression."""

    def __init__(
        self,
        dataset_root,
        patch_size=50,
        train=True,
        train_points=(1000, 2000),
        val_points=50,
        split_ratio=0.9,
        seed=1234,
        d_min=0.1,
        d_max=10.0,
        depth_erode_ks=0,
    ):
        super(DPpatch, self).__init__()
        self.dataset_root = dataset_root
        self.patch_size = int(patch_size)
        self.train = train
        self.train_points = train_points
        self.val_points = int(val_points)
        self.seed = int(seed)
        self.margin = self.patch_size // 2

        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.depth_erode_ks = int(depth_erode_ks)

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")

        all_scenes = self._scan_scenes(dataset_root)

        split_idx = int(len(all_scenes) * split_ratio)
        split_idx = max(1, min(split_idx, len(all_scenes) - 1)) if len(all_scenes) > 1 else 1
        if self.train:
            self.scenes = all_scenes[:split_idx]
        else:
            self.scenes = all_scenes[split_idx:] if len(all_scenes) > 1 else all_scenes

    @staticmethod
    def _scan_scenes(dataset_root: str):
        """Collect scene directories.

        Supports either:
        - dataset_root/<lens_name>/scenes/00000
        - dataset_root/scenes/00000
        """
        patterns = [
            os.path.join(dataset_root, "scenes", "[0-9][0-9][0-9][0-9][0-9]"),
            os.path.join(dataset_root, "*", "scenes", "[0-9][0-9][0-9][0-9][0-9]"),
        ]
        scenes = []
        for p in patterns:
            scenes.extend(glob(p))
        scenes = sorted(set(scenes))
        if len(scenes) == 0:
            raise RuntimeError(
                "No scene found. Tried patterns:\n"
                + "\n".join(patterns)
                + f"\nGot dataset_root={dataset_root}"
            )
        return scenes

    def __len__(self):
        return len(self.scenes)

    def _read_scene(self, scene_dir):
        l_path = os.path.join(scene_dir, "L.png")
        r_path = os.path.join(scene_dir, "R.png")
        d_path = os.path.join(scene_dir, "D.png")

        img_l = cv.imread(l_path, cv.IMREAD_COLOR)
        img_r = cv.imread(r_path, cv.IMREAD_COLOR)
        dep = cv.imread(d_path, cv.IMREAD_UNCHANGED)

        if img_l is None or img_r is None or dep is None:
            raise RuntimeError(f"Invalid scene files in {scene_dir}")

        img_l = cv.cvtColor(img_l, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_r = cv.cvtColor(img_r, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # D.png is 16-bit depth in millimeters.
        dep = dep.astype(np.float32) / 1000.0
        return img_l, img_r, dep

    def _get_candidate_points(self, depth):
        h, w = depth.shape
        y0, y1 = self.margin, h - self.margin
        x0, x1 = self.margin, w - self.margin
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)

        depth_use = depth
        if self.depth_erode_ks and self.depth_erode_ks > 1:
            k = int(self.depth_erode_ks)
            kernel = np.ones((k, k), dtype=np.float32)
            depth_use = cv.erode(depth_use, kernel)

        roi = depth_use[y0:y1, x0:x1]
        valid_roi = (roi > 0) & (roi >= self.d_min) & (roi <= self.d_max)
        ys, xs = np.where(valid_roi)
        if ys.size == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)

        ys = ys.astype(np.int64) + y0
        xs = xs.astype(np.int64) + x0
        points = np.stack([ys, xs], axis=1)
        dvals = depth[ys, xs].astype(np.float32)
        return points, dvals

    def _sample_indices(self, num_candidates, num_points, rng):
        if num_candidates == 0:
            return np.zeros((0,), dtype=np.int64)
        replace = num_candidates < num_points
        return rng.choice(num_candidates, size=num_points, replace=replace)

    def _extract_patch(self, img, y, x):
        m = self.margin
        if self.patch_size % 2 == 0:
            return img[y - m : y + m, x - m : x + m]
        return img[y - m : y + m + 1, x - m : x + m + 1]

    def __getitem__(self, index):
        scene_dir = self.scenes[index]
        img_l, img_r, depth = self._read_scene(scene_dir)

        rng = np.random.default_rng(self.seed + index)
        candidates, candidate_dvals = self._get_candidate_points(depth)

        if self.train:
            low, high = int(self.train_points[0]), int(self.train_points[1])
            num_points = int(rng.integers(low, high + 1))
        else:
            num_points = self.val_points

        sample_idx = self._sample_indices(len(candidates), num_points, rng)
        if len(sample_idx) == 0:
            # Fallback: return one zero sample to avoid dataloader crash.
            lp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            rp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            dp = torch.zeros((1, 1), dtype=torch.float32)
            return lp, rp, dp

        points = candidates[sample_idx]
        sampled_dvals = candidate_dvals[sample_idx]

        l_patches = []
        r_patches = []
        d_vals = []
        for (y, x), d in zip(points, sampled_dvals):
            y = int(y)
            x = int(x)
            l_patch = self._extract_patch(img_l, y, x)
            r_patch = self._extract_patch(img_r, y, x)
            if l_patch.shape[:2] != (self.patch_size, self.patch_size):
                continue
            if r_patch.shape[:2] != (self.patch_size, self.patch_size):
                continue

            l_patches.append(np.transpose(l_patch, (2, 0, 1)))
            r_patches.append(np.transpose(r_patch, (2, 0, 1)))
            d_vals.append(float(d))

        if len(l_patches) == 0:
            lp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            rp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            dp = torch.zeros((1, 1), dtype=torch.float32)
            return lp, rp, dp

        l_patches = torch.from_numpy(np.ascontiguousarray(np.stack(l_patches, axis=0))).float()
        r_patches = torch.from_numpy(np.ascontiguousarray(np.stack(r_patches, axis=0))).float()
        d_vals = torch.from_numpy(np.ascontiguousarray(np.array(d_vals, dtype=np.float32))).unsqueeze(1)

        return l_patches, r_patches, d_vals


class DPrealpatch(Dataset):
    """Patch dataset from real Ours captures with selectable aperture."""

    def __init__(
        self,
        indoors_dir,
        outdoors_dir,
        patch_size=50,
        train=True,
        train_points=(1000, 2000),
        val_points=50,
        split_ratio=0.9,
        seed=1234,
        d_min=0.1,
        d_max=10.0,
        aperture="F4",
    ):
        super(DPrealpatch, self).__init__()
        self.patch_size = int(patch_size)
        self.margin = self.patch_size // 2
        self.train = train
        self.train_points = train_points
        self.val_points = int(val_points)
        self.seed = int(seed)
        self.d_min = float(d_min)
        self.d_max = float(d_max)
        self.resize_h = 1024
        self.resize_w = 1536
        self.depth_erode_ks = 9
        self.aperture = str(aperture).upper()
        if self.aperture not in {"F4", "F8"}:
            raise ValueError(f"Unsupported aperture: {aperture}. Choose from ['F4', 'F8']")

        self.lr_dir = self.aperture
        self.depth_dir = f"Depth_{self.aperture}"
        self.corrupt_file = f"corrupt_{self.aperture}.txt"

        all_records = self._collect_records(indoors_dir) + self._collect_records(outdoors_dir)
        if len(all_records) == 0:
            raise RuntimeError(
                f"No valid Ours_{self.aperture} samples found in indoors/outdoors dirs."
            )

        split_idx = int(len(all_records) * split_ratio)
        split_idx = max(1, min(split_idx, len(all_records) - 1)) if len(all_records) > 1 else 1
        if self.train:
            self.records = all_records[:split_idx]
        else:
            self.records = all_records[split_idx:] if len(all_records) > 1 else all_records

    def _collect_records(self, dataset_dir):
        records = []
        if not os.path.isdir(dataset_dir):
            return records

        corrupt = set()
        corrupt_file = os.path.join(dataset_dir, self.corrupt_file)
        if os.path.exists(corrupt_file):
            with open(corrupt_file, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        corrupt.add(line)

        f20_imgs = sorted(glob(os.path.join(dataset_dir, "F20", "*")))
        for f20_path in f20_imgs:
            base = os.path.splitext(os.path.basename(f20_path))[0]
            if base in corrupt:
                continue

            l_path = os.path.join(dataset_dir, self.lr_dir, f"{base}_l.JPG")
            r_path = os.path.join(dataset_dir, self.lr_dir, f"{base}_r.JPG")
            d_path = os.path.join(dataset_dir, self.depth_dir, f"{base}.png")

            if not os.path.exists(l_path):
                l_path = os.path.join(dataset_dir, self.lr_dir, f"{base}_l.jpg")
            if not os.path.exists(r_path):
                r_path = os.path.join(dataset_dir, self.lr_dir, f"{base}_r.jpg")

            if not (os.path.exists(l_path) and os.path.exists(r_path) and os.path.exists(d_path)):
                continue

            records.append(
                {
                    "rgb": f20_path,
                    "l": l_path,
                    "r": r_path,
                    "d": d_path,
                }
            )
        return records

    def __len__(self):
        return len(self.records)

    def _read_record(self, rec):
        img_l = cv.imread(rec["l"], cv.IMREAD_COLOR)
        img_r = cv.imread(rec["r"], cv.IMREAD_COLOR)
        dep = cv.imread(rec["d"], cv.IMREAD_UNCHANGED)

        if img_l is None or img_r is None or dep is None:
            raise RuntimeError(f"Invalid sample files: {rec}")

        target_size = (self.resize_w, self.resize_h)
        img_l = cv.resize(img_l, target_size, interpolation=cv.INTER_AREA)
        img_r = cv.resize(img_r, target_size, interpolation=cv.INTER_AREA)
        dep = cv.resize(dep, target_size, interpolation=cv.INTER_NEAREST)

        img_l = cv.cvtColor(img_l, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_r = cv.cvtColor(img_r, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        dep = dep.astype(np.float32) / 1000.0  # mm -> m
        return img_l, img_r, dep

    @staticmethod
    def _map_lr_to_depth(ys, xs, lr_h, lr_w, d_h, d_w):
        """Map LR pixel coordinates to depth coordinates with nearest-neighbor scaling."""
        if lr_h <= 1:
            yd = np.zeros_like(ys, dtype=np.int64)
        else:
            yd = np.rint(ys * (d_h - 1) / (lr_h - 1)).astype(np.int64)

        if lr_w <= 1:
            xd = np.zeros_like(xs, dtype=np.int64)
        else:
            xd = np.rint(xs * (d_w - 1) / (lr_w - 1)).astype(np.int64)

        yd = np.clip(yd, 0, d_h - 1)
        xd = np.clip(xd, 0, d_w - 1)
        return yd, xd

    def _get_candidate_points(self, img_l, depth):
        """Build valid LR candidates and their depth values before random sampling."""
        lr_h, lr_w = img_l.shape[:2]
        d_h, d_w = depth.shape

        kernel = np.ones((self.depth_erode_ks, self.depth_erode_ks), dtype=np.float32)
        depth_eroded = cv.erode(depth, kernel)

        y0, y1 = self.margin, lr_h - self.margin
        x0, x1 = self.margin, lr_w - self.margin
        if y1 <= y0 or x1 <= x0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)

        ys_grid, xs_grid = np.meshgrid(
            np.arange(y0, y1, dtype=np.int64),
            np.arange(x0, x1, dtype=np.int64),
            indexing="ij",
        )
        ys = ys_grid.reshape(-1)
        xs = xs_grid.reshape(-1)

        yd, xd = self._map_lr_to_depth(ys, xs, lr_h, lr_w, d_h, d_w)
        dvals_eroded = depth_eroded[yd, xd]
        valid = (dvals_eroded >= self.d_min) & (dvals_eroded <= self.d_max)

        if not np.any(valid):
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)

        points = np.stack([ys[valid], xs[valid]], axis=1)
        dvals = depth[yd[valid], xd[valid]].astype(np.float32)
        return points, dvals

    def _sample_indices(self, num_candidates, num_points, rng):
        if num_candidates == 0:
            return np.zeros((0,), dtype=np.int64)
        replace = num_candidates < num_points
        return rng.choice(num_candidates, size=num_points, replace=replace)

    def _extract_patch(self, img, y, x):
        m = self.margin
        return img[y - m : y + m, x - m : x + m]

    def __getitem__(self, index):
        rec = self.records[index]
        img_l, img_r, depth = self._read_record(rec)

        if img_l.shape[:2] != img_r.shape[:2]:
            raise RuntimeError(f"L/R shape mismatch in sample: {rec}")

        rng = np.random.default_rng(self.seed + index)
        candidates, candidate_dvals = self._get_candidate_points(img_l, depth)

        if self.train:
            low, high = int(self.train_points[0]), int(self.train_points[1])
            num_points = int(rng.integers(low, high + 1))
        else:
            num_points = self.val_points

        sample_idx = self._sample_indices(len(candidates), num_points, rng)
        if len(sample_idx) == 0:
            lp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            rp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            dp = torch.zeros((1, 1), dtype=torch.float32)
            return lp, rp, dp

        points = candidates[sample_idx]
        sampled_dvals = candidate_dvals[sample_idx]

        l_patches = []
        r_patches = []
        d_vals = []
        for (y, x), d in zip(points, sampled_dvals):
            y = int(y)
            x = int(x)

            l_patch = self._extract_patch(img_l, y, x)
            r_patch = self._extract_patch(img_r, y, x)
            if l_patch.shape[:2] != (self.patch_size, self.patch_size):
                continue
            if r_patch.shape[:2] != (self.patch_size, self.patch_size):
                continue

            l_patches.append(np.transpose(l_patch, (2, 0, 1)))
            r_patches.append(np.transpose(r_patch, (2, 0, 1)))
            d_vals.append(d)

        if len(l_patches) == 0:
            lp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            rp = torch.zeros((1, 3, self.patch_size, self.patch_size), dtype=torch.float32)
            dp = torch.zeros((1, 1), dtype=torch.float32)
            return lp, rp, dp

        l_patches = torch.from_numpy(np.ascontiguousarray(np.stack(l_patches, axis=0))).float()
        r_patches = torch.from_numpy(np.ascontiguousarray(np.stack(r_patches, axis=0))).float()
        d_vals = torch.from_numpy(np.ascontiguousarray(np.array(d_vals, dtype=np.float32))).unsqueeze(1)
        return l_patches, r_patches, d_vals


class DPrealpatchF4(DPrealpatch):
    """Backward-compatible wrapper for F4 aperture real patch dataset."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("aperture", "F4")
        super(DPrealpatchF4, self).__init__(*args, **kwargs)


def collate_patch_samples(batch):
    """Flatten variable-N patch samples from scenes into [sumN, ...]."""
    l_list = [x[0] for x in batch]
    r_list = [x[1] for x in batch]
    d_list = [x[2] for x in batch]
    l_out = torch.cat(l_list, dim=0)
    r_out = torch.cat(r_list, dim=0)
    d_out = torch.cat(d_list, dim=0)
    return l_out, r_out, d_out
    
class ShotDataset(Dataset):
    def __init__(self, dataset_dir, resize=None):
        super(ShotDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.img_series = [img for img in glob(f'{dataset_dir}/*')]
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(resize, transforms.InterpolationMode.BICUBIC)
        ])

    def __len__(self):
            return len(self.img_series)
    
    def __getitem__(self, index):
        img_use = self.img_series[index]
        img_b = cv.cvtColor(cv.imread(f'{img_use}'), cv.COLOR_BGR2RGB) / 255.0
        img_b = self.transform(img_b.astype('float32'))
        return img_b

    
def depth2disp(depth, d_max=20, d_min=0.2):
    """Convert depth to disparity (inverse depth)"""
    a = 1/d_max
    b = 1/d_min - 1/d_max

    disp = np.zeros_like(depth, dtype=np.float32)
    valid = depth > 0
    d_min = np.min(depth[valid])
    disp[valid] = (1.0 / depth[valid] - a) / b    
    return disp

def random_crop_same(img1, img2, img3, img4, crop_h, crop_w):
    H, W = img1.shape[:2]

    assert img1.shape[:2] == img2.shape[:2] == img3.shape[:2] == img4.shape[:2]

    top = 0 # np.random.randint(0, H - crop_h + 1)
    left = 0 # np.random.randint(0, W - crop_w + 1)

    img1_c = img1[top:top+crop_h, left:left+crop_w]
    img2_c = img2[top:top+crop_h, left:left+crop_w]
    img3_c = img3[top:top+crop_h, left:left+crop_w]
    img4_c = img4[top:top+crop_h, left:left+crop_w]

    return img1_c, img2_c, img3_c, img4_c

def map_depth_to_range(depth, mask=None, d_min=1.0, d_max=10.0, eps=1e-6):
    """
    depth : torch.Tensor [B,1,H,W] or [H,W]
    mask  : torch.Tensor, bool or {0,1}, optional
    d_min : target minimum depth (meters)
    d_max : target maximum depth (meters)
    """

    out = torch.zeros_like(depth)

    if mask is None:
        valid = depth > 0
    else:
        valid = mask.bool() & (depth > 0)

    if not torch.any(valid):
        return out

    depth_valid = depth[valid]

    src_min = depth_valid.min()
    src_max = depth_valid.max()

    # 防止退化
    scale = (src_max - src_min).clamp(min=eps)

    out[valid] = d_min + (depth_valid - src_min) / scale * (d_max - d_min)

    return out

def disp2depth(disp, d_min=1.0, d_max=10.0, eps=1e-6):
    """
    Convert normalized disparity to depth.

    disp : torch.Tensor
           disparity map (assumed >= 0)
    d_min, d_max : float
           target depth range (meters)
    """

    disp = disp.float()

    a = 1.0 / d_max
    b = 1.0 / d_min - 1.0 / d_max

    depth = torch.zeros_like(disp)

    valid = disp >= 0

    denom = a + b * disp[valid]
    # denom = a + b * disp
    denom = torch.clamp(denom, min=eps)

    depth[valid] = 1.0 / denom

    return depth

if __name__ == "__main__":
    dataset_dir = "./dataset/ICCP2020_DP_dataset"
    a = ICCP2020(dataset_dir,resize=[512, 768])
    x = next(iter(a))
    a = 1


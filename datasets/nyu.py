import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import glob
import random
from PIL import Image
import tqdm

from utils import make_coord, add_noise

def to_pixel_samples(depth):
    """ Convert the image to coord-RGB pairs.
        depth: Tensor, (1, H, W)
    """
    coord = make_coord(depth.shape[-2:], flatten=True) # [H*W, 2]
    pixel = depth.view(-1, 1) # [H*W, 1]
    return coord, pixel

class NYUDataset(Dataset):
    def __init__(self, root='/data3/tang/nyu_labeled', split='train', scale=8, augment=True, downsample='bicubic', pre_upsample=False, to_pixel=False, sample_q=None, input_size=None, noisy=False):
        super().__init__()
        self.root = root
        self.split = split
        self.scale = scale
        self.augment = augment
        self.downsample = downsample
        self.pre_upsample = pre_upsample
        self.to_pixel = to_pixel
        self.sample_q = sample_q
        self.input_size = input_size
        self.noisy = noisy

        # use the first 1000 data as training split
        if self.split == 'train':
            self.size = 1000
        else:
            self.size = 449

    def __getitem__(self, idx):
        if self.split != 'train':
            idx += 1000

        image_file = os.path.join(self.root, 'RGB', f'{idx}.jpg')
        depth_file = os.path.join(self.root, 'Depth', f'{idx}.npy')             

        image = cv2.imread(image_file) # [H, W, 3]
        depth_hr = np.load(depth_file) # [H, W]
        
        # crop after rescale
        if self.input_size is not None:
            x0 = random.randint(0, image.shape[0] - self.input_size)
            y0 = random.randint(0, image.shape[1] - self.input_size)
            image = image[x0:x0+self.input_size, y0:y0+self.input_size]
            depth_hr = depth_hr[x0:x0+self.input_size, y0:y0+self.input_size]


        h, w = image.shape[:2]

        if self.downsample == 'bicubic':
            depth_lr = np.array(Image.fromarray(depth_hr).resize((w//self.scale, h//self.scale), Image.BICUBIC)) # bicubic, RMSE=7.13
            image_lr = np.array(Image.fromarray(image).resize((w//self.scale, h//self.scale), Image.BICUBIC)) # bicubic, RMSE=7.13
            #depth_lr = cv2.resize(depth_hr, (w//self.scale, h//self.scale), interpolation=cv2.INTER_CUBIC) # RMSE=8.03, cv2.resize is different from Image.resize.
        elif self.downsample == 'nearest-right-bottom':
            depth_lr = depth_hr[(self.scale - 1)::self.scale, (self.scale - 1)::self.scale] # right-bottom, RMSE=14.22, finally reproduced it...
            image_lr = image[(self.scale - 1)::self.scale, (self.scale - 1)::self.scale] # right-bottom, RMSE=14.22, finally reproduced it...
        elif self.downsample == 'nearest-center':
            depth_lr = np.array(Image.fromarray(depth_hr).resize((w//self.scale, h//self.scale), Image.NEAREST)) # center (if even, prefer right-bottom), RMSE=8.21
            image_lr = np.array(Image.fromarray(image).resize((w//self.scale, h//self.scale), Image.NEAREST)) # center (if even, prefer right-bottom), RMSE=8.21
        elif self.downsample == 'nearest-left-top':
            depth_lr = depth_hr[::self.scale, ::self.scale] # left-top, RMSE=13.94
            image_lr = image[::self.scale, ::self.scale] # left-top, RMSE=13.94
        else:
            raise NotImplementedError

        if self.noisy:
            depth_lr = add_noise(depth_lr, sigma=0.04, inv=False)

        # normalize
        depth_min = depth_hr.min()
        depth_max = depth_hr.max()
        depth_hr = (depth_hr - depth_min) / (depth_max - depth_min)
        depth_lr = (depth_lr - depth_min) / (depth_max - depth_min)
        
        image = image.astype(np.float32).transpose(2,0,1) / 255
        image_lr = image_lr.astype(np.float32).transpose(2,0,1) / 255 # [3, H, W]

        image = (image - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        image_lr = (image_lr - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

        # follow DKN, use bicubic upsampling of PIL
        depth_lr_up = np.array(Image.fromarray(depth_lr).resize((w, h), Image.BICUBIC))

        if self.pre_upsample:
            depth_lr = depth_lr_up

        # to tensor
        image = torch.from_numpy(image).float()
        image_lr = torch.from_numpy(image_lr).float()
        depth_hr = torch.from_numpy(depth_hr).unsqueeze(0).float()
        depth_lr = torch.from_numpy(depth_lr).unsqueeze(0).float()
        depth_lr_up = torch.from_numpy(depth_lr_up).unsqueeze(0).float()

        # transform
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                return x

            image = augment(image)
            image_lr = augment(image_lr)
            depth_hr = augment(depth_hr)
            depth_lr = augment(depth_lr)
            depth_lr_up = augment(depth_lr_up)

        image = image.contiguous()
        image_lr = image_lr.contiguous()
        depth_hr = depth_hr.contiguous()
        depth_lr = depth_lr.contiguous()
        depth_lr_up = depth_lr_up.contiguous()

        # to pixel
        if self.to_pixel:
            
            hr_coord, hr_pixel = to_pixel_samples(depth_hr)

            lr_pixel = depth_lr_up.view(-1, 1)

            if self.sample_q is not None:
                sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
                hr_coord = hr_coord[sample_lst]
                hr_pixel = hr_pixel[sample_lst]
                lr_pixel = lr_pixel[sample_lst]

            cell = torch.ones_like(hr_coord)
            cell[:, 0] *= 2 / depth_hr.shape[-2]
            cell[:, 1] *= 2 / depth_hr.shape[-1]
        
            return {
                'image': image,
                'lr_image': image_lr,
                'lr': depth_lr,
                'hr': hr_pixel,
                'hr_depth': depth_hr,
                'lr_pixel': lr_pixel,
                'hr_coord': hr_coord,
                'min': depth_min * 100,
                'max': depth_max * 100,
                'cell': cell,
                'idx': idx,
            }   

        else:
            return {
                'image': image,
                'lr': depth_lr,
                'hr': depth_hr,
                'min': depth_min * 100,
                'max': depth_max * 100,
                'idx': idx,
            }

    def __len__(self):
        return self.size


if __name__ == '__main__':
    print('===== test direct bicubic upsampling =====')
    for method in ['bicubic']:
        for scale in [4, 8, 16]:
            print(f'[INFO] scale = {scale}, method = {method}')
            d = NYUDataset(root='/data3/tang/nyu_labeled', split='test', pre_upsample=True, augment=False, scale=scale, downsample=method, noisy=False)
            #d = NYUDataset(root='/data3/tang/nyu_labeled', split='test', pre_upsample=True, augment=False, scale=scale, downsample=method, noisy=True)
            rmses = []
            for i in tqdm.trange(len(d)):
                x = d[i]
                lr = ((x['lr'].numpy() * (x['max'] - x['min'])) + x['min'])
                hr = ((x['hr'].numpy() * (x['max'] - x['min'])) + x['min'])
                rmse = np.sqrt(np.mean(np.power(lr - hr, 2)))
                rmses.append(rmse)
            print('RMSE = ', np.mean(rmses))
import os
from pathlib import Path
import random
import jittor as jt
from jittor.dataset import Dataset
import numpy as np
import scipy.io as sio
from numba import jit
import cv2


class SceneNetDataset(Dataset):
    def getfiles_ordered(self, folder):
        files = sorted(os.listdir(folder), key=lambda x: int(Path(x).stem))
        return [folder / x for x in files]

    def __init__(self, opt):
        super(SceneNetDataset, self).__init__()
        self.dataroot = Path(opt.dataroot)
        self.num_labels = 14
        self.ignore_label = 0
        self.use_dhac = opt.use_dhac
        self.depth_folder = 'dhac2' if self.use_dhac else 'depth'
        self.default_size = (320, 240)
        self.default_shape = (240, 320)

        self.rgb_frames = self.getfiles_ordered(self.dataroot / 'photo')
        self.depth_frames = self.getfiles_ordered(self.dataroot / self.depth_folder)
        self.labels = self.getfiles_ordered(self.dataroot / 'label')

        self.total_frames = len(self.rgb_frames)
        self.set_attrs(batch_size=1, total_len=self.total_frames, shuffle=False, num_workers=8)

    def fetch_depth(self, path):
        depth_image = cv2.imread(str(path), -1)
        if depth_image.shape[:2] != self.default_shape:
            depth_image = cv2.resize(depth_image, self.default_size, interpolation=cv2.INTER_NEAREST)
        depth_min = depth_image.min() - 0.01
        depth_max = depth_image.max() + 0.01
        depth_image = (depth_image - depth_min) / (depth_max - depth_min) * 255
        depth_image = depth_image.astype(np.uint8)
        depth_image = depth_image[:, :, np.newaxis]
        return depth_image

    def fetch_dhac(self, path):
        depth_image = cv2.imread(str(path), -1).astype(np.uint8)
        if depth_image.shape[:2] != self.default_shape:
            depth_image = cv2.resize(depth_image, self.default_size, interpolation=cv2.INTER_NEAREST)
        return depth_image

    def fetch_rgb(self, path):
        rgb_image = cv2.imread(str(path), -1).astype(np.uint8)
        if rgb_image.shape[:2] != self.default_shape:
            rgb_image = cv2.resize(rgb_image, self.default_size, interpolation=cv2.INTER_LINEAR)
        return rgb_image

    def fetch_label(self, path):
        label = cv2.imread(str(path), -1).astype(np.int32)
        if label.shape[:2] != self.default_shape:
            label = cv2.resize(label, self.default_size, interpolation=cv2.INTER_NEAREST)
        return label

    def __getitem__(self, index):

        rgb_image = self.fetch_rgb(self.rgb_frames[index])

        if self.use_dhac:
            depth_image = self.fetch_dhac(self.depth_frames[index])
        else:
            depth_image = self.fetch_depth(self.depth_frames[index])

        # print(rgb_image.shape, depth_image.shape)

        if self.labels:
            label = self.fetch_label(self.labels[index])
        else:  # test phase
            label = np.zeros(self.default_shape, dtype=np.int32)

        rgb_image = (rgb_image.transpose(2, 0, 1) / 255.0).astype(np.float32)
        depth_image = (depth_image.transpose(2, 0, 1) / 255.0).astype(np.float32)

        return rgb_image, depth_image, label, int(self.rgb_frames[index].stem)

    def __len__(self):
        return self.total_frames

    def name(self):
        return 'SceneNet_Scan'
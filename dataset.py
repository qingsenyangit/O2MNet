# -*- coding:utf-8 _*-
"""
@author: LiuZhen
@license: Apache Licence
@file: dataset.py
@time: 2021/08/02
@contact: liuzhen.pwd@gmail.com
@site:
@software: PyCharm

"""
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import glob
import os.path as osp
from utils.utils import *
import random
import cv2


class NTIRE_Training_Dataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = osp.join(root_dir, 'crop_training_p128_s64_siggraph17')

        self.image_ids = [i for i in range(22792)]

        self.image_list = []

        for image_id in self.image_ids:
            exposures_path = osp.join(self.root_dir, "{:06d}_exposures.npy".format(image_id))  # chen
            align_ratio_path = ''  # chen
            image_short_path = os.path.join(self.root_dir, "{:06d}_short.tif".format(image_id))  # chen
            image_medium_path = os.path.join(self.root_dir, "{:06d}_medium.tif".format(image_id))  # chen
            image_long_path = os.path.join(self.root_dir, "{:06d}_long.tif".format(image_id))  # chen
            image_gt_path = os.path.join(self.root_dir, "{:06d}_gt.hdr".format(image_id))  # chen
            image_path = [image_short_path, image_medium_path, image_long_path, image_gt_path]
            self.image_list += [[exposures_path, align_ratio_path, image_path]]

    def __getitem__(self, index):
        # Read exposure times and alignratio
        exposures = np.load(self.image_list[index][0])
        floating_exposures = exposures - exposures[1]
        # Read LDR images
        ldr_images = ReadImages(self.image_list[index][2][:-1])

        # Read HDR label
        label = imread_uint16_png(self.image_list[index][2][-1])

        ldr_images[0], ldr_images[1], ldr_images[2], label = self.image_Geometry_Aug(ldr_images, label)

        # ldr images process
        s_gamma = 2.2  # chen

        img_medium_gamma = gamma_correction(ldr_images[1], exposures[1], s_gamma)
        img_short_gamma = gamma_correction(ldr_images[0], exposures[0], s_gamma)
        img_long_gamma = gamma_correction(ldr_images[2], exposures[2], s_gamma)

        image_short_concat = np.concatenate((ldr_images[0], img_short_gamma), 2)
        image_medium_concat = np.concatenate((ldr_images[1], img_medium_gamma), 2)
        image_long_concat = np.concatenate((ldr_images[2], img_long_gamma), 2)


        img0 = image_short_concat.astype(np.float32).transpose(2, 0, 1)
        img1 = image_medium_concat.astype(np.float32).transpose(2, 0, 1)
        img2 = image_long_concat.astype(np.float32).transpose(2, 0, 1)
        label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)
        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'label': label}
        return sample

    def __len__(self):
        return len(self.image_ids)

    def random_number(self, num):
        return random.randint(1, num)

    def image_Geometry_Aug(self, data, label):
        short = data[0]
        mid = data[1]
        long = data[2]
        h, w, c = short.shape
        num = self.random_number(4)

        if num == 1:
            in_data_short = short
            in_data_mid = mid
            in_data_long = long
            in_label = label

        if num == 2:  # flip_left_right
            index = np.arange(w, 0, -1) - 1
            in_data_short = short[:, index, :]
            in_data_mid = mid[:, index, :]
            in_data_long = long[:, index, :]
            in_label = label[:, index, :]

        if num == 3:  # flip_up_down
            index = np.arange(h, 0, -1) - 1
            in_data_short = short[index, :, :]
            in_data_mid = mid[index, :, :]
            in_data_long = long[index, :, :]
            in_label = label[index, :, :]

        if num == 4:  # rotate 180
            index = np.arange(w, 0, -1) - 1
            in_data_short = short[:, index, :]
            in_data_mid = mid[:, index, :]
            in_data_long = long[:, index, :]
            in_label = label[:, index, :]
            index = np.arange(h, 0, -1) - 1
            in_data_short = in_data_short[index, :, :]
            in_data_mid = in_data_mid[index, :, :]
            in_data_long = in_data_long[index, :, :]
            in_label = in_label[index, :, :]

        return in_data_short, in_data_mid, in_data_long, in_label


class NTIRE_Validation_Dataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = osp.join(root_dir, 'test_siggraph17')
        self.val_samples = [i for i in range(15)]
        # self.val_samples = [i for i in range(4)]
        self.image_list = []

        for image_id in self.val_samples:
            exposures_path = osp.join(self.root_dir, "{:04d}_exposures.npy".format(image_id))
            align_ratio_path = ''
            image_short_path = os.path.join(self.root_dir, "{:04d}_short.tif".format(image_id))
            image_medium_path = os.path.join(self.root_dir, "{:04d}_medium.tif".format(image_id))
            image_long_path = os.path.join(self.root_dir, "{:04d}_long.tif".format(image_id))
            image_gt_path = os.path.join(self.root_dir, "{:04d}_gt.hdr".format(image_id))
            image_path = [image_short_path, image_medium_path, image_long_path, image_gt_path]
            self.image_list += [[exposures_path, align_ratio_path, image_path]]

    def __getitem__(self, index):
        # Read exposure times and alignratio
        exposures = np.load(self.image_list[index][0])
        floating_exposures = exposures - exposures[1]
        # Read LDR images
        ldr_images = ReadImages(self.image_list[index][2][:-1])

        # Read HDR label
        label = imread_uint16_png(self.image_list[index][2][-1])
        # ldr images process
        s_gamma = 2.2  # chen

        img_medium_gamma = gamma_correction(ldr_images[1], exposures[1], s_gamma)
        img_short_gamma = gamma_correction(ldr_images[0], exposures[0], s_gamma)
        img_long_gamma = gamma_correction(ldr_images[2], exposures[2], s_gamma)

        image_short_concat = np.concatenate((ldr_images[0], img_short_gamma), 2)
        image_medium_concat = np.concatenate((ldr_images[1], img_medium_gamma), 2)
        image_long_concat = np.concatenate((ldr_images[2], img_long_gamma), 2)


        img0 = image_short_concat.astype(np.float32).transpose(2, 0, 1)
        img1 = image_medium_concat.astype(np.float32).transpose(2, 0, 1)
        img2 = image_long_concat.astype(np.float32).transpose(2, 0, 1)
        label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)
        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'label': label}
        return sample

    def __len__(self):
        return len(self.val_samples)


class NTIRE_trainVal_Dataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = osp.join(root_dir, 'crop_training_p128_s64_noBoundary')
        self.val_samples = [i for i in range(100)]
        self.image_list = []

        for image_id in self.val_samples:
            exposures_path = osp.join(self.root_dir, "{:06d}_exposures.npy".format(image_id))
            align_ratio_path = ''
            image_short_path = os.path.join(self.root_dir, "{:06d}_short.tif".format(image_id))
            image_medium_path = os.path.join(self.root_dir, "{:06d}_medium.tif".format(image_id))
            image_long_path = os.path.join(self.root_dir, "{:06d}_long.tif".format(image_id))
            image_gt_path = os.path.join(self.root_dir, "{:06d}_gt.hdr".format(image_id))
            image_path = [image_short_path, image_medium_path, image_long_path, image_gt_path]
            self.image_list += [[exposures_path, align_ratio_path, image_path]]

    def __getitem__(self, index):
        # Read exposure times and alignratio
        exposures = np.load(self.image_list[index][0])
        floating_exposures = exposures - exposures[1]
        # Read LDR images
        ldr_images = ReadImages(self.image_list[index][2][:-1])

        # Read HDR label
        label = imread_uint16_png(self.image_list[index][2][-1])
        # ldr images process
        s_gamma = 2.2  # chen

        img_medium_gamma = gamma_correction(ldr_images[1], exposures[1], s_gamma)
        img_short_gamma = gamma_correction(ldr_images[0], exposures[0], s_gamma)
        img_long_gamma = gamma_correction(ldr_images[2], exposures[2], s_gamma)

        image_short_concat = np.concatenate((ldr_images[0], img_short_gamma), 2)
        image_medium_concat = np.concatenate((ldr_images[1], img_medium_gamma), 2)
        image_long_concat = np.concatenate((ldr_images[2], img_long_gamma), 2)


        img0 = image_short_concat.astype(np.float32).transpose(2, 0, 1)
        img1 = image_medium_concat.astype(np.float32).transpose(2, 0, 1)
        img2 = image_long_concat.astype(np.float32).transpose(2, 0, 1)
        label = label.astype(np.float32).transpose(2, 0, 1)

        img0 = torch.from_numpy(img0)
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        label = torch.from_numpy(label)
        sample = {'input0': img0, 'input1': img1, 'input2': img2, 'label': label}
        return sample

    def __len__(self):
        return len(self.val_samples)

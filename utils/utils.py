#-*- coding:utf-8 _*-  
""" 
@author: LiuZhen
@license: Apache Licence 
@file: utils.py 
@time: 2020/06/30
@contact: liuzhen.pwd@gmail.com
@site:  
@software: PyCharm 

"""
import numpy as np
import os, glob
import cv2
import imageio
from math import log10
import torch, math
import torch.nn as nn
import torch.nn.init as init
#from skimage.measure.simple_metrics import compare_psnr
# imageio.plugins.freeimage.download()


# def list_all_files_sorted(folderName, extension=""):
#     return sorted(glob.glob(os.path.join(folderName, "*" + extension)))
#
#
# def ReadExpoTimes(fileName):
#     return np.power(2, np.loadtxt(fileName))
#
#
# def ReadImages(fileNames):
#     imgs = []
#     for imgStr in fileNames:
#         img = cv2.imread(imgStr, -1)
#
#         # equivalent to im2single from Matlab
#         img = img / 2 ** 16
#         img = np.float32(img)[:, :, [2, 1, 0]]
#
#         img.clip(0, 1)
#
#         imgs.append(img)
#     return np.array(imgs)
#
#
# def ReadLabel(fileName):
#     label = imageio.imread(os.path.join(fileName, 'HDRImg.hdr'), 'hdr')
#     # label = label[:, :, [2, 1, 0]]  ##cv2
#     return label
#
#
# def LDR_to_HDR(imgs, expo, gamma):
#     return (imgs ** gamma) / expo




def ReadImages(fileNames):
    imgs = []
    for imgStr in fileNames:
        img = cv2.cvtColor(cv2.imread(imgStr),
                           cv2.COLOR_BGR2RGB) / 255.0   #chen
        imgs.append(img)
    return np.array(imgs)

def imread_uint16_png(image_path):
    # Load image without changing bit depth
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)  #chen

def ev_alignment(img, expo, gamma):
    return ((img ** gamma) * 2.0**expo)**(1/gamma)  #chen

def gamma_correction(img, expo, gamma):
    return (img ** gamma) / 2.0**expo  #chen


def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)


def range_compressor_tensor(x):
    const_1 = torch.from_numpy(np.array(1.0)).cuda()
    const_5000 = torch.from_numpy(np.array(5000.0)).cuda()
    return (torch.log(const_1 + const_5000 * x)) / torch.log(const_1 + const_5000)


def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        # PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
        PSNR += PSNR1(Iclean[i,:,:,:], Img[i,:,:,:])
    return (PSNR/Img.shape[0])


# def batch_PSNRMu(img, imclean, data_range):
#     Img = img.data.cpu().numpy().astype(np.float32)
#     Iclean = imclean.data.cpu().numpy().astype(np.float32)
#     PSNR = 0
#     Img = np.log(1 + 5000 * Img) / np.log((np.array([1 + 5000])))
#     Iclean = np.log(1 + 5000 * Iclean) / np.log((np.array([1 + 5000])))
#     print('out image')
#     import matplotlib.image as mpimg
#     mpimg.imsave("new_panda.jpg", Img[0,:,500:600,500:600].transpose(1,2,0))
#     print(Img.shape)
#     cv2.imwrite('/home/qingsen/HDR/AHDRNet-master/res.jpg', Img[0,:,:,:].transpose(1,2,0) * 255)
#     cv2.imwrite('/home/qingsen/HDR/AHDRNet-master/ground.jpg', Iclean[0,:,:,:].transpose(1,2,0) * 255)
#
#     for i in range(Img.shape[0]):
#         PSNR += PSNR1(Iclean[i,:,:,:], Img[i,:,:,:])
#     return (PSNR/Img.shape[0])


def PSNR1(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    return 20 * math.log10(1/math.sqrt(mse))




def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr #* (0.1 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_parameters(net):
    """Init layer parameters"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


def set_random_seed(seed):
    """Set random seed for reproduce"""
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


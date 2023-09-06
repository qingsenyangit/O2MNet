import os
import random
import time
import numpy as np
import torch
import math
import cv2

import torch.nn as nn
import torchvision as tv
import matplotlib.pyplot as plt
import torch.utils.data as data
import torch.nn.functional as F
from utils.utils import *


from glob import glob
from torch.autograd import Variable
from model import ELBO

def mk_trained_dir_if_not(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def model_restore(model, trained_model_dir):
    model_list = glob(trained_model_dir + "/*.pkl")
    a = []
    for i in range(len(model_list)):
        try:
            index = int(model_list[i].split('checkpoint')[-1].split('.')[0])
            a.append(index)
        except:
            continue
    epoch = np.sort(a)[-1]
    model_path = trained_model_dir + 'checkpoint{}.pkl'.format(epoch)
    # model_path ='trained-model/trained_model20320.pkl'
    model.load_state_dict(torch.load(model_path))
    return model, epoch


# class data_loader(data.Dataset):
#     def __init__(self, list_dir):
#         f = open(list_dir)
#         self.list_txt = f.readlines()
#         self.length = len(self.list_txt)
#
#     def __getitem__(self, index):
#
#         sample_path = self.list_txt[index][:-1]
#
#         if os.path.exists(sample_path):
#
#             f = h5py.File(sample_path, 'r')
#             # batch = 200
#             # batch_start = random.randint(1, f['IN'].shape[0]-batch)
#             # batch_end = batch_start + batch
#             # data = f['IN'][batch_start:batch_end, :, :, :]
#             # label = f['GT'][batch_start:batch_end, :, :, :]
#             # f.close()
#
#             data = f['IN'][:]
#             label = f['GT'][:]
#             f.close()
#             crop_size = 256
#             data, label = self.imageCrop(data, label, crop_size)
#             data, label = self.image_Geometry_Aug(data, label)
#
#
#         # print(sample_path)
#         return torch.from_numpy(data).float(), torch.from_numpy(label).float()
#
#     def __len__(self):
#         return self.length
#
#     def random_number(self, num):
#         return random.randint(1, num)
#
#     def imageCrop(self, data, label, crop_size):
#         c, w, h = data.shape
#         w_boder = w - crop_size  # sample point y coordinet
#         h_boder = h - crop_size  # sample point x ...
#
#         start_w = self.random_number(w_boder - 1)
#         start_h = self.random_number(h_boder - 1)
#
#         crop_data = data[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
#         crop_label = label[:, start_w:start_w + crop_size, start_h:start_h + crop_size]
#         return crop_data, crop_label
#
#     def image_Geometry_Aug(self, data, label):
#         c, w, h = data.shape
#         num = self.random_number(4)
#
#         if num == 1:
#             in_data = data
#             in_label = label
#
#         if num == 2:  # flip_left_right
#             index = np.arange(w, 0, -1) - 1
#             in_data = data[:, index, :]
#             in_label = label[:, index, :]
#
#         if num == 3:  # flip_up_down
#             index = np.arange(h, 0, -1) - 1
#             in_data = data[:, :, index]
#             in_label = label[:, :, index]
#
#         if num == 4:  # rotate 180
#             index = np.arange(w, 0, -1) - 1
#             in_data = data[:, index, :]
#             in_label = label[:, index, :]
#             index = np.arange(h, 0, -1) - 1
#             in_data = in_data[:, :, index]
#             in_label = in_label[:, :, index]
#
#         return in_data, in_label

def get_lr(epoch, lr, max_epochs):
    if epoch <= max_epochs * 0.8:
        lr = lr
    # elif epoch <= max_epochs * 0.8:
    #     lr = 0.1 * lr
    else:
        lr = 0.1 * lr
    return lr

def train(epoch, model, train_loaders, optimizer, args, elbo):
    lr = get_lr(epoch, args.lr, args.epochs)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('lr: {}'.format(optimizer.param_groups[0]['lr']))
    kl_weight = args.kl_weight

    model.train()
    num = 0
    trainloss = 0
    start = time.time()
    for batch_idx, batch_data in enumerate(train_loaders):
        data1, data2, data3 = batch_data['input0'].cuda(), batch_data['input1'].cuda(), \
                                             batch_data['input2'].cuda()
        target = batch_data['label'].cuda()
        end = time.time()

############  used for End-to-End code
        # data1 = torch.cat((data[0, :, 0:3, :], data[0, :, 9:12, :]), dim=1)
        # data2 = torch.cat((data[0, :, 3:6, :], data[0, :, 12:15, :]), dim=1)
        # data3 = torch.cat((data[0, :, 6:9, :], data[0, :, 15:18, :]), dim=1)

        # data1 = torch.cat((data[:, 0:3, :, :], data[:, 9:12, :, :]), dim=1)
        # data2 = torch.cat((data[:, 3:6, :, :], data[:, 12:15, :, :]), dim=1)
        # data3 = torch.cat((data[:, 6:9, :, :], data[:, 15:18, :, :]), dim=1)
        #
        # data1 = Variable(data1)
        # data2 = Variable(data2)
        # data3 = Variable(data3)
        # target = Variable(target)
        optimizer.zero_grad()
        output = model(data1, data2, data3)

        # loss, kl, mse, sparsity = elbo(output, target, kl_weight)
        loss, kl, mse = elbo(output, target, kl_weight)

        # num += 1
        # print('iterate num: {}'.format(num))
#########  make the loss
        # output = torch.log(1 + 5000 * output.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        # target = torch.log(1 + 5000 * target).cpu() / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        #
        # loss = F.l1_loss(output, target)

        loss.backward()
        optimizer.step()
        trainloss = trainloss + loss
        if (batch_idx +1) % 4 == 0:
            trainloss = trainloss / 4
            print('train Epoch {} iteration: {} loss: {:.6f} MSEloss: {:.6f}'.format(epoch, batch_idx, trainloss.data, mse.data))
            fname = args.trained_model_dir + 'lossTXT.txt'
            try:
                fobj = open(fname, 'a')

            except IOError:
                print('open error')
            else:
                fobj.write('train Epoch {} iteration: {} Loss: {:.6f}\n'.format(epoch, batch_idx, trainloss.data))
                fobj.close()
            trainloss = 0

        # print('time: {}'.format(end-start))

def test(model, test_loaders, args,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    #num = 0
    n_val = len(test_loaders)

    val_psnr = 0
    val_psnrMu = 0
    for batch_idx, batch_data in enumerate(test_loaders):

        with torch.no_grad():
            data1, data2, data3 = batch_data['input0'].cuda(), batch_data['input1'].cuda(), \
                                  batch_data['input2'].cuda()
            target1 = batch_data['label'].cuda()

           # weightS = 1408
           # highS = 896
           # PointW = int((data1.shape[2] - weightS)/2)
           # PointH = int((data1.shape[3] - highS)/2)
           # data1 = data1[:, :, PointW:(data1.shape[2]-PointW), PointH:(data1.shape[3]-PointH)]
           # data2 = data2[:, :, PointW:(data2.shape[2]-PointW), PointH:(data2.shape[3]-PointH)]
           # data3 = data3[:, :, PointW:(data3.shape[2]-PointW), PointH:(data3.shape[3]-PointH)]
           # target1 = target[:, :, PointW:(target.shape[2]-PointW), PointH:(target.shape[3]-PointH)]
            output = model(data1, data2, data3)


        psnr = batch_PSNR(output, target1, 1.0)
        print('Validation set: PSNR: {:.4f}'.format(psnr))

        output = range_compressor_tensor(output)
        target = range_compressor_tensor(target1)
        psnrMu = batch_PSNR(output, target, 1.0)
        print('Validation set: PSNRMu: {:.4f}'.format(psnrMu))

        val_psnr += psnr
        val_psnrMu += psnrMu





        # output1 = torch.log(1 + 5000 * output) / torch.log(1 + output)
        # target = torch.log(1 + 5000 * target) / torch.log(1 + target)
        #
        # test_loss += F.mse_loss(output1, target)
        #num = num + 1

        # test_loss += F.smooth_l1_loss(output, target, size_average=False).data[0]

    val_psnr /= n_val
    val_psnrMu /= n_val
    print('Validation set: Average PSNR: {:.4f}'.format(val_psnr))
    print('Validation set: Average PSNRMu: {:.4f}\n'.format(val_psnrMu))

    fname = args.trained_model_dir + 'PSNR.txt'
    try:
        fobj = open(fname, 'a')

    except IOError:
        print('open error')
    else:
        fobj.write('epoch:{}  PSNR: {:.4f}, PSNRMu: {:.4f}\n'.format(epoch,val_psnr, val_psnrMu))
        fobj.close()


    #test_loss = test_loss.cpu().data / len(test_loaders.dataset)
    # print('\n Test set: Average Loss: {:.4f}'.format(test_loss.data[0]))
    if args.mode == 'test':
        return test_loss, output
    else:
        return test_loss, val_psnr, val_psnrMu



# class image_testdata_loader(data.Dataset):
#     def __init__(self, list_dir):
#         f = open(list_dir)
#         self.list_txt = f.readlines()
#         self.length = len(self.list_txt)
#
#     def __getitem__(self, index):
#
#         sample_path = self.list_txt[index][:-1]
#
#         if os.path.exists(sample_path):
#             f = h5py.File(sample_path, 'r')
#             data = f['IN'][:]
#             label = f['GT'][:]
#
#             f.close()
#         # print(sample_path)
#         return torch.from_numpy(data).float(), torch.from_numpy(label).float()
#
#     def __len__(self):
#         return self.length
#
#     def random_number(self, num):
#         return random.randint(1, num)


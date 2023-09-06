
import torch
import numpy as np
import time
import argparse
import torch.optim as optim
import torch.utils.data
import scipy.io as scio

# from model import SigUNet
# from model import U_Net
# from Multi_Dense import Multi_Dense_Net
# from model import HDRDense_Net
# from Multi_Unet import Multi_UNet
from model import *
from torch.nn import init
from dataset import NTIRE_Training_Dataset,NTIRE_Validation_Dataset
from torch.utils.data import DataLoader


from function_for_train import *


# hyper-parameters


parser = argparse.ArgumentParser(description='End to end for HDR')
# parser.add_argument('--train-data', default='/home/yan/yqs/HDR/dataSets/no_flow/Training.h5')
# parser.add_argument('--test-data', default='/home/yan/yqs/HDR/dataSets/no_flow/Training.h5')
parser.add_argument('--train-data', default='train.txt')
parser.add_argument('--test-data', default='test.txt')
parser.add_argument('--test_whole_Image', default='./test.txt')


parser.add_argument('--dataset_dir', default='../data/')

parser.add_argument('--trained_model_dir', default='./trained-model/')
parser.add_argument('--result_dir', default='./result/')
parser.add_argument('--mode', default='train')
parser.add_argument('--use_cuda', default=True)
parser.add_argument('--restore', default=True)
parser.add_argument('--batchsize', default=8)
parser.add_argument('--epochs', default=800000)
parser.add_argument('--kl_weight', default=1e-8)
parser.add_argument('--lr', default=1e-4)
parser.add_argument('--momentum', default=0.9)
parser.add_argument('--seed', default=1)
parser.add_argument('--save_model_interval', default=100000)

parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')
parser.add_argument('--scale', type=int, default= 1, help='scale output size /input size')


args = parser.parse_args()

torch.manual_seed(args.seed)
if args.use_cuda and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

#load data

train_dataset = NTIRE_Training_Dataset(root_dir=args.dataset_dir)
train_loaders = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, pin_memory=True)
val_dataset = NTIRE_Validation_Dataset(root_dir=args.dataset_dir)
Image_test_loaders = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)



# data_loader(args.test_data)
# train_loaders = torch.utils.data.DataLoader(
#     data_loader(args.train_data),
#     batch_size=args.batchsize, shuffle=True, num_workers=4)
# test_loaders = torch.utils.data.DataLoader(
#     data_loader(args.test_data),
#     batch_size=args.batchsize, num_workers=4)
#
# Image_test_loaders = torch.utils.data.DataLoader(
#     image_testdata_loader(args.test_whole_Image),
#     batch_size=1)
# ##################  used for all the dataset in .h5
# train_set = DatasetFromHdf5(args.train_data)
# train_loaders = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batchsize, shuffle=True)



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data)

#make dir of trained model
mk_trained_dir_if_not(args.trained_model_dir)

mk_trained_dir_if_not(args.result_dir)

model = RDN(args)
model.apply(weights_init_kaiming)
elbo = ELBO(model, len(train_loaders.dataset)).cuda()

# model = Net()

if args.use_cuda:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-7)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

start_step = 0
if args.restore and len(os.listdir(args.trained_model_dir)):
    model, start_step = model_restore(model, args.trained_model_dir)
    print('restart from {} step'.format(start_step))


# when we test, we need test on the whole image, so we defind a new variable
#  'Image_test_loaders' used to load the whole image
if args.mode == 'test':
    args.batchsize = 1
    start = time.time()
    loss, outImage = test(model, Image_test_loaders, args,-1)

    end = time.time()
    print('cost {} seconds'.format(end - start))



else:
    bestpsnr = 0
    besepsnrMu = 0
    for epoch in range(start_step + 1, args.epochs + 1):
        start = time.time()
        train(epoch, model, train_loaders, optimizer, args, elbo)
        end = time.time()
        print('epoch:{}, cost {} seconds'.format(epoch, end - start))

        outImage, psnr, mu = test(model, Image_test_loaders, args,epoch)
        model_name = args.trained_model_dir + 'val_latest.pkl'
        torch.save(model.state_dict(), model_name)
        if psnr>bestpsnr:
            model_name = args.trained_model_dir + 'best_psnr.pkl'
            torch.save(model.state_dict(), model_name)
            bestpsnr = psnr
        if mu>besepsnrMu:
            model_name = args.trained_model_dir + 'best_mu.pkl'
            torch.save(model.state_dict(), model_name)
            besepsnrMu = mu
        if epoch % 20 == 0:
            model_name = args.trained_model_dir + 'checkpoint'+str(epoch)+'.pkl'
            torch.save(model.state_dict(), model_name)

        # if epoch % args.save_model_interval == 0:
        #     model_name = args.trained_model_dir + 'trained_model{}.pkl'.format(epoch)
        #     torch.save(model.state_dict(), model_name)
        #
        #     outImage, psnr, mu = test(model, Image_test_loaders, args)
            #if psnr>bestpsnr:
                #model_name = args.trained_model_dir + 'best.pkl'
                #torch.save(model.state_dict(), model_name)




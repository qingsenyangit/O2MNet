from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os.path as osp
import os
from utils.utils import *
# modify uncertainty model...
from model import RDN
# modify uncertainty model...
import torch.nn.functional as F
import math
from utils.utils import psnr, batch_PSNR
from tqdm import tqdm
import cv2

def get_args():
    parser = argparse.ArgumentParser(description='ADNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=int, default=0,
                        help="dataset: ntire dataset")
	# modify dataset path...
    parser.add_argument("--dataset_dir", type=str, default='../data/test_siggraph17',
                        help='dataset directory')
    # modify dataset path...

	# modify ckpt path...
    parser.add_argument('--logdir', type=str, default='ablation_BasicRFB_DRDB/',
                        help='target log directory')
	# modify ckpt path...
	
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='number of workers to fetch data (default: 8)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                        help='testing batch size (default: 1)')
    parser.add_argument("--preddir", type=str, default='prediction')


    parser.add_argument('--nDenselayer', type=int, default=6, help='nDenselayer of RDB')
    parser.add_argument('--growthRate', type=int, default=32, help='growthRate of dense net')
    parser.add_argument('--nBlock', type=int, default=16, help='number of RDB block')
    parser.add_argument('--nFeat', type=int, default=64,  help='number of feature maps')
    parser.add_argument('--nChannel', type=int, default=6, help='number of color channels to use')
    parser.add_argument('--scale', type=int, default= 1, help='scale output size /input size')

    return parser.parse_args()


class NTIRE_Testing_Dataset(Dataset):

    def __init__(self, root_dir):
		# modify dataset path...
        self.root_dir = root_dir
		# modify dataset path...
        self.val_samples = [i for i in range(15)]
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

        image_short_concat = np.concatenate((ldr_images[0],img_short_gamma), 2)
        image_medium_concat = np.concatenate((ldr_images[1],img_medium_gamma), 2)
        image_long_concat = np.concatenate((ldr_images[2],img_long_gamma), 2)

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

def HDR2LDR(pred):
    mu=5000
    pred_tonemap = np.log(1 + mu * pred) / np.log(1 + mu)
    pred_RGB = pred_tonemap[:, :, ::-1]
    pred_int8 = pred_RGB * 255

    return pred_int8


def prediction(args, model, device, test_loader, savedir):
    model.eval()
    n_val = len(test_loader)
    val_psnr = 0
    val_mulaw = 0

    with torch.no_grad():
        for batch_idx, batch_data in tqdm(enumerate(test_loader)):
            batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), \
                                                 batch_data['input1'].to(device), \
                                                 batch_data['input2'].to(device)
            label = batch_data['label'].to(device)
            batch_ldr0 = F.pad(batch_ldr0, [2, 2, 0, 0], 'constant', 0)
            batch_ldr1 = F.pad(batch_ldr1, [2, 2, 0, 0], 'constant', 0)
            batch_ldr2 = F.pad(batch_ldr2, [2, 2, 0, 0], 'constant', 0)
            pred = model(batch_ldr0, batch_ldr1, batch_ldr2)
            pred = pred[:, :, ::, 2:1502]

            psnr_pred = torch.squeeze(pred.clone())
            psnr_label = torch.squeeze(label.clone())
            psnr_pred = psnr_pred.data.cpu().numpy().astype(np.float32)  # chen
            psnr_label = psnr_label.data.cpu().numpy().astype(np.float32)

            pred_rgb = psnr_pred.transpose(1,2,0)
            pred_rgb = pred_rgb[:,:,::-1]
			
			# modify dataset path...
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            cv2.imwrite(savedir+'/'+str(batch_idx)+'.hdr',pred_rgb)
			# modify dataset path...

            # psnr_ = psnr(psnr_pred, psnr_label)  # chen
            # mu_law = batch_PSNR(psnr_pred, psnr_label)  # chen
            
            # val_psnr += psnr_
            # val_mulaw += mu_law

    # val_mulaw /= n_val
    # val_psnr /= n_val
    # print("PSNR:",val_psnr)
    # print("mu_PSNR:",val_mulaw)


def main():
    # settings
    args = get_args()

    if not os.path.exists(args.preddir):
        os.makedirs(args.preddir)
    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # model architectures
    # to modify ....
    model = RDN(args)
	# to modify ....
	
	# to modify ....
    ckptdir = osp.join(args.logdir, 'best_mu.pkl')
    checkpoint = torch.load(ckptdir)  #best_checkpoint.pkl ,best_psnr.pkl
	# to modify ....

    # from collections import OrderedDict
    old_paras = checkpoint#['state_dict']
    # new_paras = OrderedDict()
    # for key, value in old_paras.items():
    #     key = key[7:]
    #     new_paras[key] = value
    model.load_state_dict(old_paras)
    model.to(device)

    # test_epoch = checkpoint['epoch']
    # print('test epoch:', test_epoch)

    # dataset and dataloader
    test_dataset = NTIRE_Testing_Dataset(root_dir=args.dataset_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    prediction(args, model, device, test_loader,ckptdir)


if __name__ == '__main__':
    main()


import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from torch.autograd import Variable


# def conv3x3(in_chn, out_chn, bias=True):
#     layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
#     return layer
#
# def conv_down(in_chn, out_chn, bias=False):
#     layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
#     return layer
#
# def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
#     return nn.Conv2d(
#         in_channels, out_channels, kernel_size,
#         padding=(kernel_size//2), bias=bias, stride=stride)
class make_dilation_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dilation_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2+1, bias=True, dilation=2)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(DRDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dilation_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out

def Dyconv1x1(in_planes, out_planes, stride=1):
    return Dynamic_conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,)

def Dyconv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return Dynamic_conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        self.relu = nn.LeakyReLU(0.2, inplace=False)
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))




    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

        #TODO INIt
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])


    def update_temperature(self):
        self.attention.updata_temperature()


    def forward(self, x):#using batch as the varibale of dimansiona,
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width) #
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output



class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = Dynamic_conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=groups, bias=bias, dilation=dilation)
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                            groups=groups, bias=bias, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        # self.ConvLinear = BasicConv(2*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        # out = x2
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out

class ResidualBlockNoBN(nn.Module):

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Pyramid(nn.Module):
    def __init__(self, in_channels=6, n_feats=64):
        super(Pyramid, self).__init__()
        self.in_channels = in_channels
        self.n_feats = n_feats
        num_feat_extra = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.n_feats, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        layers = []
        for _ in range(num_feat_extra):
            layers.append(ResidualBlockNoBN())
        self.feature_extraction = nn.Sequential(*layers)

        self.downsample1 = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        x_in = self.conv1(x)
        x1 = self.feature_extraction(x_in)
        x2 = self.downsample1(x1)
        x3 = self.downsample2(x2)
        return [x1, x2, x3]


class SBP_layer(nn.Module):
    """
    Structured Bayesian Pruning layer
    #don't forget add kl to loss
    y, kl = sbp_layer(x)
    loss = loss + kl
    """
    def __init__(self, input_dim, init_logsigma2=9.0):
        super(SBP_layer, self).__init__()

        self.log_sigma2 = nn.Parameter(torch.Tensor(input_dim))
        self.mu = nn.Parameter(torch.Tensor(input_dim))

        self.mu.data.normal_(1.0,0.01)
        self.log_sigma2.data.fill_(-init_logsigma2)

    def forward(self, input):

        self.log_alpha = self.log_sigma2 - 2.0 * torch.log(abs(self.mu) + 1e-8)
        self.log_alpha = torch.clamp(self.log_alpha, -10.0, 10.0)


        #mask = (self.log_alpha < 1.0).float()
        self.mask = (self.log_alpha < 0.0).float()


        if self.training:
            si = (self.log_sigma2).mul(0.5).exp_()
            eps = si.data.new(si.size()).normal_()
            multiplicator = self.mu + si*eps
        else:
            multiplicator = self.mu*self.mask

        multiplicator = multiplicator.unsqueeze(1).unsqueeze(2)

        return multiplicator*input

    def kl_reg_input(self):
        kl = 0.5 * torch.log1p(torch.exp(-self.log_alpha))
        a = torch.sum(kl)
        return a
    def sparse_reg_input(self):
        s_ratio = torch.sum(self.mask.view(-1)==0.0).item() / self.mask.view(-1).size(0)
        return s_ratio


        
class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2+1, bias=True, dilation=2)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    modules = []
    for i in range(nDenselayer):    
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate 
    self.dense_layers = nn.Sequential(*modules)    
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out)
    out = out + x
    return out


# Residual Dense Network
class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        nChannel = args.nChannel
        nDenselayer = args.nDenselayer
        nFeat = args.nFeat
        scale = args.scale
        growthRate = args.growthRate
        self.args = args


        # inpaintting module
        self.pyramid_feats = Pyramid(3)

        # F-1
        self.conv1 = nn.Conv2d(nChannel, nFeat, kernel_size=3, padding=1, bias=True)
        self.csl1 = SBP_layer(nFeat)
        # self.csl1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # self.csl1 = nn.Dropout(0.8)
        # F0
        self.conv2 = nn.Conv2d(nFeat*3, nFeat, kernel_size=3, padding=1, bias=True)
        self.csl2 = SBP_layer(nFeat)
        # self.csl2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # self.csl2 = nn.Dropout(0.8)
        self.att11 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att12 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.att31 = nn.Conv2d(nFeat*2, nFeat*2, kernel_size=3, padding=1, bias=True)
        self.att32 = nn.Conv2d(nFeat*2, nFeat, kernel_size=3, padding=1, bias=True)
        self.attConv3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # RDBs 3 
        self.RDB1 = BasicRFB(nFeat, nFeat, stride=1)
        self.RDB2 = BasicRFB(nFeat, nFeat, stride=1)
        self.RDB3 = BasicRFB(nFeat, nFeat, stride=1)

        # self.RDB1 = DRDB(nFeat, 6, 32)
        # self.RDB2 = DRDB(nFeat, 6, 32)
        # self.RDB3 = DRDB(nFeat, 6, 32)

        # self.RDB1 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1)
        # self.RDB2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1)
        # self.RDB3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1)
    
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat*3, nFeat, kernel_size=1, padding=0, bias=True)
        self.csl_1x1 = SBP_layer(nFeat)
        # self.csl_1x1 = nn.Conv2d(nFeat, nFeat, kernel_size=1, padding=0, bias=True)
        # self.csl_1x1 = nn.Dropout(0.8)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.csl_3x3 = SBP_layer(nFeat)
        # self.csl_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # self.csl_3x3 = nn.Dropout(0.8)
        # Upsampler
        self.conv_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        self.csl_up = SBP_layer(nFeat)
        # self.csl_up = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # self.csl_up = nn.Dropout(0.8)
        # conv 
        self.conv3 = nn.Conv2d(nFeat, 3, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()
    def forward(self, x1, x2, x3):

        F1_ = self.relu(self.csl1(self.conv1(x1)))
        F2_ = self.relu(self.csl1(self.conv1(x2)))
        F3_ = self.relu(self.csl1(self.conv1(x3)))

        F1_i = torch.cat((F1_, F2_), 1)
        F1_A = self.relu(self.att11(F1_i))
        F1_A = self.att12(F1_A)
        F1_A = nn.functional.sigmoid(F1_A)
        F1_ = F1_ * F1_A
        # F1_ = self.attConv1(F1_)
        F3_i = torch.cat((F3_, F2_), 1)
        F3_A = self.relu(self.att31(F3_i))
        F3_A = self.att32(F3_A)
        F3_A = nn.functional.sigmoid(F3_A)
        F3_ = F3_ * F3_A
        #F3_ = self.attConv1(F3_)
        F_ = torch.cat((F1_, F2_, F3_), 1)


        F_0 = self.conv2(F_)
        F_0 = self.csl2(F_0)

        # pyramid features of linear domain
        # F_0 = self.pyramid_feats(F_0)
        # PCD alignment
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)

        # FF = torch.cat((F_1, F_2, F_3), 1)
        # FdLF = self.GFF_1x1(FF)
        FdLF = self.csl_1x1(F_3)
        FGF = self.GFF_3x3(FdLF)
        FGF = self.csl_3x3(FGF)
        FDF = FGF + F2_
        us = self.conv_up(FDF)
        us = self.csl_up(us)

        output = self.conv3(us)
        output = nn.functional.sigmoid(output)


        return output

    def layerwise_sparsity(self):
        return [self.csl_up.sparse_reg_input()]




class ELBO(nn.Module):

    def __init__(self, net, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size
        self.net = net

    def forward(self, input, target, kl_weight=1.0):
        assert not target.requires_grad
        kl = 0.0
        for module in self.net.children():
            if hasattr(module, 'kl_reg'):
               kl = kl + module.kl_reg()
            if hasattr(module, 'kl_reg_input'):
               kl = kl + module.kl_reg_input()


        # sparsity_arr = self.net.layerwise_sparsity()

        output = torch.log(1 + 5000 * input.cpu()) / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))
        target = torch.log(1 + 5000 * target).cpu() / torch.log(Variable(torch.from_numpy(np.array([1+5000])).float()))

       # print('l1-Sparsity: %.4f, l2-Sparsity: %.4f, l3-Sparsity: %.4f' % (sparsity_arr[0], sparsity_arr[1], sparsity_arr[2]))
        loss = kl_weight * kl + F.l1_loss(output, target)
        return loss, kl, F.l1_loss(output, target)#, sparsity_arr 
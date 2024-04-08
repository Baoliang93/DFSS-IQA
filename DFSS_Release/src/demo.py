# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:31:05 2023

@author: h'p
"""
import torch.utils.data
from skimage import io
from os.path import join
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from demomodel import IQANet
from dataset import Transforms
import ipdb
from thop import profile

demo_tfs = Transforms()

data_dir = '../demoimgs/'
img1_name = 'SCI01_1_1.bmp'
img2_name = 'SCI01_1_5.bmp'
resume = '../models/scid-all/checkpoint_latest_0.pkl'
# resume = '../models/siqad-all/checkpoint_latest_0.pkl'
patch_size = 32
img1 = io.imread(join(data_dir, img1_name))
img2 = io.imread(join(data_dir, img2_name))
cudnn.benchmark = True
def to_patch_tensors_single(img,patch_size):
    img_ptchs = demo_tfs.to_patches(img,ptch_size=patch_size, n_ptchs=1024)
    img_ptchs = demo_tfs.to_tensor(img_ptchs)
    return img_ptchs

img1 = to_patch_tensors_single(img1, patch_size).unsqueeze(0).cuda()
img2 = to_patch_tensors_single(img2, patch_size).unsqueeze(0).cuda()

# print(img1.shape, img2.shape)
model = IQANet(istrain=False, n_class=46).cuda()
model = nn.DataParallel(model)
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint['state_dict'])
model = model.eval().cuda()
score1 = model(img1)
score2 = model(img2)
print(score1, score2)


    


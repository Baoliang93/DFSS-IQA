#!/usr/bin/env python3
"""
Main Script
"""

import sys
import os

import shutil
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch import nn

from model import IQANet
from dataset import TID2013Dataset, IQADataset
from utils import AverageMeter, SROCC, PLCC, RMSE
from utils import SimpleProgressBar as ProgressBar


def test(test_data_loader, model):
    srocc = SROCC()
    plcc = PLCC()
    rmse = RMSE()
    len_test = len(test_data_loader)
    pb = ProgressBar(len_test, show_step=True)

    print("Testing")

    model.eval()
    with torch.no_grad():
        for i, (img, ref, pref, score, dtype) in enumerate(test_data_loader):
            img = img.cuda()
            ref = ref.cuda()
            pref = pref.cuda()
            output = model(img, img, img).squeeze()
           
            
            output = output.cpu().data.numpy()
            score = score.data.numpy()

            srocc.update(score, output)
            plcc.update(score, output)
            rmse.update(score, output)

            pb.show(i, "Test: [{0:5d}/{1:5d}]\t"
                    "Score: {2:.4f}\t"
                    "Label: {3:.4f}"
                    .format(i+1, len_test, float(output), float(score)))

    print("\n\nSROCC: {0:.4f}\n"
            "PLCC: {1:.4f}\n"
            "RMSE: {2:.4f}"
            .format(srocc.compute(), plcc.compute(), rmse.compute())
    )



def test_iqa(args):
    batch_size = 1
    pro = args.pro
    num_workers = args.workers
    subset = args.subset
    data_dir = args.data_dir
    list_dir = args.list_dir
    resume = args.resume
    patch_size = args.patch_size
    n_ptchs = args.n_ptchs_per_img
    dtypes =  args.n_dtype


    for k, v in args.__dict__.items():
        print(k, ':', v)

    model = IQANet(istrain=False, n_class=dtypes)
    model = nn.DataParallel(model)
    test_loader = torch.utils.data.DataLoader(
        Dataset(data_dir, 'test_'+str(pro), list_dir=list_dir,  patch_size=patch_size,
        n_ptchs=n_ptchs),
        batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True
    )


    cudnn.benchmark = True

    # Resume from a checkpoint
    if resume:
        resume = resume.split('t.')[0]+'t_'+str(pro)+'.pkl'
        # resume ='../models/checkpoint_latest_0.pkl'
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    test(test_loader, model.cuda())


# def parse_args():
#     # Training settings
#     parser = argparse.ArgumentParser(description='')
#     parser.add_argument('-cmd', type=str,default='test')
#     parser.add_argument('-d', '--data-dir', default='../../../datasets/SCID/')
#     parser.add_argument('-l', '--list-dir', default='../sci_scripts/siqad-scripts-all/',
#                         help='List dir to look for train_images.txt etc. '
#                               'It is the same with --data-dir if not set.')
#     parser.add_argument('-n', '--n-ptchs-per-img', type=int, default=1024, metavar='N', 
#                         help='number of patches for each image (default: 32)')
#     parser.add_argument('-nd', '--n-dtype', type=int, default=50, metavar='N', 
#                         help='number of distortion types (siqad:7，scid:10)')
#     parser.add_argument('-psz', '--patch-size', type=int, default=32, metavar='N', 
#                         help='size of cropped patches for each image (default: 64)')
#     parser.add_argument('--step', type=int, default=200)
#     parser.add_argument('--batch-size', type=int, default=32, metavar='B',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--epochs', type=int, default=1000, metavar='NE',
#                         help='number of epochs to train (default: 1000)')
#     parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
#                         help='learning rate (default: 1e-4)')
#     parser.add_argument('--lr-mode', type=str, default='const')
#     parser.add_argument('--weight-decay', default=1e-4, type=float,
#                         metavar='W', help='weight decay (default: 1e-4)')
#     parser.add_argument('--resume', default='../models/siqad-all/checkpoint_latest.pkl',type=str, metavar='PATH',
#             help='path to latest checkpoint')
#     parser.add_argument('--workers', type=int, default=8)
#     parser.add_argument('--pro', type=int, default=0)
#     parser.add_argument('--subset', default='test')
#     parser.add_argument('--evaluate', dest='evaluate',
#                         action='store_true',
#                         help='evaluate model on validation set')
#     parser.add_argument('--weighted',default=True, dest='weighted', 
#             action='store_true')
#     parser.add_argument('--dump_per', type=int, default=50, 
#                         help='the number of epochs to make a checkpoint')
#     parser.add_argument('--dataset', type=str, default='IQA')
#     parser.add_argument('--anew', action='store_true')

#     args = parser.parse_args()

#     return args

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-cmd', type=str,default='test')
    parser.add_argument('-d', '--data-dir', default='../../../datasets/SIQAD/')
    parser.add_argument('-l', '--list-dir', default='../sci_scripts/scid-scripts-all/',
                        help='List dir to look for train_images.txt etc. '
                              'It is the same with --data-dir if not set.')
    parser.add_argument('-n', '--n-ptchs-per-img', type=int, default=1024, metavar='N', 
                        help='number of patches for each image (default: 32)')
    parser.add_argument('-nd', '--n-dtype', type=int, default=46, metavar='N', 
                        help='number of distortion types (siqad:7，scid:10)')
    parser.add_argument('-psz', '--patch-size', type=int, default=32, metavar='N', 
                        help='size of cropped patches for each image (default: 64)')
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='NE',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-mode', type=str, default='const')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='../models/scid-all/checkpoint_latest.pkl',type=str, metavar='PATH',
            help='path to latest checkpoint')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--pro', type=int, default=0)
    parser.add_argument('--subset', default='test')
    parser.add_argument('--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--weighted',default=True, dest='weighted', 
            action='store_true')
    parser.add_argument('--dump_per', type=int, default=50, 
                        help='the number of epochs to make a checkpoint')
    parser.add_argument('--dataset', type=str, default='IQA')
    parser.add_argument('--anew', action='store_true')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    # Choose dataset
    global Dataset
    Dataset = globals().get(args.dataset+'Dataset', None)
    test_iqa(args)


if __name__ == '__main__':
    main()

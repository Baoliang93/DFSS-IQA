"""
Dataset and Transforms
"""


import torch.utils.data
import numpy as np
import random
import json
from skimage import io
from os.path import join, exists
from utils import limited_instances, SimpleProgressBar
import matplotlib.pyplot as plt  
import torchvision
import argparse
from torchvision import transforms

class IQADataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, patch_size=64, n_ptchs=256, n_class = 46, sample_once=False, subset='', list_dir=''):
        super(IQADataset, self).__init__()

        self.list_dir = data_dir if not list_dir else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.subset = phase if not subset.strip() else subset
        self.n_ptchs = n_ptchs
        self.img_list = []
        self.ref_list = []
        self.score_list = []
        self.sample_once = sample_once
        self._from_pool = False
        self.patch_size = patch_size
        if n_class == 46:
           self.n_levels =5
        elif n_class == 50:
           self.n_levels =7
        else:
            print('Wrong number of dist clsses!')
           

        self._read_lists()
        self._aug_lists()
        # print(self.score_list)
        self.normal = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))

        self.tfs = Transforms()
        if sample_once:
            @limited_instances(self.__len__())
            class IncrementCache:
                def store(self, data):
                    self.data = data

            self._pool = IncrementCache
            self._to_pool()
            self._from_pool = True

    def __getitem__(self, index):
        img = self._loader(self.img_list[index])
        ref = self._loader(self.ref_list[index])
        # print(self.img_list[index])
  
        score = self.score_list[index]
        score = torch.tensor(score).float()
#        print(self.img_list[index])
        res_split = self.img_list[index].split('_')
        if len(res_split)>1:
           dist_type = torch.tensor((int(res_split[1])-1)*self.n_levels+int(res_split[2].split('.')[0])).long()
        else:
           dist_type = torch.tensor(0).long()
        
        lenth = len(self.ref_list)
        pref_id = np.random.randint(0,lenth)
#        print(lenth,pref_id,index)
        while self.ref_list[pref_id]==self.ref_list[index]:
            pref_id = np.random.randint(0,lenth)
        pref = self._loader(self.ref_list[pref_id])


        if self._from_pool:
            (img_ptchs, ref_ptchs,pref_ptchs) = self._pool(index).data
        else:
            if self.phase.split('_')[0] == 'train':
                img, ref = self.tfs.horizontal_flip(img, ref)
                img_ptchs, ref_ptchs = self._to_patch_tensors(img, ref,self.patch_size)
                pref = self.tfs.horizontal_flip(pref)
                pref_ptchs = self._to_patch_tensors_single(pref,self.patch_size)               
            elif self.phase.split('_')[0] == 'val':
                img_ptchs, ref_ptchs = self._to_patch_tensors(img, ref,self.patch_size)
                pref_ptchs = self._to_patch_tensors_single(pref,self.patch_size)
            elif self.phase.split('_')[0] == 'test':
                img_ptchs, ref_ptchs = self._to_patch_tensors(img, ref,self.patch_size)
                pref_ptchs = self._to_patch_tensors_single(pref,self.patch_size)
            else:
                pass

        return img_ptchs, ref_ptchs, pref_ptchs,score, dist_type

    def __len__(self):
        return len(self.img_list)

    def _loader(self, name):
        return io.imread(join(self.data_dir, name))

    def _to_patch_tensors(self, img, ref,patch_size):
            img_ptchs, ref_ptchs = self.tfs.to_patches(img, ref, ptch_size=patch_size, n_ptchs=self.n_ptchs)
            img_ptchs, ref_ptchs = self.tfs.to_tensor(img_ptchs, ref_ptchs)
            return img_ptchs, ref_ptchs

    def _to_patch_tensors_single(self, img,patch_size):
            img_ptchs = self.tfs.to_patches(img,ptch_size=patch_size, n_ptchs=self.n_ptchs)
            img_ptchs = self.tfs.to_tensor(img_ptchs)
            return img_ptchs
        
    def _to_pool(self):
        len_data = self.__len__()
        pb = SimpleProgressBar(len_data)
        print("\ninitializing data pool...")
        for index in range(len_data):
            self._pool(index).store(self.__getitem__(index)[0])
            pb.show(index, "[{:d}]/[{:d}] ".format(index+1, len_data))

    def _aug_lists(self):
        if self.phase.split('_')[0] == 'test':
            return
        # Make samples from the reference images
        # The number of the reference samples appears 
        # CRITICAL for the training effect!
        len_aug = len(self.ref_list)//5 if self.phase.split('_')[0] == 'train' else 10
        aug_list = self.ref_list*(len_aug//len(self.ref_list)+1)
        random.shuffle(aug_list)
        aug_list = aug_list[:len_aug]
        self.img_list.extend(aug_list)
        self.score_list += [0.0]*len_aug
        self.ref_list.extend(aug_list)

        if self.phase.split('_')[0] == 'train':
            # More samples in one epoch
            # This accelerates the training indeed as the cache
            # of the file system could then be fully leveraged
            # And also, augment the data in terms of number
            mul_aug = 16
            self.img_list *= mul_aug
            self.ref_list *= mul_aug
            self.score_list *= mul_aug

    def _read_lists(self):
        img_path = join(self.list_dir, self.phase + '_data.json')
        print(img_path)
        assert exists(img_path)

        with open(img_path, 'r') as fp:
            data_dict = json.load(fp)

        self.img_list = data_dict['img']
        self.ref_list = data_dict.get('ref', self.img_list)
        self.score_list = data_dict.get('score', [0.0]*len(self.img_list))


class TID2013Dataset(IQADataset):
    def _read_lists(self):
        super()._read_lists()
        # For TID2013
        self.score_list = [(9.0 - s) / 9.0 * 100.0 for s in self.score_list]

class SCIDDataset(IQADataset):
    def _aug_lists(self):
        if self.phase.split('_')[0] == 'test':
            return
        # Make samples from the reference images
        # The number of the reference samples appears 
        # CRITICAL for the training effect!
        len_aug = len(self.ref_list)//5 if self.phase.split('_')[0] == 'train' else 10
        aug_list = self.ref_list*(len_aug//len(self.ref_list)+1)
        random.shuffle(aug_list)
        aug_list = aug_list[:len_aug]
        self.img_list.extend(aug_list)
        self.score_list += [90.0]*len_aug
        self.ref_list.extend(aug_list)

        if self.phase.split('_')[0] == 'train':
            # More samples in one epoch
            # This accelerates the training indeed as the cache
            # of the file system could then be fully leveraged
            # And also, augment the data in terms of number
            mul_aug = 16
            self.img_list *= mul_aug
            self.ref_list *= mul_aug
            self.score_list *= mul_aug


class WaterlooDataset(IQADataset):
    def _read_lists(self):
        super()._read_lists()
        self.score_list = [(1.0 - s) * 100.0 for s in self.score_list]


    
class Transforms:
    """
    Self-designed transformation class
    ------------------------------------
    
    Several things to fix and improve:
    1. Strong coupling with Dataset cuz transformation types can't 
        be simply assigned in training or testing code. (e.g. given
        a list of transforms as parameters to construct Dataset Obj)
    2. Might be unsafe in multi-thread cases
    3. Too complex decorators, not pythonic
    4. The number of params of the wrapper and the inner function should
        be the same to avoid confusion
    5. The use of params and isinstance() is not so elegant. For this, 
        consider to stipulate a fix number and type of returned values for
        inner tf functions and do all the forwarding and passing work inside
        the decorator. tf_func applies transformation, which is all it does. 
    6. Performance has not been optimized at all
    7. Doc it
    8. Supports only numpy arrays
    """
    def __init__(self):
        super(Transforms, self).__init__()

    def _pair_deco(tf_func):
        def transform(self, img, ref=None, *args, **kwargs):
            """ image shape (w, h, c) """
            if (ref is not None) and (not isinstance(ref, np.ndarray)):
                args = (ref,)+args
                ref = None
            ret = tf_func(self, img, None, *args, **kwargs)
            assert(len(ret) == 2)
            if ref is None:
                return ret[0]
            else:
                num_var = tf_func.__code__.co_argcount-3    # self, img, ref not counted
                if (len(args)+len(kwargs)) == (num_var-1): 
                    # The last parameter is special
                    # Resend it if necessary
                    var_name = tf_func.__code__.co_varnames[-1]
                    kwargs[var_name] = ret[1]
                tf_ref, _ = tf_func(self, ref, None, *args, **kwargs)
                return ret[0], tf_ref
        return transform

    def _horizontal_flip(self, img, flip):
        if flip is None:
            flip = (random.random() > 0.5)
        return (img[...,::-1,:] if flip else img), flip

    def _to_tensor(self, img):
        return torch.from_numpy((img.astype(np.float32)/255).swapaxes(-3,-2).swapaxes(-3,-1)), ()

    def _crop_square(self, img, crop_size, pos):
        if pos is None:
            h, w = img.shape[-3:-1]
            assert(crop_size <= h and crop_size <= w)
            ub = random.randint(0, h-crop_size)
            lb = random.randint(0, w-crop_size)
            pos = (ub, ub+crop_size, lb, lb+crop_size)
        return img[...,pos[0]:pos[1],pos[-2]:pos[-1],:], pos

    def _extract_patches(self, img, ptch_size):
        # Crop non-overlapping patches as the stride equals patch size
        h, w = img.shape[-3:-1]
        nh, nw = h//ptch_size, w//ptch_size
        assert(nh>0 and nw>0)
        vptchs = np.stack(np.split(img[...,:nh*ptch_size,:,:], nh, axis=-3))
        ptchs = np.concatenate(np.split(vptchs[...,:nw*ptch_size,:], nw, axis=-2))
        return ptchs, nh*nw

    def _to_patches(self, img, ptch_size, n_ptchs, idx):
        ptchs, n = self._extract_patches(img, ptch_size)
        if not n_ptchs:
            n_ptchs = n
        elif n_ptchs > n:
            n_ptchs = n  
        if idx is None:
            idx = list(range(n))
            random.shuffle(idx)
            idx = idx[:n_ptchs]
        return ptchs[idx], idx

    @_pair_deco
    def horizontal_flip(self, img, ref=None, flip=None):
        return self._horizontal_flip(img, flip=flip)

    @_pair_deco
    def to_tensor(self, img, ref=None):
        return self._to_tensor(img)

    @_pair_deco
    def crop_square(self, img, ref=None, crop_size=64, pos=None):
        return self._crop_square(img, crop_size=crop_size, pos=pos)

    @_pair_deco
    def to_patches(self, img, ref=None, ptch_size=64, n_ptchs=None, idx=None):
        return self._to_patches(img, ptch_size=ptch_size, n_ptchs=n_ptchs, idx=idx)




def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()  
    

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-cmd', type=str,default='train')
    parser.add_argument('-d', '--data-dir', default='../../../datasets/tid2013/')
    parser.add_argument('-l', '--list-dir', default='../sci_scripts/scid-scripts-6-2-2/',
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('-n', '--n-ptchs-per-img', type=int, default=8, metavar='N', 
                        help='number of patches for each image (default: 32)')
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32, metavar='B',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-psz', '--patch-size', type=int, default=32, metavar='N', 
                        help='size of cropped patches for each image (default: 64)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='NE',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-mode', type=str, default='const')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--resume', default='../models/checkpoint_latest.pkl', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--pro', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--subset', default='test')
    parser.add_argument('--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--weighted',default='False', dest='weighted', 
                        action='store_true')
    parser.add_argument('--dump_per', type=int, default=50, 
                        help='the number of epochs to make a checkpoint')
    parser.add_argument('--dataset', type=str, default='SCID')
    parser.add_argument('--anew', action='store_true')

    args = parser.parse_args()

    return args


if __name__=='__main__':
    def worker_init_fn_seed(worker_id):
        seed = 1
        seed += worker_id
        np.random.seed(seed)
        
    args = parse_args()
    Dataset = globals().get(args.dataset+'Dataset', None)
    pro = args.pro
    batch_size = args.batch_size
    num_workers = args.workers
    data_dir = args.data_dir
    list_dir = args.list_dir
    resume = args.resume
    n_ptchs = args.n_ptchs_per_img
    patch_size = args.patch_size
      
    train_loader = torch.utils.data.DataLoader(
        Dataset(data_dir, 'train_'+str(pro), list_dir=list_dir, patch_size=patch_size,\
                n_ptchs=n_ptchs),
        batch_size=1, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        worker_init_fn= worker_init_fn_seed
    )
    dataiter = iter(train_loader)
    for i in range(3):
       
      example_batch = next(dataiter)
      concatenated = torch.cat((example_batch[0][:,0,:,:,:],example_batch[1][:,0,:,:,:],\
                                example_batch[2][:,0,:,:,:]),0)
      imshow(torchvision.utils.make_grid(concatenated,8))
      print(example_batch[3].numpy(), example_batch[4].numpy())  
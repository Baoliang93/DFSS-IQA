import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import ipdb
eps=1e-12


def SPSP(x):
    pool_features = []
    M = F.adaptive_avg_pool2d(x, (3,3))
    pool_features.append(M)  # max pooling
   
    rm2 = torch.sqrt(F.relu(F.adaptive_avg_pool2d(torch.pow(x, 2), (3,3)) - torch.pow(M, 2)))  
    pool_features.append(rm2)
    return torch.cat(pool_features, dim=1)


class Conv3x3(nn.Module):
    def __init__(self, in_dim, out_dim, isbn=True):
        super(Conv3x3, self).__init__()
        if isbn==True:
            self.conv = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=(1,1), padding=(1,1), bias=True),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=(1,1), padding=(1,1), bias=True),
                nn.LeakyReLU(0.2)
            )
    def forward(self, x):
        return self.conv(x)

class Conv5x5(nn.Module):
    def __init__(self, in_dim, out_dim, isbn=True):
        super(Conv5x5, self).__init__()
        if isbn==True:
            self.conv = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=(1,1), padding=(2,2), bias=True),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=(1,1), padding=(2,2), bias=True),
                nn.LeakyReLU(0.2)
            )
    def forward(self, x):
        return self.conv(x)

class MaxPool2x2(nn.Module):
    def __init__(self):
        super(MaxPool2x2, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=(2,2), padding=(0,0))
   
    def forward(self, x):
        return self.pool(x)

class DoubleConv(nn.Module):
    def __init__(self, in_dim, out_dim,ispool = True):
        super(DoubleConv, self).__init__()
        self.conv1 = Conv3x3(in_dim, out_dim)
        self.conv2 = Conv3x3(out_dim, out_dim)
        self.pool = MaxPool2x2()
        self.ispool = ispool

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.ispool:
            y = self.pool(y)
        return y

class InterDoubleConv(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim,ispool = True):
        super(InterDoubleConv, self).__init__()
        self.conv1 = Conv3x3(in_dim, inter_dim)
        self.conv2 = Conv3x3(inter_dim, out_dim)
        self.pool = MaxPool2x2()
        self.ispool = ispool
        self.classfy = nn.Softmax(dim=1)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.ispool:
            y = self.pool(y)
        y = self.classfy(y)
        return y


class SingleConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SingleConv, self).__init__()
        self.conv = Conv3x3(in_dim, out_dim)
        self.pool = MaxPool2x2()

    def forward(self, x):
        y = self.conv(x)
        y = self.pool(y)
        return y

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
        nn.Linear(channel, channel // reduction),
        nn.LeakyReLU(),
        nn.Linear(channel // reduction, channel),
        nn.Sigmoid()
        )
    
    def forward(self, x,y):
        y = self.fc(y)
        return x * y


class IQANet(nn.Module):
    def __init__(self,istrain=False,scale = 2,n_class=10,channel_input=256):
        super(IQANet, self).__init__()

        self.istrain = istrain
        self.scale = scale
    
        
        self.sf0 =  InterDoubleConv(3, 16, 2, ispool = False)
        self.sf101 =  Conv3x3(3, 64)
        self.sf102 =  Conv5x5(3, 64)
        
        self.sfl1 = nn.Conv2d(128, 256, kernel_size=3, padding=(1,1)) 
     
      

        self.sfl21 = DoubleConv(256, 64*self.scale)
        self.sfl22 = DoubleConv(64*self.scale, 64*self.scale)
        self.sfl23 = DoubleConv(64*self.scale, 64*self.scale)
        self.sfl3 =  DoubleConv(64*self.scale, 128*self.scale)
        self.scl = nn.Sequential(
                   nn.Conv2d(1792, 512, kernel_size=3, bias=True),
                   nn.LeakyReLU(0.2),
                   nn.Conv2d(512, 512, kernel_size=1,bias=True)
                  )

        self.distype_cls = nn.Sequential(
                   nn.Linear(256, 128),
                   nn.LeakyReLU(0.2),
                   nn.Linear(128, n_class)
                  )        
        self.scw = nn.Linear(256, 1)
               

        self.selayer = SELayer(channel=256)
        
        self._initialize_weights()
 
    def _get_initial_state(self, batch_size):
            h0 = torch.zeros(1, batch_size, 32,device=0)
            return h0  
       
    def NR_extract_feature(self, x):
        sifet = []
       
        
        y101 = self.sf101(x)
        y102 = self.sf102(x)
        y = torch.cat((y101,y102),1)
        
        y= self.sfl1(y)
      
        y = F.max_pool2d(F.leaky_relu(y),kernel_size=2, stride=(2,2), padding=(0,0))
        
        sifet.append(SPSP(y))
        y = self.sfl21(y)
        sifet.append(SPSP(y))
        y = self.sfl22(y)
        sifet.append(SPSP(y))
        y = self.sfl23(y)
        sifet.append(SPSP(y))
        y = self.sfl3(y)
        sifet.append(SPSP(y))
        sifet = torch.cat(sifet, dim=1)
        sifet = self.scl(sifet).squeeze(-1).squeeze(-1)            
        return sifet


    def forward(self, x1, x2, x3):
        """ x1 as distorted, x2 as reference, x3 as preference  """
        n_imgs, n_ptchs_per_img = x1.shape[0:2]             
        # Reshape
        x1 = x1.view(-1,*x1.shape[-3:])
        x2 = x2.view(-1,*x2.shape[-3:]) 
        x3 = x3.view(-1,*x3.shape[-3:]) 
        x = torch.cat((x1,x2, x3),0)
       
        sf = self.NR_extract_feature(x)        
        ssf,sif = torch.split(sf, int(sf.shape[1]/2), dim=1)
        


        sif = sif.view(n_imgs*3,n_ptchs_per_img,-1)
        img_si,ref_si,pref_si = torch.split(sif, int(sif.shape[0]/3), dim=0)

        sif_mean = sif.mean(1,keepdim=True)
        sif_logv = sif - sif_mean
        sif_logv = torch.mean(torch.pow(sif_logv, 2.0),dim=1,keepdim=False)
        sif_logv = torch.log(sif_logv+eps)
        sif_mean = sif_mean.squeeze(1)
        
        diff_sif = -0.5*(1 + sif_logv - sif_mean ** 2 - sif_logv.exp())
        
        diff_img_si, diff_ref_si,diff_pref_si = torch.split(diff_sif, int(diff_sif.shape[0]/3), dim=0)

        
        img_si_mean = img_si.mean(1,keepdim=False)
        img_si_cls =  self.distype_cls(img_si_mean)
                
        ssf_mean = ssf.view(n_imgs*3,n_ptchs_per_img,-1).mean(1,keepdim=False)
        img_ss_mean, ref_ss_mean, pref_ss_mean = torch.split(ssf_mean, int(ssf_mean.shape[0]/3), dim=0)
        # print(diff_img_si.shape,img_ss_mean.shape )    
        reg_feat = self.selayer(diff_img_si,img_ss_mean)
        pref_mos = self.scw(reg_feat)
        # ipdb.set_trace()
        if self.istrain==1:
           return  pref_mos, img_ss_mean, ref_ss_mean, pref_ss_mean, ref_si,img_si,diff_ref_si,diff_pref_si,img_si_cls                         
        elif self.istrain==0:
           return pref_mos.squeeze()
        else:
           return  pref_mos, img_ss_mean, ref_ss_mean, diff_img_si, diff_ref_si,diff_pref_si,img_si_cls                         

        
       
       

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            else:
                pass
           
                       
def test():

    net = IQANet(istrain=True)
    net.cuda()
 
    x1 = torch.randn(2, 4,3,32,32)
    x1 = Variable(x1.cuda())
   
    y1,y2,y3,y4,y5,y6,y7, y8,y9,y10= net.forward(x1,x1,x1)


    print(y1.shape,y2.shape,y3.shape,y4.shape,y5.shape,y6.shape,y7.shape, y8.shape, y9.shape,y10.shape)

   
if __name__== '__main__':
    test()                
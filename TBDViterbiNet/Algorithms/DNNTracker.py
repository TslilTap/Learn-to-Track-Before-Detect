import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.dnn_tracker_utils as tracker_utils

class BBoxTracker(nn.Module):
    def __init__(self, n=32, Nr=200, Nv=64):
        super(BBoxTracker,self).__init__()

        self.Nr = Nr
        self.Nv = Nv

        self.MBC1 = tracker_utils.MovingBiasConv(in_channels=1,out_channels=n,Nr=self.Nr,Nv=self.Nv)
        self.MBC2 = tracker_utils.MovingBiasConv(in_channels=n,out_channels=n*2,Nr=self.Nr,Nv=self.Nv)
        self.MBC3 = tracker_utils.MovingBiasConv(in_channels=n*2,out_channels=n*4,Nr=self.Nr,Nv=self.Nv)
        self.MBC4 = tracker_utils.MovingBiasConv(in_channels=n*4,out_channels=n*4,Nr=self.Nr,Nv=self.Nv)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=4 * n, out_channels=2 * n, kernel_size=3, padding=1),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=2 * n, out_channels=2 * n, kernel_size=3, padding=1),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=2 * n, out_channels= n, kernel_size=3, padding=1),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=n,out_channels=1,kernel_size=1,padding=0)
        )

    def forward(self,x,bbox=None,restore = True):
        '''
        :param x: input of size [c,Nr,Nv]
        :param bbox:
        :param restore:
        :return:
        '''



        if x.shape[1] != self.Nr or x.shape[2] != self.Nv:
            raise ValueError("Input size")
        if bbox is not None:
            x = tracker_utils.crop(x,bbox)
        shape_crop = x.shape
        x = self.MBC1(x,bbox)
        x = self.MBC2(x,bbox)
        x = self.MBC3(x,bbox)
        x = self.MBC4(x,bbox)
        x = self.block(x)
        x = x.view(-1)
        x = F.log_softmax(x,dim=1)
        x = x.view(shape_crop)
        if restore:
            val_min = torch.min(x).item()
            y = torch.ones([self.Nr, self.Nv]) * val_min
            y[bbox[0]:bbox[2], bbox[1]:bbox[3]] = x
            return y
        return x
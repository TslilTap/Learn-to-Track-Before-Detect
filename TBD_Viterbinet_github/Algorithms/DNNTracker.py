import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.dnn_tracker_utils as tracker_utils
from Configuration.tracker_config import tracker_param

class DNNTracker(nn.Module):
    def __init__(self, tracker_param):
        super(DNNTracker,self).__init__()
        n = tracker_param["num_channels"]
        drop = tracker_param["drop_rate"]
        self.env = tracker_param["environment"]
        self.Nr = self.env.num_range_bins
        self.Nd = self.env.num_doppler_bins

        self.MBC1 = tracker_utils.MovingBiasConv(
            in_channels=1,
            out_channels=n,
            height=self.Nr,
            width=self.Nd,
            drop=drop
        )
        self.MBC2 = tracker_utils.MovingBiasConv(
            in_channels=n,
            out_channels=n*2,
            height=self.Nr,
            width=self.Nd,
            drop=drop
        )
        self.MBC3 = tracker_utils.MovingBiasConv(
            in_channels=n*2,
            out_channels=n*4,
            height=self.Nr,
            width=self.Nd,
            drop=drop
        )
        self.MBC4 = tracker_utils.MovingBiasConv(
            in_channels=n*4,
            out_channels=n*4,
            height=self.Nr,
            width=self.Nd,
            drop=drop
        )
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=4 * n,
                out_channels=2 * n,
                kernel_size=3,
                padding=1
            ),
            nn.Dropout(drop),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=2 * n
                , out_channels=2 * n
                , kernel_size=3
                , padding=1
            ),
            nn.Dropout(drop),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=2 * n
                , out_channels=n
                , kernel_size=3
                , padding=1
            ),
            nn.Dropout(drop),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=n,
                out_channels=1,
                kernel_size=1,
                padding=0,
            )
        )

    def forward(self,x,bbox=None,restore = True):
        '''
        :param x: input of size [c,Nr,Nv]
        :param bbox:
        :param restore:
        :return:
        '''



        if x.shape[1] != self.Nr or x.shape[2] != self.Nd:
            raise ValueError(f"Input size {x.shape} does not match {[self.Nr,self.Nd]}")
        if bbox is not None:
            x = tracker_utils.crop(x,bbox)
        shape_crop = x.shape
        x = self.MBC1(x,bbox)
        x = self.MBC2(x,bbox)
        x = self.MBC3(x,bbox)
        x = self.MBC4(x,bbox)
        x = self.block(x)
        x = x.view(-1)
        x = F.log_softmax(x,dim=0)
        x = x.view(shape_crop)
        if restore and bbox is not None:
            val_min = torch.min(x).item()
            y = self.env.ones() * val_min
            y[bbox[0]:bbox[2], bbox[1]:bbox[3]] = x
            return y
        return x

dnn_tracker = DNNTracker(tracker_param=tracker_param)
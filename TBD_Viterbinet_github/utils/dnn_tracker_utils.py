import torch
import torch.nn as nn


class MovingBiasConv(nn.Module):
    '''
    This class is meant for implementing a convolution layer on a cropped part of an image
    In this scenario the size of the image is constant [Nr,Nv]
    the input is a cropped part of the image, size [nr,nv]
    along with the bbox that was used to create it [r0,v0,r0+nr,v0+nv]

    The 'bias' parameter is of size [in_channels,Nr,Nv], when an input is given the bias is then
    cropped using the given bbox, the cropped bias is then subtracted from the cropped image.
    We refer to this stage as semi-normalization, as it severs a similar function to that of a batch normalization layer.

    Do note that the size of the input needs to match the parameters of the bbox.
    Also, if no bbox is given then the model assumes that no cropping took place and that the input was the entire image

    After the semi-normalization, the new tensor is forwarded through a convolutional layer, followed by a droupout and a leakyrelu
    '''


    def __init__(self,in_channels,out_channels,height,width,drop=0.25):
        '''
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param Nr: number of range bins
        :param Nv: number of velocity bins
        :param drop: drop rate. optional. default: 0.25
        '''
        super(MovingBiasConv,self).__init__()
        self.bias = nn.Parameter(torch.randn(in_channels,height,width))
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=True)
        self.drop = nn.Dropout(drop)
        self.LeakyRelu = nn.LeakyReLU()

    def forward(self,x,bbox=None):
        '''
        :param input: cropped image of size [1,in_channels,nr,nv]
        :param bbox: bounding box. written as '[r0,v0,r0+nr,v0+nr]'
        :return: image of size [1,in_channels,nr,nv]
        '''
        c = x.shape[0]
        if c != self.bias.shape[0]:
            raise ValueError(f"input of shape {x.shape} does not match with bias tensor of shape {self.bias.shape}")

        nr = x.shape[1]
        nv = x.shape[2]

        if bbox is None:
            if nr != self.bias.shape[1] or nv != self.bias.shape[2]:
                raise ValueError(f"input of shape {x.shape} does not match with bias tensor of shape {self.bias.shape}")
            bias = self.bias
        else:
            nr_b = bbox[2] - bbox[0]
            nv_b = bbox[3] - bbox[1]
            if nr != nr_b or nv != nv_b:
                raise ValueError(f"input of shape {x.shape} does not match with the bounding box of shape {[nr_b,nv_b]}")
            bias = crop(self.bias,bbox)
        x = x - bias
        x = self.conv(x)
        x = self.drop(x)
        x = self.LeakyRelu(x)
        return x

def crop(x,bbox):
    if len(x.shape) == 2:
        return x[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    elif len(x.shape) == 3:
        return x[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
    elif len(x.shape) == 4:
        return x[:, :, bbox[0]:bbox[2], bbox[1]:bbox[3]]
    else:
        raise ValueError("Cropped input needs to be a tensor of 2, 3 or 4 dimensions")
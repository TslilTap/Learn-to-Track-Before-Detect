""" Helper functions used to implement the ViterbiNet algorithm"""

################ IMPORTS #########
import torch                     #
##################################

def beam_mask(emis_k,beta=0.5):
    val_max = torch.max(emis_k)
    val_min = torch.min(emis_k)
    thresh = val_min + ((val_max - val_min) * beta)
    mask = (emis_k >= thresh)
    return mask


def get_bbox(center,nr=8,nv=8,Nr=200,Nv=64):
    r = max(nr,min(Nr-nr,center[0]))
    v = max(nv,min(Nv-nv,center[1]))
    return [int(r - nr),int(v - nv),int(r + nr),int(v + nv)]

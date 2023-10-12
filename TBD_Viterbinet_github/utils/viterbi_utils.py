""" Helper functions used to implement the ViterbiNet algorithm"""

################ IMPORTS #########
import torch                     #
##################################

def beam_mask(emis_k,beta=0.7):
    val_max = torch.max(emis_k)
    val_min = torch.min(emis_k)
    thresh = val_min + ((val_max - val_min) * beta)
    mask = (emis_k >= thresh)
    return mask


# def get_bbox(center,environment):
#     bbox_param = environment["bbox_param"]
#     Nr = environment["num_range_bins"]
#     Nd = environment["num_doppler_bins"]
#     r = max(bbox_param[0],min(Nr-bbox_param[0],center[0]))
#     d = max(bbox_param[0],min(Nd-bbox_param[1],center[1]))
#     return [int(r - bbox_param[0]),int(d - bbox_param[1]),int(r + bbox_param[0]),int(d + bbox_param[1])]

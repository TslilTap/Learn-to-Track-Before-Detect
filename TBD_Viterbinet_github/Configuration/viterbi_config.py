## Viterbi algorithm parameters
import torch

# algorithmic models
from Algorithms.MotionModel import motion_model
from Algorithms.DNNTracker import dnn_tracker

# Deteriming Viterbi Parameters
SNR = 20

beam_search = 0.7
bounding_region = 'wo'  # 'wo' weighted origin 'gb' look back m steps
look_back_m_steps = 3

dnn_dict_path = f"G:/Shared drives/Track-Before-Detect/Track-Before-Detect/TBDViterbiNet/Training/{SNR}_SNR_stats" # dnn tracker dictionary path

dnn_tracker.load_state_dict(torch.load(dnn_dict_path))



viterbinet_param = {"motion_model": motion_model,
                    "dnn_tracker": dnn_tracker,
                    'beta': beam_search,
                    'bbox_type': bounding_region,
                    'm': look_back_m_steps}


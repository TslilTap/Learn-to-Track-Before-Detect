import torch
from torch.utils.data import DataLoader
from Algorithms.ViterbiNet import ViterbiNet
from Configuration.viterbi_config import viterbinet_param


SNR = 20

track_data_path = f"G:/Shared drives/Track-Before-Detect/Track-Before-Detect/TBDViterbiNet/Data/track_data/track_data_{SNR}_SNR"
name = "viterbinet"


track_loader = DataLoader(torch.load(track_data_path))
viterbinet = ViterbiNet(viterbinet_param=viterbinet_param)

simulation_param = {
    "track_loader": track_loader,
    "tracker_model": viterbinet,
    "tracker_name": name
}


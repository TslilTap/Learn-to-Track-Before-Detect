"""configuration of training"""
import torch
from Configuration.environment_config import Environment
from Configuration.environment_config import environment
from utils.train_utils import Training_Stats
from torch.utils.data import DataLoader

SNR = 20

learning_rate = 0.0001
batch_size = 500
weight_decay = 0
environment_model = Environment(environment)
train_data_file = f"{SNR} train data file name"
valid_data_file = f"{SNR} valid data file name"
checkpoint_path = f"{SNR} checkpoint"

nr, nv = (8, 8)

training_stats_list = [
    Training_Stats(
        environment_model,
        ce_weight=0.01,
        frame_weight=0.0,
        epochs=50,
        bbox_param=(nr, nv)
    ),
    Training_Stats(
        environment_model,
        ce_weight=0.00,
        frame_weight=0.0,
        epochs=50,
        bbox_param=(nr, nv)
    )
]

valid_stats = Training_Stats(
    environment_model,
    ce_weight=0.0,
    frame_weight=0.1,
    epochs=0,
    bbox_param=(nr, nv)
)

train_loader = DataLoader(
    torch.load(train_data_file),
    batch_size=batch_size,
    shuffle=True)

valid_loader = DataLoader(
    torch.load(valid_data_file),
    batch_size=batch_size,
    shuffle=True)

train_param = {
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "train_loader": train_loader,
    "valid_loader": valid_loader,
    "train_stats_list": training_stats_list,
    "valid_stats": valid_stats
}
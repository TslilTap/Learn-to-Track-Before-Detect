"""configuration of training"""
import torch
from Configuration.environment_config import environment
from utils.train_utils import loss_param
from torch.utils.data import DataLoader

SNR = 20

learning_rate = 0.001
batch_size = 500
weight_decay = 0

train_data_path = "insert train data path here"
valid_data_path = "insert valid data path here"
checkpoint_path = f"{SNR}_SNR_stats"

training_loss_param_list = [
    loss_param(
        epochs=50,
        environment=environment,
        ce_weight=0.2,
        frame_weight=0.0),
    loss_param(
        epochs=50,
        environment=environment,
        ce_weight=0.0,
        frame_weight=0.0
    )
]

valid_loss_param = loss_param(
    epochs=0,
    environment=environment,
    ce_weight=0.0,
    frame_weight=0.0
)

train_loader = DataLoader(
    torch.load(train_data_path),
    batch_size=batch_size,
    shuffle=True)

valid_loader = DataLoader(
    torch.load(valid_data_path),
    batch_size=batch_size,
    shuffle=True)

train_param = {
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "train_loader": train_loader,
    "valid_loader": valid_loader,
    "train_loss_param_list": training_loss_param_list,
    "valid_loss_param": valid_loss_param,
    "checkpoint_path": checkpoint_path
}

## DNN tracker parameters
from Configuration.environment_config import environment

num_channels = 32
drop_rate = 0.25

tracker_param = {
    "num_channels": num_channels,
    "drop_rate": drop_rate,
    "environment": environment
}
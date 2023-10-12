from Configuration.training_config import train_param
from Configuration.tracker_config import tracker_param
from utils.train_utils import train_model
from Algorithms.DNNTracker import DNNTracker

model = DNNTracker(tracker_param=tracker_param)
train_model(model=model,train_param=train_param)

from Configuration.simulation_config import simulation_param
from utils.simulation_utils import Find_Accuracy

track_loader = simulation_param["track_loader"]
tracker = simulation_param["tracker_model"]
name = simulation_param["tracker_name"]

for (observation,label) in track_loader:
    observation = observation.squeeze(0)
    cheat_state = label[0]
    estimated_track = tracker(observation,cheat_state=cheat_state)
    Find_Accuracy(estimated_track,label,name)
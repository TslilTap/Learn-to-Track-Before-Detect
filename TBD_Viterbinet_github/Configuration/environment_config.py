""" Setting the image dimensions and bounding region """
from utils.environment_utils import Environment_model

# Global parameters ( determine the image size)
num_range_bins = 200
num_doppler_bins = 64

# Possible range for our simulation
r_min = 0.0
r_max = 2985.0

# Possible velocities in our simulation
vr_min = -369.1406
vr_max = 369.1406

bbox_distance_range = 8
bbox_distance_doppler = 8

range_bounds = (r_min, r_max)
doppler_bound = (vr_min, vr_max)

environment_param = {'image_size': (num_range_bins, num_doppler_bins),
                     'bbox_param':(bbox_distance_range,bbox_distance_doppler),
                     'range_bounds': range_bounds,
                     'doppler_bound': doppler_bound}



environment = Environment_model(environment_param)
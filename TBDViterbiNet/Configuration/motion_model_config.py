""" Setting the motion model distribution and type """
from environment_config import Environment
from environment_config import environment

mm_type = 'RandomWalk'
sigma_r = 15.0
sigma_d = 10.0
T = 1.0


if mm_type == 'RandomWalk':
    motion_model = {'environment': Environment(environment),
                    'sigma_r': sigma_r,
                    'sigma_v': sigma_d,
                    'T': 0.0}
elif mm_type == 'StraightLine':
    motion_model = {'environment': Environment(environment),
                    'sigma_r': sigma_r,
                    'sigma_v': sigma_d,
                    'T': T}
else:
    raise ValueError("Invalid motion model type input")

""" Setting the motion model distribution and type """
from Configuration.environment_config import environment
mm_type = 'RandomWalk'
sigma_r = 15.0
sigma_d = 10.0
T = 1.0



if mm_type == 'RandomWalk':
    motion_model_param = {'environment': environment,
                    'sigma_r': sigma_r,
                    'sigma_d': sigma_d,
                    'T': 0.0}
elif mm_type == 'StraightLine':
    motion_model_param = {'environment': environment,
                    'sigma_r': sigma_r,
                    'sigma_d': sigma_d,
                    'T': T}
else:
    raise ValueError("Invalid motion model type input")

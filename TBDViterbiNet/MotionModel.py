import torch
import torch.distributions as dist
import torch.nn.functional as F
from Configuration.motion_model_config import motion_model


class MotionModel:
    """
    Random Walk Motion Model
    state  variable: x_k = [r_k, f_k]^T
    State evolution equation: x_k = x_(k-1) + w_k;        w_k~[N(0,sigma_r), N(0, sigma_d)]^T
    """
    def __init__(self, motion_moodel, epsilon = 1e-10):
        self.env = motion_moodel['environment'] # model of range and velocity bins
        self.sigma_r = torch.tensor(motion_moodel['sigma_r']) # range transition variance
        self.sigma_d = torch.tensor(motion_moodel['sigma_d']) # doppler velocity transition variance
        self.T = motion_moodel['T'] # time [sec] passed between each sample

        r_bin = self.env.range_vec[1]-self.env.range_vec[0] # size of range bin [m]
        d_bin = self.env.doppler_vec[1]-self.env.doppler_vec[0] # size of doppler velocity bin [m/sec]

        # Defines the weight of each pixels
        self.log_prob_r = torch.zeros([self.env.num_range_bins])
        self.log_prob_d = torch.zeros([self.env.num_doppler_bins])

        dist_r = dist.Normal(0,self.sigma_r)
        dist_d = dist.Normal(0,self.sigma_d)

        r_min = -r_bin/2
        d_min = -d_bin/2

        # write as a separate function which the Init method calls
        for r in range(self.env.num_range_bins):
            r_max = r_min.clone() + r_bin
            prob = dist_r.cdf(r_max) - dist_r.cdf(r_min)
            self.log_prob_r[r] = torch.log(prob + epsilon)
            r_min = r_max.clone()

        for d in range(self.env.num_doppler_bins):
            d_max = d_min.clone() + d_bin
            prob = dist_d.cdf(d_max) - dist_d.cdf(d_min)
            self.log_prob_d[d] = torch.log(prob + epsilon)
            d_min = d_max.clone()

    # computes the transition probability
    def step_cost(self, current_state, prev_state):
        "tranistion probability"
        r_curr_idx,d_curr_idx = current_state   # range and doppler velocity current idx
        r_prev_idx,d_prev_idx = prev_state      # previous range and doppler state idx
        r_prev_val,d_prev_val = self.env.idx2val(r_prev_idx,d_prev_idx)  # previous range and doppler state values
        expected_r_val, expected_d_val = self.next(r_prev_val,d_prev_val) # expected range and doppler state values
        expected_r_idx, expected_d_idx = self.env.val2idx(expected_r_val,expected_d_val) # expected range and doppler state idx

        r_diff = int(abs(r_curr_idx - expected_r_idx)) # pixel distance between expected and current range
        d_diff = int(abs(d_curr_idx - expected_d_idx)) # pixel distance between expected and current doppler velocity
        return self.log_prob_r[r_diff].item() + self.log_prob_d[d_diff].item()  # assume independence



    def next(self, range_value:float, velocity_value: float, rnd:bool=False) -> tuple:
        if rnd:
            dr = torch.randn(1)*self.sigma_r
            dd = torch.randn(1)*self.sigma_d
        else:
            dr = 0
            dd = 0
        r_next = range_value + velocity_value * self.T + dr # R(k+1) = R(k) + v(k)*T + w_r(k)
        d_next = velocity_value + dd # V(k+1) = V(k) + w_v(k)

        return r_next,d_next

    def prev(self,range_value: float, velocity_value:float) -> tuple:
        r_prev = range_value - velocity_value
        v_prev = velocity_value
        return r_prev, v_prev

#
# class Track:
# ''' trellis diagram '''
#     def __init__(self,r,v,cost=0,prev=None):
#         self.r = r
#         self.v = v
#         if prev is None:
#             self.track = list()
#         else:
#             self.track = prev.track.copy()
#         self.cost = cost
#         self.track.append((r,v))
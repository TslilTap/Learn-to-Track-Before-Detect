import utils.viterbi_utils as vit_utils  # maybe change name
import torch
from Configuration.viterbi_config import viterbinet_param

class ViterbiNet:
    def __init__(self,
                 viterbinet_param):

        # viterbi parameters
        self.mm = viterbinet_param["motion_model"]  # motion model
        self.cost_dnn = viterbinet_param["dnn_tracker"]  # cost dnn
        self.beta = viterbinet_param["beta"]
        self.bbox_type = viterbinet_param["bbox_type"]
        self.m = viterbinet_param["m"]
        self.env = self.mm.env

        # global parameters
        self.Nr = self.env.num_range_bins
        self.Nv = self.env.num_doppler_bins

    def __call__(self,
                 observations: torch.Tensor,
                 cheat_state: tuple = None,
                 beta: float = None,
                 bbox_type: str = None,
                 m: int = None):
        # Pass dictionary from Viterbi Configuration
        if bbox_type is not None:
            self.bbox_type = bbox_type
        if beta is not None:
            self.beta = beta
        if m is not None:
            self.m = m
        self.bbox_possible = False

        num_frames = observations.shape[0]  # Time horizon (length of the track)
        self.tracks = list()
        # initialize the algorithm
        if cheat_state is not None:
            self.cheat_init(cheat_state)

        # process each iteration
        for k in range(num_frames):
            z_k = observations[k, :, :, :].clone()
            self.Viterbi_step(z_k)  # implements a step of the viterbi algorithm
            # updates viterbi parameters trellis diagram cost and checks if the boundign box is possible

        best_track = self.tracks[int(torch.argmax(self.costs).item())]
        if cheat_state is not None:
            del best_track[0]
        return best_track

    def cheat_init(self, cheat_state):
        r = int(cheat_state[0])
        d = int(cheat_state[1])

        ranges = torch.zeros([1])
        dopplers = torch.zeros([1])
        costs = torch.zeros([1])

        ranges[0] = r
        dopplers[0] = d
        costs[0] = 0
        track = list()
        track.append((r, d))
        tracks = list()
        tracks.append(track)

        self.tracks = tracks
        self.ranges = ranges
        self.dopplers = dopplers
        self.costs = costs

        self.bbox_possible = True

    def Viterbi_step(self, z_k):
        bbox_bool = (self.bbox_possible and self.bbox_type is not None)
        if bbox_bool:
            bbox = self.env.get_bbox(self.set_bbox_origin())
        else:
            bbox = None
        LogLikelihood = self.cost_dnn(x=z_k, bbox=bbox)

        mask = vit_utils.beam_mask(LogLikelihood, self.beta)
        ranges, dopplers = torch.nonzero(mask, as_tuple=True)
        num_candidates = len(ranges)

        costs = torch.zeros([num_candidates])
        tracks = list()

        for i in range(num_candidates):
            r = int(ranges[i])
            d = int(dopplers[i])
            if len(self.tracks) == 0:
                # initial state
                track_prev = list()
                step_cost = 0
            else:
                track_prev, step_cost = self.find_prev((r, d))

            costs[i] = LogLikelihood[r, d] + step_cost
            track_prev.append((r, d))
            tracks.append(track_prev)

        self.tracks = tracks
        self.ranges = ranges
        self.dopplers = dopplers
        self.costs = costs

        if not bbox_bool:
            self.check_bbox_possible()

    def find_prev(self, current):
        """ Trellis diagram """
        costs = torch.zeros([len(self.tracks)])

        for i in range(len(self.tracks)):
            r = int(self.ranges[i])
            d = int(self.dopplers[i])
            cost = self.costs[i].clone()
            costs[i] = cost + self.mm.step_cost(current, (r, d))
        prev_idx = torch.argmax(costs).item()
        track_prev = self.tracks[prev_idx].copy()
        track_prev.append(current)
        cost = costs[prev_idx]
        return track_prev, cost

    def set_bbox_origin(self):
        if self.bbox_type == 'wo':
            center = self.Weighted_Origin()
        elif self.bbox_type == 'gb':
            center = self.Go_Back_m()
        else:
            raise ValueError(
                self.bbox_type + " is not a registered bbox type. try to use 'wo' (weighted origin) or 'gb' (go back m)")
        return center

    def check_bbox_possible(self):
        check = True
        r_true, v_true = self.ranges[0], self.dopplers[0]

        if self.bbox_type == 'gb':
            if len(self.tracks[0]) < self.m:
                check = False
        for i in range(len(self.tracks)):
            r, d = self.tracks[i][0]
            if r != r_true or d != v_true:
                check = False
        self.bbox_possible = check

    def Weighted_Origin(self):
        N = len(self.tracks)
        ranges = torch.zeros([N])
        velocities = torch.zeros([N])
        for i in range(N):
            r, v = self.env.idx2val(int(self.ranges[i]), int(self.dopplers[i]))
            ranges[i], velocities[i] = self.mm.next(r, v)

        weights = torch.softmax(self.costs, dim=0)
        r_avg = torch.sum(ranges * weights, dim=0)
        v_avg = torch.sum(velocities * weights, dim=0)
        r_idx, v_idx = self.env.val2idx(r_avg, v_avg)
        return [r_idx, v_idx]

    def Go_Back_m(self):
        N = len(self.tracks)
        weights = torch.zeros([N])
        for i in range(N):
            track = self.tracks[i]
            weights[i] = track.cost.detach
        best = int(torch.argmax(weights).item())
        best_track = self.tracks[best].track.detach()
        past_state = best_track[len(best_track) - self.m - 1]
        r = past_state[0]
        d = past_state[1]
        r, v = self.env.idx2val(r, d)
        for i in range(self.m):
            r, v = self.mm.next(r, d)
        r_idx, v_idx = self.env.val2idx(r, d)
        center = torch.tensor([r_idx, v_idx])
        return center.unsqueeze(0)

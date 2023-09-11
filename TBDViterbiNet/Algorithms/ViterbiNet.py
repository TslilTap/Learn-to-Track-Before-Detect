import utils.viterbi_utils as vit_utils  # maybe change name
import torch


class ViterbiNet:
    def __init__(self,
                 mm,
                 BBoxTracker):

        # viterbi parameters
        self.mm = mm # motion model
        self.tracks = list()

        # global parameters
        self.Nr = self.mm.Nr
        self.Nv = self.mm.Nv

        # Tracker Networks
        self.BBT = BBoxTracker


    def __call__(self, observations:torch.Tensor, cheat_state:tuple=None, beta: float =0.7, bbox_type:str =None,m: int =3):
        # Pass dictionary from Viterbi Configuration
        self.bbox_type = bbox_type
        self.beta = beta
        self.m = m
        self.bbox_possible = False



        K = observations.shape[0]  # Time horizon (length of the track)
        self.tracks = list()
        # initialize the algorithm
        if cheat_state is not None:
            self.cheat_init(cheat_state)

        # process each iteration
        for k in range(K):
            x_k = observations[k, :, :, :].detach().clone()
            self.Viterbi_step(x_k)  # implements a step of the viterbi algorithm
            #updates viterbi parameters trellis diagram cost and checks if the boundign box is possible

        best_track = self.tracks[int(torch.argmax(self.costs).item())]
        if cheat_state is not None:
            del best_track[0]
        return best_track



    def cheat_init(self,cheat_state):
        r = int(cheat_state[0])
        v = int(cheat_state[1])


        ranges = torch.zeros([1])
        vels = torch.zeros([1])
        costs = torch.zeros([1])
        tracks = list()

        ranges[0] = r
        vels[0] = v
        costs[0] = 0
        tracks.append((r,v))

        self.tracks = tracks
        self.ranges = ranges
        self.vels = vels
        self.costs = costs

        self.bbox_possible = True

    def Viterbi_step(self,x_k):
        bbox_bool = (self.bbox_possible and self.bbox_type is not None)
        if bbox_bool:
            bbox = vit_utils.get_bbox(self.set_bbox_origin())
        else:
            bbox = None
        x = self.BBT(x_k,bbox)

        mask = vit_utils.beam_mask(x, self.beta)


        N = sum(sum(mask))
        ranges = torch.zeros([N])
        vels = torch.zeros([N])
        costs = torch.zeros([N])
        tracks = list()

        i = 0
        for r in range(self.Nr):
            for v in range(self.Nv):
                if mask[r,v]:
                    # this state is a candidate
                    if len(self.tracks) == 0:
                        # initial state
                        track_prev = list()
                        step_cost = 0
                    else:
                        track_prev, step_cost = self.backward_step((r,v))

                    ranges[i] = r
                    vels[i] = v
                    costs[i] = x[r,v] + step_cost
                    track_prev.append((r,v))
                    tracks.append(track_prev)

                    i += 1
        self.tracks = tracks
        self.ranges = ranges
        self.vels = vels
        self.costs = costs

        if not bbox_bool:
            self.check_bbox_possible()


    def backward_step(self,current):
        """ Trellis diagram """
        N = len(self.tracks)
        costs = torch.zeros([N])

        for i in range(N):
            r = self.ranges[i].clone()
            v = self.vels[i].clone()
            cost = self.costs[i].clone()
            costs[i] = cost + self.mm.step_cost(current,(r,v))
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
            raise ValueError(self.bbox_type + " is not a registered bbox type. try to use 'wo' (weighted origin) or 'gb' (go back m)")
        return center

    def check_bbox_possible(self):
        check = True
        r_true,v_true = self.tracks[0].track[0]

        if self.bbox_type == 'gb':
            if len(self.tracks[0].track) < self.m:
                check = False
        for i in range(len(self.tracks)):
            r, v = self.tracks[i].track[0]
            if r != r_true or v != v_true:
                check = False
        self.bbox_possible = check



    def Weighted_Origin(self):
        N = len(self.tracks)
        ranges = torch.zeros([N])
        velocities = torch.zeros([N])
        for i in range(N):
            r,v = self.mm.ind2val(self.ranges[i].clone(),self.vels[i].clone())
            ranges[i],velocities[i] = self.mm.next(r,v)

        weights = torch.softmax(self.costs,dim=0)
        r_avg = torch.sum(ranges*weights,dim=0)
        v_avg = torch.sum(velocities*weights,dim=0)
        r_idx,v_idx = self.mm.val2idx(r_avg,v_avg)
        return [r_idx,v_idx]

    def Go_Back_m(self):
        N = len(self.tracks)
        weights = torch.zeros([N])
        for i in range(N):
            track = self.tracks[i]
            weights[i] = track.cost.detach
        best = int(torch.argmax(weights).item())
        best_track = self.tracks[best].track.detach()
        past_state = best_track[len(best_track)-self.m-1]
        r = past_state[0]
        v = past_state[1]
        r,v = self.mm.ind2val(r,v)
        for i in range(self.m):
            r,v = self.mm.next(r,v)
        r_idx,v_idx = self.mm.val2idx(r,v)
        center = torch.tensor([r_idx,v_idx])
        return center.unsqueeze(0)
import torch

class SingleFrameDetection():
    def __init__(self,
                 BBoxTracker):
        self.BBT = BBoxTracker
        self.env = BBoxTracker.env

    def __call__(self, emis):
        X = emis.detach().clone()
        track_len = X.shape[0]
        track = list()
        for t in range(track_len):
            x = self.BBT(X[t,:,:,:])
            r,d = self.env.idx2tuple(torch.argmax(x,dim=1))
            track.append((r,d))
        return track

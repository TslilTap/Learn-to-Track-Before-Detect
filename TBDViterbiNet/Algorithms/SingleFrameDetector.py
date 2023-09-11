import torch

class SingleFrameDetection():
    def __init__(self,
                 BBoxTracker):
        self.BBT = BBoxTracker
        self.Nr = self.BBT.Nr
        self.Nv = self.BBT.Nv

    def __call__(self, emis):
        X = emis.detach().clone()
        track_len = X.shape[0]
        track = list()
        for t in range(track_len):
            x = self.BBT(X[t,:,:,:])
            SFD = torch.argmax(x,dim=1)
            r = SFD[t].item() // self.Nv
            v = SFD[t].item() % self.Nv
            track.append((r,v))
        return track

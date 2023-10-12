from torch.utils.data import Dataset
import torch


class Radar_Dataset(Dataset):
    def __init__(self, observation, range_label, doppler_label, environment):
        self.data = observation  # radar observation
        self.range_label = range_label  # label, range axis
        self.doppler_label = doppler_label  # label, doppler axis
        self.env = environment

        self.datasize = range_label.size()

    def __getitem__(self, index, flat_label=True):
        data = self.data[index]

        # get label value
        r_val = self.range_label[index]
        d_val = self.doppler_label[index]

        r, d = self.env.val2idx(r_val, d_val)  # get pixel
        label = torch.tensor([r, d])
        return data, label

    def __len__(self):
        return len(self.data)


class Track_Dataset(Dataset):
    def __init__(self, observation, range_label, doppler_label, environment):
        self.data = observation
        self.range_label = range_label
        self.doppler_label = doppler_label
        self.env = environment

        self.datasize = range_label.size()
        self.num_tracks = self.datasize[0]
        self.num_samples = self.datasize[1]

    def __getitem__(self, index):
        data = self.data[index]
        r_val = self.range_label[index]
        d_val = self.doppler_label[index]

        label = list()
        for i in range(self.num_samples):
            r, d = self.env.val2idx(r_val[i], d_val[i])
            label.append((r, d))
        return data, label

    def __len__(self):
        return len(self.data)

import datetime
import random

import torch
import torch.nn.functional as F

import utils.viterbi_utils as vit_utils
import utils.dnn_tracker_utils as trac_utils


class Loss_Param:
    def __init__(self,
                 environment,
                 ce_weight: float = 0.0,
                 frame_weight: float = 0.0,
                 epochs: int = 50,
                 bbox_param: tuple = (8, 8)):

        '''

        :param ce_weight: float in range: [0,1]. the weight of the cross entropy loss. default: 0.0
        :param frame_weight: float in range: [0,1]. the weight of the frame loss. default: 0.0
        :param epochs: positive integer. number of epochs. default: 50.
        :param bbox_param: tuple of two positive integers. default: (8,8)
        '''
        self.Nr = environment.num_range_bins  # number of range bins
        self.Nd = environment.num_doppler_bins  # number of doppler velocity bins
        self.epochs = epochs  # number of epochs
        if ce_weight > 1.0 or ce_weight < 0.0:
            raise ValueError("CE_weight must be a float between 0.0 and 1.0")
        if frame_weight > 1.0 or frame_weight < 0.0:
            raise ValueError("frame_weight must be a float between 0.0 and 1.0")

        self.CE_weight = ce_weight  # weight of the Cross Entropy loss.
        self.frame_weight = frame_weight  # weight of the frame loss
        self.bbox_param = bbox_param  # tuple, max distance from the center in range and doppler axis

        # initialize the loop
        self.len_loader = 0  # number of batches in the epoch
        self.loss = torch.tensor(0.0)  # loss
        self.bbox_acc = 0.0  # bounding box accuracy
        self.frame_acc = 0.0  # frame accuracy

    def __call__(self, model, observation, labels):
        batch_size = observation.shape[0]  # amount of samples in the batch

        bbox_acc = 0.0  # bounding box accuracy
        frame_acc = 0.0  # frame accuracy
        loss = torch.tensor(0.0)

        for i in range(batch_size):
            # create a one_hot
            one_hot = torch.zeros([self.Nr, self.Nd])
            label = labels[i]
            r = label[0]
            v = label[1]
            one_hot[r, v] = 1

            if self.frame_weight < 1:  # calculate bbox loss
                # randomly move the bbox around the center
                center = [r + random.randint(-self.bbox_param[0] + 1, self.bbox_param[0]),
                          v + random.randint(-self.bbox_param[1] + 1, self.bbox_param[1])]

                bbox = vit_utils.get_bbox(center, nr=self.bbox_param[0], nv=self.bbox_param[1], Nr=self.Nr, Nv=self.Nd)

                one_hot, true_label = self.get_one_hot(labels[i], bbox)

                x_bbox = model(observation, bbox, restore=False).view(-1)
                loss_bbox, acc_bbox_temp = self.get_loss_and_accuracy(x_bbox, one_hot, true_label)
                loss += loss_bbox * (1 - self.frame_weight)
                bbox_acc += acc_bbox_temp

            if self.frame_weight < 1:
                one_hot, true_label = self.get_one_hot(labels[i])

                x_frame = model(observation, restore=False).view(-1)
                loss_frame, acc_frame_temp = self.get_loss_and_accuracy(x_frame, one_hot, true_label)
                loss += loss_frame * self.frame_weight
                frame_acc += acc_frame_temp

        bbox_acc /= batch_size
        frame_acc /= batch_size
        loss /= batch_size
        loss.backward()

        self.bbox_acc += bbox_acc
        self.frame_acc += frame_acc
        self.loss += loss.item()
        self.len_loader += 1

    def get_one_hot(self, label, bbox=None):
        one_hot = torch.zeros([self.Nr, self.Nd])  # a one_hot of the entire frame
        one_hot[label[0], label[1]] = 1  # the target pixel
        if bbox is not None:
            # if bbox loss
            one_hot = trac_utils.crop(one_hot, bbox).view(-1)  # crop and flatten one_hot
            true_label = int(torch.argmax(one_hot, dim=1).item())  # find true label
        else:
            true_label = label[0] * self.Nd + label[1]  # find true label
        return one_hot, true_label

    def get_loss_and_accuracy(self, x, one_hot, true_label):
        loss = 0.0
        acc = 0
        if self.CE_weight < 1:
            num_neg = len(one_hot) - 1  # number of pixels with 'no target'
            W = num_neg * one_hot.detach() + num_neg
            W = W / torch.sum(W)  # the pixel with the target has half of the weight. the other half is equally
            # distributed among all the other pixels
            loss += F.binary_cross_entropy(torch.exp(x), one_hot, weight=W) * (
                        1 - self.CE_weight)  # weighted binary cross entropy
        if self.CE_weight > 0:
            loss += F.cross_entropy(x, true_label) * self.CE_weight  # weighted cross entropy
        if torch.argmax(x, dim=1).long() == true_label:
            acc = 1
        return loss, acc

    def reset(self):
        loss = (self.loss / self.len_loader).clone()
        print(f"Loss: {self.loss / self.len_loader:.4f}")
        if self.frame_weight < 1:
            print(f"bbox acc: {(self.bbox_acc / self.len_loader) * 100: .2f}%")
        if self.frame_weight > 0:
            print(f"frame acc: {(self.frame_acc / self.len_loader) * 100:.2f}%")

        self.loss = torch.tensor(0.0)
        self.bbox_acc = 0.0
        self.frame_acc = 0.0
        return loss


def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

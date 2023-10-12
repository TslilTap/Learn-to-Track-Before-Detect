import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import datetime
import random
import utils.dnn_tracker_utils as trac_utils

def train_model(model,
                train_param):
    optimizer = optim.Adam(model.parameters(),
                           lr=train_param['learning_rate'],
                           weight_decay=train_param['weight_decay'],
                           betas=(0.90, 0.999))
    train_loader = train_param['train_loader']
    valid_loader = train_param['valid_loader']
    train_loss_param_list = train_param['train_loss_param_list']
    valid_loss_param = train_param['valid_loss_param']
    checkpoint_path = train_param['checkpoint_path']

    val_loss_best = torch.tensor(float('inf'))
    for loss_param in train_loss_param_list:
        epochs = loss_param['epochs']

        for epoch in range(epochs):
            start_time = time.time()
            train_loss = 0.0
            train_bbox_acc = 0.0
            train_frame_acc = 0.0
            valid_loss = 0.0
            valid_bbox_acc = 0.0
            valid_frame_acc = 0.0

            # Train
            model.train()
            for i, (observation, label) in enumerate(train_loader):
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                loss, bbox_acc, frame_acc = bbox_loss_and_accuracy(model,
                                                                   observation=observation.clone(),
                                                                   labels=label,
                                                                   loss_param=loss_param)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_bbox_acc += bbox_acc
                train_frame_acc += frame_acc

            model.eval()
            # Validate
            with torch.no_grad():
                for i, (observation, label) in enumerate(valid_loader):
                    torch.cuda.empty_cache()
                    loss_valid, bbox_acc_valid, frame_acc_valid = bbox_loss_and_accuracy(model,
                                                                                         observation=observation.clone(),
                                                                                         labels=label,
                                                                                         loss_param=valid_loss_param)
                    valid_loss += loss_valid.item()
                    valid_bbox_acc += bbox_acc_valid
                    valid_frame_acc += frame_acc_valid

            train_loss /= len(train_loader)
            train_bbox_acc /= len(train_loader)
            train_frame_acc /= len(train_loader)
            valid_loss /= len(valid_loader)
            valid_bbox_acc /= len(valid_loader)
            valid_frame_acc /= len(valid_loader)

            # Print statistics
            epoch_time = time.time() - start_time
            remaining_time = epoch_time * (epochs - epoch - 1)

            formatted_epoch_time = format_time(epoch_time)
            formatted_remaining_time = format_time(remaining_time)

            print(f"Epoch {epoch + 1}/{epochs}")
            print("Training stats:")
            print(f"Training Loss = {train_loss: .4f}")
            if loss_param["frame_weight"] < 1:
                print(f"Training BBox Accuracy = {train_bbox_acc*100: .2f}%")
            if loss_param["frame_weight"] > 0:
                print(f"Training Frame Accuracy = {train_frame_acc*100: .2f}%")

            print("Valid stats:")
            print(f"Valid Loss = {valid_loss: .4f}")
            if valid_loss_param["frame_weight"] < 1:
                print(f"Valid BBox Accuracy = {valid_bbox_acc*100: .2f}%")
            if valid_loss_param["frame_weight"] > 0:
                print(f"Valid Frame Accuracy = {valid_frame_acc*100: .2f}%")

            print(f"Time taken: {formatted_epoch_time}, Estimated remaining time: {formatted_remaining_time}")

            # saving the model
            if checkpoint_path is not None:
                if valid_loss < val_loss_best:
                    val_loss_best = valid_loss
                    torch.save(model.state_dict(), checkpoint_path)



def bbox_loss_and_accuracy(model,
                           observation,
                           labels,
                           loss_param):
    batch_size = observation.shape[0]  # amount of samples in the batch
    environment = loss_param["environment"]
    ce_weight = loss_param["ce_weight"]
    frame_weight = loss_param["frame_weight"]
    bbox_param = environment.bbox_param

    bbox_acc = 0.0  # bounding box accuracy
    frame_acc = 0.0  # frame accuracy
    loss = torch.tensor(0.0)

    if frame_weight > 1.0 or frame_weight < 0.0:
        raise ValueError("frame_weight needs to be in range [0,1]")

    for i in range(batch_size):
        if frame_weight < 1.0:
            center = [labels[i][0] + random.randint(-bbox_param[0] + 1, bbox_param[0]),
                      labels[i][1] + random.randint(-bbox_param[1] + 1, bbox_param[1])]
            bbox = environment.get_bbox(center)
            one_hot, true_label = get_one_hot(environment,labels[i], bbox)
            x_bbox = model(observation[i], bbox, restore=False).view(-1)
            loss += (1-frame_weight)*get_loss(x_bbox,one_hot,true_label,ce_weight)
            bbox_acc += get_accuracy(x_bbox,true_label)

        if frame_weight > 0.0:
            one_hot, true_label = get_one_hot(environment,labels[i])
            x_frame = model(observation[i], restore=False).view(-1)
            loss += (1-frame_weight)*get_loss(x_frame,one_hot,true_label,ce_weight)
            frame_acc += get_accuracy(x_frame,true_label)
    loss /= batch_size
    bbox_acc /= batch_size
    frame_acc /= batch_size
    return loss, bbox_acc, frame_acc


def get_one_hot(environment,label,bbox=None):
    one_hot = environment.zeros()  # a one_hot of the entire frame
    one_hot[label[0], label[1]] = 1  # the target pixel
    if bbox is not None:
        # if bbox loss
        one_hot = trac_utils.crop(one_hot, bbox).reshape(-1)  # crop and flatten one_hot
        true_label = torch.argmax(one_hot, dim=0)  # find true label
    else:
        true_label = environment.tuple2idx(label) # find true label
    return one_hot, true_label

def get_loss(x, one_hot, true_label,ce_weight):
    '''
    :param x: input tensor
    :param one_hot: target tensor
    :param true_label: target index
    :param ce_weight: cross entropy weight. needs to be in range [0,1]
    :return:
    '''
    loss = 0.0
    if ce_weight > 1.0 or ce_weight < 0.0:
        raise ValueError("ce_weight needs to be in range [0,1]")
    if ce_weight < 1.0:
        num_neg = len(one_hot) - 1  # number of pixels with 'no target'
        W = num_neg * one_hot.detach() + num_neg
        # the pixel with the target has half of the weight. the other half is equally
        # distributed among all the other pixels
        loss += F.binary_cross_entropy(torch.exp(x), one_hot, weight=W) * (
                    1 - ce_weight)  # weighted binary cross entropy
    if ce_weight > 0.0:
        loss += F.cross_entropy(x, true_label) * ce_weight  # weighted cross entropy
    return loss

def get_accuracy(x,true_label):
    if torch.argmax(x, dim=0).item() == true_label.item():
        return 1
    return 0

def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def loss_param(epochs,
               environment,
               ce_weight=0.0,
               frame_weight=0.0):
    return {"epochs": epochs,
            "environment": environment,
            "ce_weight": ce_weight,
            "frame_weight": frame_weight}

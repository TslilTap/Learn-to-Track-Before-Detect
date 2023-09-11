import torch.optim as optim
import time
import torch
from utils.train_utils import format_time

def train_model(model,
               train_param):
    optimizer = optim.Adam(model.parameters(),
                           lr=train_param['learning_rate'],
                           weight_decay=train_param['weight_decay'],
                           betas=(0.90, 0.999))
    train_loader = train_param['train_loader']
    valid_loader = train_param['valid_loader']
    train_stats_list = train_param['train_stats_list']
    valid_stats = train_param['valid_stats']
    checkpoint_path = train_param['checkpoint_path']


    val_loss_best = torch.tensor(float('inf'))
    for train_stats in train_stats_list:
        epochs = train_stats


        for epoch in range(epochs):
            start_time = time.time()

            # Train
            model.train()
            for i, (input, label) in enumerate(train_loader):
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                train_stats(model,input.clone(),label.clone())
                optimizer.step()

            model.eval()
            # Validate
            with torch.no_grad():
                for i, (input, label) in enumerate(valid_loader):
                    torch.cuda.empty_cache()
                    valid_stats(model,input.clone(),label.clone())


            # Print statistics
            epoch_time = time.time() - start_time
            remaining_time = epoch_time * (epochs - epoch - 1)

            formatted_epoch_time = format_time(epoch_time)
            formatted_remaining_time = format_time(remaining_time)

            print(f"Epoch {epoch + 1}/{epochs}")
            print("Training stats")
            _ = train_stats.reset()
            print("Valid stats")
            valid_loss = valid_stats.reset()
            print(f"Time taken: {formatted_epoch_time}, Estimated remaining time: {formatted_remaining_time}")


            #saving the model
            if checkpoint_path is not None:
                if valid_loss < val_loss_best:
                    val_loss_best = valid_loss
                    torch.save(model.state_dict(), checkpoint_path)
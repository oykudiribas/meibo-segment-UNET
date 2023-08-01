
import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import DriveDataset
from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
import matplotlib.pyplot as plt





def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        #gpu
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("new_data/train/image/*"))[:3600]
    train_y = sorted(glob("new_data/train/gland/*"))[:3600]
    # train_y = sorted(glob("new_data/train/mask/*"))[:3600]

    valid_x = sorted(glob("new_data/test/image/*"))[900:1000]
    valid_y = sorted(glob("new_data/test/gland/*"))[900:1000]
    # valid_y = sorted(glob("new_data/test/mask/*"))[900:1000]

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 256
    W = 128
    size = (H, W)
    batch_size = 2
    num_epochs = 12
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")

    train_losses = []
    valid_losses = []
    with open('train_val_loss.txt', 'w') as file:

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss = train(model, train_loader, optimizer, loss_fn, device)
            valid_loss = evaluate(model, valid_loader, loss_fn, device)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            """ Saving the model """
            if valid_loss < best_valid_loss:
                data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}"
                print(data_str)

                file.write(data_str + '\n')

                best_valid_loss = valid_loss
                torch.save(model.state_dict(), checkpoint_path)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
            data_str += f'\tTrain Loss: {train_loss:.3f}\n'
            data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
            print(data_str)

            file.write(data_str + '\n')

        plt.plot(range(1, num_epochs + 1), valid_losses, label='Validation Loss', marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Epoch vs. Validation Loss')
        plt.grid(True)
        plt.legend()
        # plt.show()
        plt.xticks(range(1, num_epochs + 1))
        plt.savefig('validation_loss_graph.png')


        plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Losses')
        plt.title('Epoch vs. Validation and Train Loss')
        plt.grid(True)
        plt.legend()
        # plt.show()
        plt.xticks(range(1, num_epochs + 1))
        plt.savefig('loss_graph.png')



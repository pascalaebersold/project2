"""Code template for training a model on the ETHMugs dataset."""

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader
from eth_mugs_dataset import ETHMugsDataset
from utils import IMAGE_SIZE, compute_iou

import numpy as np

import matplotlib.pyplot as plt

# TODO: Add your model definition here
class build_model(nn.Module):  
    def __init__(self, num_classes=1):
        super(build_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * 47 * 31, 512)
        self.fc2 = nn.Linear(512, num_classes * 252 * 378)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 47 * 31)
        x = self.dropout(x)
        x = nn.functional.sigmoid(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, 252, 378)
        return x


def train(
    ckpt_dir: str,
    train_data_root: str,
    val_data_root: str,
):
    """Train function."""
    # Logging and validation settings
    log_frequency = 10
    val_batch_size = 1

    val_frequency = 5

    # TODO: Set your own values for the hyperparameters
    num_epochs = 200
    lr = 0.001
    train_batch_size = 6
    # val_batch_size = 1
    # ...

    print(f"[INFO]: Number of training epochs: {num_epochs}")
    # print(f"[INFO]: Image scale: {image_scale}")
    print(f"[INFO]: Learning rate: {lr}")
    print(f"[INFO]: Training batch size: {train_batch_size}")

    # Choose Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # TODO: Define your Dataset and DataLoader
    # ETHMugsDataset
    # Data loaders
    train_dataset = ETHMugsDataset(root_dir=train_data_root, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    val_dataset = ETHMugsDataset(root_dir=val_data_root, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True)

    # TODO: Define you own model
    model = build_model(num_classes=1)
    model.to(device)

    # TODO: Define Loss function
    #criterion = nn.CrossEntropyLoss()
    weight = torch.tensor([2.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    # TODO: Define Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # TODO: Define Learning rate scheduler if needed
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # TODO: Write the training loop!
    print("[INFO]: Starting training...")
    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (image, gt_mask) in enumerate(train_dataloader):
            image = image.to(device)
            gt_mask = gt_mask.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(image)
            loss = criterion(output, gt_mask)

            # Backward pass
            loss.backward()
            optimizer.step()

            if batch_idx % log_frequency == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

        lr_scheduler.step()

        # Save model
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch}.pth"))

        if epoch % val_frequency == 0:
            model.eval()

            val_iou = 0.0
            with torch.no_grad():
                for val_image, val_gt_mask in val_dataloader:
                    val_image = val_image.to(device)
                    val_gt_mask = val_gt_mask.to(device)

                    # Forward pass
                    val_output = model(val_image)
                    
                    val_output = (val_output>0.0025).float()
                    
                    mask = val_gt_mask.squeeze().cpu().numpy()
                    o = val_output.squeeze().cpu().numpy()
                    val_output = val_output.cpu().detach().numpy().astype(int)
                    val_gt_mask = val_gt_mask.cpu().detach().numpy().astype(int)

                    # Convert tensor to PIL Image
                    image = val_image.squeeze().permute(1,2,0)
                    image = image.cpu().numpy()
                    #o = TF.to_pil_image(val_output.squeeze())
                    #mask = TF.to_pil_image(val_gt_mask.squeeze())

                    val_iou += compute_iou(val_output, val_gt_mask)

                val_iou /= len(val_dataloader)

                val_iou *= 100

                print(f"[INFO]: Validation IoU: {val_iou.item():.2f}")

                fig, axes = plt.subplots(1, 3, figsize=(12,3))

                #print(o)
                #print(mask)
                axes[0].imshow(mask, cmap='gray')
                axes[0].axis("off")
                axes[0].set_title("Mask")

                axes[1].imshow(image)
                axes[1].axis("off")
                axes[1].set_title("Image")

                axes[2].imshow(o, cmap='gray')
                axes[2].axis("off")
                axes[2].set_title("Output")

                plt.show()

                input()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SML Project 2.")
    parser.add_argument(
        "-d",
        "--data_root",
        default="./datasets",
        help="Path to the datasets folder.",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="./checkpoints",
        help="Path to save the model checkpoints to.",
    )
    args = parser.parse_args()

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_dir = os.path.join(args.ckpt_dir, dt_string)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)

    # Set data root
    train_data_root = os.path.join(args.data_root, "train_images_378_252")
    print(f"[INFO]: Train data root: {train_data_root}")

    val_data_root = os.path.join(args.data_root, "public_test_images_378_252")
    print(f"[INFO]: Validation data root: {val_data_root}")

    train(ckpt_dir, train_data_root, val_data_root)

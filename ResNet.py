import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision import transforms
import torch.optim as optim
import time
from tqdm import tqdm
from torch.autograd import Variable


import argparse
import os
from datetime import datetime

from torch.utils.data import DataLoader
from eth_mugs_dataset import ETHMugsDataset
from utils import compute_iou

import matplotlib.pyplot as plt



class ResNet(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet, self).__init__()
        model_resnet18 = models.resnet34(weights = 73.314)
        # Modify the first convolution layer to accept 1 channel (grayscale) instead of 3 channels (RGB)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2    
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4

        # Adjust deconvolution layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=1, padding=1)

        # Upsampling layers
        self.upsample = nn.Upsample(size=(252, 378), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.upsample(x)
        x = x[:, :, :252, :378] if x.size(2) > 252 else nn.functional.pad(x, (0, 378 - x.size(3), 0, 252 - x.size(2)))
        return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)

def train(ckpt_dir: str, train_data_root: str, val_data_root: str, checkpoint_path=None):
    """Train function."""
    #log_frequency = 1
    val_frequency = 1

    num_epochs = 200
    lr = 0.001
    train_batch_size = 8
    val_batch_size = 4

    print(f"[INFO]: Number of training epochs: {num_epochs}")
    print(f"[INFO]: Learning rate: {lr}")
    print(f"[INFO]: Training batch size: {train_batch_size}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    train_dataset = ETHMugsDataset(root_dir=train_data_root, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    val_dataset = ETHMugsDataset(root_dir=val_data_root, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, pin_memory=True)

    model = ResNet()
    model.to(device)

    weight = torch.tensor([5.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

    start_epoch = 0
    # if checkpoint_path!=None:
    #     checkpoint = torch.load(checkpoint_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     print(f"[INFO]: Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")

    print("[INFO]: Starting training...")
    for epoch in range(start_epoch, num_epochs):
        model.train()

        epoch_loss = 0
        for batch_idx, (image, gt_mask) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            image, gt_mask = image.to(device), gt_mask.to(device)

            optimizer.zero_grad()

            output = model(image)
            loss = criterion(torch.mean(output, dim=1), gt_mask)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            #if batch_idx % log_frequency == 0:
                #print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")


        scheduler.step()

        print(f"Epoch {epoch+1}, Average Loss: {epoch_loss / len(train_dataloader)}")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss / len(train_dataloader),
        }, os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pth"))

        if (epoch + 1) % val_frequency == 0:
            model.eval()
            val_iou = 0.0

            with torch.no_grad():
                for val_image, val_gt_mask in val_dataloader:
                    val_image, val_gt_mask = val_image.to(device), val_gt_mask.to(device)
                    val_output = model(val_image)

                    val_output = torch.sigmoid(val_output.float())
                    val_output = (val_output > 0.75).float()

                    val_output = val_output.cpu().numpy().squeeze().astype(int)
                    val_gt_mask = val_gt_mask.cpu().numpy().squeeze().astype(int)
                    
                    #print(val_image.shape)
                    #print(val_gt_mask.shape)
                    #print(val_output.shape)
                    val_iou += compute_iou(val_output, val_gt_mask)

                val_iou = (val_iou / len(val_dataloader)) * 100
                print(f"[INFO]: Validation IoU: {val_iou:.2f}")

                # Visualization
                fig, axes = plt.subplots(1, 3, figsize=(12, 3))
                image = val_image.cpu().numpy().squeeze()
                mask = val_gt_mask
                output = val_output


                axes[0].imshow(mask, cmap='gray')
                axes[0].axis("off")
                axes[0].set_title("Mask")
                axes[1].imshow(image, cmap='gray')
                axes[1].axis("off")
                axes[1].set_title("Image")
                axes[2].imshow(output, cmap='gray')
                axes[2].axis("off")
                axes[2].set_title("Output")

                plt.savefig(os.path.join(ckpt_dir, f"validation_img_epoch_{epoch}" + f"IoU_{val_iou:.2f}" + f".jpg"))


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
    parser.add_argument(
        "--checkpoint_path",
        default="./checkpoints",
        help="Path to checkpoint",
    )
    args = parser.parse_args()

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_dir = os.path.join(args.ckpt_dir, dt_string)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)

    train_data_root = os.path.join(args.data_root, "train_images_378_252")
    print(f"[INFO]: Train data root: {train_data_root}")

    val_data_root = os.path.join(args.data_root, "public_test_images_378_252")
    print(f"[INFO]: Validation data root: {val_data_root}")



    train(ckpt_dir, train_data_root, val_data_root, args.checkpoint_path)
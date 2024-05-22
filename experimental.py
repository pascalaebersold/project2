import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from eth_mugs_dataset import ETHMugsDataset
from utils import compute_iou

import matplotlib.pyplot as plt
from tqdm import tqdm


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.up_conv4 = self.up_conv(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)
        self.up_conv3 = self.up_conv(512, 256)
        self.decoder3 = self.conv_block(512, 256)
        self.up_conv2 = self.up_conv(256, 128)
        self.decoder2 = self.conv_block(256, 128)
        self.up_conv1 = self.up_conv(128, 64)
        self.decoder1 = self.conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        enc4 = self.encoder4(self.maxpool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.maxpool(enc4))

        # Decoder
        dec4 = self.up_conv4(bottleneck)
        dec4 = self.crop_and_concat(enc4, dec4)
        dec4 = self.decoder4(dec4)

        dec3 = self.up_conv3(dec4)
        dec3 = self.crop_and_concat(enc3, dec3)
        dec3 = self.decoder3(dec3)

        dec2 = self.up_conv2(dec3)
        dec2 = self.crop_and_concat(enc2, dec2)
        dec2 = self.decoder2(dec2)

        dec1 = self.up_conv1(dec2)
        dec1 = self.crop_and_concat(enc1, dec1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)

    def maxpool(self, x):
        return nn.MaxPool2d(kernel_size=2, stride=2)(x)

    def crop_and_concat(self, enc_feature, dec_feature):
        _, _, H, W = dec_feature.size()
        enc_feature = transforms.functional.center_crop(enc_feature, [H, W])
        return torch.cat([enc_feature, dec_feature], dim=1)


def train(ckpt_dir: str, train_data_root: str, val_data_root: str):
    """Train function."""
    log_frequency = 1140000
    val_frequency = 5

    num_epochs = 200
    lr = 0.0001
    train_batch_size = 4
    val_batch_size = 1

    print(f"[INFO]: Number of training epochs: {num_epochs}")
    print(f"[INFO]: Learning rate: {lr}")
    print(f"[INFO]: Training batch size: {train_batch_size}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    train_dataset = ETHMugsDataset(root_dir=train_data_root, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    val_dataset = ETHMugsDataset(root_dir=val_data_root, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, pin_memory=True)

    model = UNet(num_classes=4)
    model.to(device)

    weight = torch.tensor([10.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    print("[INFO]: Starting training...")
    for epoch in range(num_epochs):
        model.train()

        epoch_loss = 0
        for batch_idx, (image, gt_mask) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            image, gt_mask = image.to(device), gt_mask.to(device)

            optimizer.zero_grad()

            output = model(image)
            output = nn.functional.interpolate(output, size=(252, 378), mode='bilinear', align_corners=False)
            #print(output.shape, type)
            #print(gt_mask.shape, type(gt_mask))
            #output = (output>0.25).float()
            #gt_mask = gt_mask.unsqueeze(1).float()
            loss = criterion(torch.mean(output, dim=1), gt_mask)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            #if batch_idx % log_frequency == 0:
                #print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item()}")

        scheduler.step()

        print(f"Epoch {epoch+1}, Average Loss: {epoch_loss / len(train_dataloader)}")
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth"))

        if (epoch + 1) % val_frequency == 0:
            model.eval()
            val_iou = 0.0

            with torch.no_grad():
                for val_image, val_gt_mask in val_dataloader:
                    val_image, val_gt_mask = val_image.to(device), val_gt_mask.to(device)
                    val_output = model(val_image)
                    val_output = nn.functional.interpolate(val_output, size=(252, 378), mode='bilinear', align_corners=False)
                    #val_output = torch.argmax(val_output, dim=1, keepdim=True) # take the maximum value of the 4 Matrixes and put it in a new one to reduce to one dim.
                    # Get the top 2 channels for each pixel
                    top2_vals, top2_indices = torch.topk(val_output, 2, dim=1)
                    
                    # Combine the top 2 channels using their values
                    val_output = (top2_vals[:, 0, :, :] + top2_vals[:, 1, :, :]) / 2.0
                    
                    val_output = torch.sigmoid(val_output.float())  # Apply sigmoid to the output
                    encoder_shades2 = val_output
                    val_output = (val_output > 0.4).float()  # Threshold to obtain binary mask
                    
                    # Ensure consistent binary mask (Forground and Background were inverted)
                    if val_output.mean() > 0.5:  # If more than half of the values are 1
                        val_output = 1 - val_output  # Invert the mask


                    val_iou += compute_iou(val_output.cpu().numpy().squeeze().astype(int), val_gt_mask.cpu().numpy().squeeze().astype(int))

                val_iou = (val_iou / len(val_dataloader)) * 100
                print(f"[INFO]: Validation IoU: {val_iou:.2f}")

                # Visualization
                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                image = val_image.squeeze().permute(1, 2, 0).cpu().numpy()
                mask = val_gt_mask.squeeze().cpu().numpy()
                output = val_output.squeeze().cpu().numpy()
                encoder_shades2 = encoder_shades2.squeeze().cpu().numpy()

                axes[0].imshow(mask, cmap='gray')
                axes[0].axis("off")
                axes[0].set_title("Mask")
                axes[1].imshow(image)
                axes[1].axis("off")
                axes[1].set_title("Image")
                axes[2].imshow(output, cmap='gray')
                axes[2].axis("off")
                axes[2].set_title("Output")
                axes[3].imshow(encoder_shades2, cmap='tab10')
                axes[3].axis("off")
                axes[3].set_title("Shades")
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

    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_dir = os.path.join(args.ckpt_dir, dt_string)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)

    train_data_root = os.path.join(args.data_root, "train_images_378_252")
    print(f"[INFO]: Train data root: {train_data_root}")

    val_data_root = os.path.join(args.data_root, "public_test_images_378_252")
    print(f"[INFO]: Validation data root: {val_data_root}")

    train(ckpt_dir, train_data_root, val_data_root)
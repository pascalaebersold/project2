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


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.encoder5 = self.conv_block(512, 1024)

        self.bottleneck = self.conv_block(1024, 2048)

        self.up_conv5 = self.up_conv(2048, 1024)
        self.decoder5 = self.conv_block(2048, 1024)
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
            DepthwiseSeparableConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.maxpool(enc1))
        enc3 = self.encoder3(self.maxpool(enc2))
        enc4 = self.encoder4(self.maxpool(enc3))
        enc5 = self.encoder5(self.maxpool(enc4))

        # Bottleneck
        bottleneck = self.bottleneck(self.maxpool(enc5))

        # Decoder
        dec5 = self.up_conv5(bottleneck)
        dec5 = self.crop_and_concat(enc5, dec5)
        dec5 = self.decoder5(dec5)

        dec4 = self.up_conv4(dec5)
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


def train(ckpt_dir: str, train_data_root: str, val_data_root: str, checkpoint_path=None):
    """Train function."""
    #log_frequency = 1
    val_frequency = 1

    num_epochs = 200
    lr = 0.0001
    train_batch_size = 4
    val_batch_size = 4

    print(f"[INFO]: Number of training epochs: {num_epochs}")
    print(f"[INFO]: Learning rate: {lr}")
    print(f"[INFO]: Training batch size: {train_batch_size}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    train_dataset = ETHMugsDataset(root_dir=train_data_root, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    val_dataset = ETHMugsDataset(root_dir=val_data_root, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, pin_memory=True)

    model = UNet(num_classes=4)
    model.to(device)

    weight = torch.tensor([5.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

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


        scheduler.step(epoch_loss)

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
                    val_output = nn.functional.interpolate(val_output, size=(252, 378), mode='bilinear', align_corners=False)
                    #val_output = torch.argmax(val_output, dim=1, keepdim=True) # take the maximum value of the 4 Matrixes and put it in a new one to reduce to one dim.
                    # Get the top 2 channels for each pixel
                    o = val_output

                    top2_vals, top2_indices = torch.topk(val_output, 2, dim=1)
                    # Combine the top 2 channels using their values
                    val_output = (top2_vals[:, 0, :, :] + top2_vals[:, 1, :, :]) / 2.0
                    
                    val_output = torch.sigmoid(val_output.float())  # Apply sigmoid to the output
                    encoder_shades2 = val_output
                    val_output = (val_output > 0.75).float()  # Threshold to obtain binary mask
                    
                    # Ensure consistent binary mask (Forground and Background were inverted)
                    if val_output.mean() > 0.5:  # If more than half of the values are 1
                        val_output = 1 - val_output  # Invert the mask


                    val_iou += compute_iou(val_output.cpu().numpy().squeeze().astype(int), val_gt_mask.cpu().numpy().squeeze().astype(int))

                val_iou = (val_iou / len(val_dataloader)) * 100
                print(f"[INFO]: Validation IoU: {val_iou:.2f}")

                # Visualization
                fig, axes = plt.subplots(2, 4, figsize=(12, 3))
                image = val_image.squeeze().cpu().numpy()
                mask = val_gt_mask.squeeze().cpu().numpy()
                output = val_output.squeeze().cpu().numpy()
                o = o.squeeze().cpu().numpy()
                encoder_shades2 = encoder_shades2.squeeze().cpu().numpy()

                axes[0,0].imshow(mask, cmap='gray')
                axes[0,0].axis("off")
                axes[0,0].set_title("Mask")
                axes[0,1].imshow(image, cmap='gray')
                axes[0,1].axis("off")
                axes[0,1].set_title("Image")
                axes[0,2].imshow(output, cmap='gray')
                axes[0,2].axis("off")
                axes[0,2].set_title("Output")
                axes[0,3].imshow(encoder_shades2, cmap='tab10')
                axes[0,3].axis("off")
                axes[0,3].set_title("Shades")

                axes[1,0].imshow(o[0,:,:], cmap='gray')
                axes[1,0].axis("off")
                axes[1,0].set_title("o0")
                axes[1,1].imshow(o[1,:,:], cmap='gray')
                axes[1,1].axis("off")
                axes[1,1].set_title("o1")
                axes[1,2].imshow(o[2,:,:], cmap='gray')
                axes[1,2].axis("off")
                axes[1,2].set_title("o2")
                axes[1,3].imshow(o[3,:,:], cmap='gray')
                axes[1,3].axis("off")
                axes[1,3].set_title("o3")
                #plt.show()
                plt.savefig(os.path.join(ckpt_dir, f"validation_img_epoch_{epoch}" + f"IoU_{val_iou:.2f}" + f".jpg"))
                #input()


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
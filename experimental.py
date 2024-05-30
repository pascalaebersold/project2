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

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0 = self.up(x2_0)
        x2_0 = nn.functional.interpolate(x2_0, size=[x2_0.shape[2],x1_0.shape[3]])
        x1_1 = self.conv1_1(torch.cat([x1_0, x2_0], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0 = self.up(x3_0)
        x3_0 = nn.functional.interpolate(x3_0, size=[x3_0.shape[2],x2_0.shape[3]])
        x2_1 = self.conv2_1(torch.cat([x2_0, x3_0], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, x2_1], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0 = self.up(x4_0)
        x4_0 = nn.functional.interpolate(x4_0, size=[x4_0.shape[2],x3_0.shape[3]])
        x3_1 = self.conv3_1(torch.cat([x3_0, x4_0], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, x3_1], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, x2_2], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

def train(ckpt_dir: str, train_data_root: str, val_data_root: str, checkpoint_path: str):
    """Train function."""
    #log_frequency = 1
    val_frequency = 1

    num_epochs = 200
    lr = 0.0001
    train_batch_size = 6
    val_batch_size = 2

    print(f"[INFO]: Number of training epochs: {num_epochs}")
    print(f"[INFO]: Learning rate: {lr}")
    print(f"[INFO]: Training batch size: {train_batch_size}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    train_dataset = ETHMugsDataset(root_dir=train_data_root, mode='train')
    # train_dataset.transform = transforms.Compose([
    #     #transforms.RandomRotation(15),
    #     transforms.RandomHorizontalFlip(0.2),
    #     transforms.RandomVerticalFlip(0.2),
    #     #transforms.ColorJitter(brightness=0.2, contrast=0.2),
    #     transforms.ToTensor()
    # ])
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    val_dataset = ETHMugsDataset(root_dir=val_data_root, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True, pin_memory=True)

    model = UNet(num_classes=1)
    model.to(device)

    weight = torch.tensor([5.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

    start_epoch = 0
    if checkpoint_path!=None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"[INFO]: Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch})")

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
                    
                    val_output = torch.sigmoid(val_output.float())  # Apply sigmoid to the output

                    val_output = (val_output > 0.85).float()  # Threshold to obtain binary mask


                    val_iou += compute_iou(val_output.cpu().numpy().squeeze().astype(int), val_gt_mask.cpu().numpy().squeeze().astype(int))

                val_iou = (val_iou / len(val_dataloader)) * 100
                print(f"[INFO]: Validation IoU: {val_iou:.2f}")

                # Visualization
                fig, axes = plt.subplots(1, 3, figsize=(12, 3))
                image = val_image.squeeze().cpu().numpy()
                mask = val_gt_mask.squeeze().cpu().numpy()
                output = val_output.squeeze().cpu().numpy()

                axes[0].imshow(mask, cmap='gray')
                axes[0].axis("off")
                axes[0].set_title("Mask")
                axes[1].imshow(image)
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
"""Code template for training a model on the ETHMugs dataset."""

import argparse
import os
from datetime import datetime

import torch

from eth_mugs_dataset import ETHMugsDataset
from utils import IMAGE_SIZE, compute_iou


def build_model():  # TODO: Add your model definition here
    """Build the model."""


def train(
    ckpt_dir: str,
    train_data_root: str,
    val_data_root: str,
):
    """Train function."""
    # Logging and validation settings
    log_frequency = 10
    val_batch_size = 1
    val_frequency = 1

    # TODO: Set your own values for the hyperparameters
    num_epochs = 50
    # lr = 1e-4
    # train_batch_size = 8
    # val_batch_size = 1
    # ...

    print(f"[INFO]: Number of training epochs: {num_epochs}")
    # print(f"[INFO]: Image scale: {image_scale}")
    # print(f"[INFO]: Learning rate: {lr}")
    # print(f"[INFO]: Training batch size: {train_batch_size}")

    # Choose Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # TODO: Define your Dataset and DataLoader
    # ETHMugsDataset
    # Data loaders
    # train_dataset = ...
    # train_dataloader = ...
    # val_dataset = ...
    # val_dataloader = ...

    # TODO: Define you own model
    # model = build_model(...)
    # model.to(device)

    # TODO: Define Loss function
    # criterion = ...

    # TODO: Define Optimizer
    # optimizer = ...

    # TODO: Define Learning rate scheduler if needed
    # lr_scheduler = ...

    # TODO: Write the training loop!
    print("[INFO]: Starting training...")
    for epoch in range(num_epochs):
        model.train()

        for image, gt_mask in train_dataloader:
            image = image.to(device)
            gt_mask = gt_mask.to(device)

            optimizer.zero_grad()

            # Forward pass
            # output = model(image ...)

            # loss = criterion(output ...)

            # Backward pass
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

        # Save model
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last_epoch.pth"))

        if epoch % val_frequency == 0:
            model.eval()

            val_iou = 0.0
            with torch.no_grad():
                for val_image, val_gt_mask in val_dataloader:
                    val_image = val_image.to(device)
                    val_gt_mask = val_gt_mask.to(device)

                    # Forward pass
                    # output = model(image ...)

                    # val_iou += compute_iou(...)

                val_iou /= len(val_dataloader)

                val_iou *= 100

                print(f"[INFO]: Validation IoU: {val_iou.item():.2f}")


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

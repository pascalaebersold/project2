"""Test a pre-trained model."""

import argparse
import os
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

from eth_mugs_dataset import ETHMugsDataset
from train import build_model
from utils import IMAGE_SIZE, load_mask, compute_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SML Project 2.")
    parser.add_argument(
        "-d",
        "--data_root",
        default="./datasets",
        type=str,
        help="Path to the datasets folder.",
    )
    parser.add_argument(
        "-s",
        "--split",
        choices=["public_test", "private_test"],
        default="public_test",
        help="Choose the data split. If using public test, then your model will also be evaluated.",
    )
    parser.add_argument("--ckpt", type=str, help="model checkpoint.")
    args = parser.parse_args()

    # Set data root
    if args.split == "public_test":
        test_data_root = os.path.join(args.data_root, "public_test_images_378_252")
        out_dir = os.path.join("public_test", "prediction")
    else:
        test_data_root = os.path.join(args.data_root, "private_test_images_378_252")
        out_dir = os.path.join("private_test", "prediction")
    print(f"[INFO]: Test data root: {test_data_root}")

    # Set output directory
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO]: Saving the predicted segmentation masks to {out_dir}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Build Model
    # model = build_model(...)

    # Load pre-trained model
    print(f"[INFO]: Loading the pre-trained model: {args.ckpt}")
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))

    model.to(device)
    model.eval()

    # Create an instance of the custom dataset
    test_dataset = ETHMugsDataset(root_dir=test_data_root, mode="test")

    # Create dataloaders
    test_batch_size = 1
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False
    )

    with torch.no_grad():
        for i, test_image in enumerate(tqdm(test_dataloader)):
            test_image = test_image.to(device)

            # TODO: Forward pass
            # test_output = model(test_image)

            # Save the predicted mask
            resized_pred_mask = Image.fromarray(test_output.cpu().numpy())
            resized_pred_mask.save(
                os.path.join(out_dir, str(i).zfill(4) + "_mask.png")
            )

    # Run evaluation if using public test split
    if args.split == "public_test":
        gt_dir = os.path.join(args.data_root, "public_test_images_378_252", "masks")

        # Load GT and prediction mask filenames
        gt_mask_filenames = [
            el for el in sorted(os.listdir(gt_dir)) if el.endswith("_mask.png")
        ]

        pred_mask_filenames = [
            el for el in sorted(os.listdir(out_dir)) if el.endswith("_mask.png")
        ]
        assert (
            gt_mask_filenames == pred_mask_filenames
        ), "predictions must have been saved with the same file names as the GT files"
        num_samples_to_evaluate = len(gt_mask_filenames)

        test_iou_sum = 0.0
        for idx in tqdm(range(num_samples_to_evaluate)):
            gt_mask_path = os.path.join(gt_dir, gt_mask_filenames[idx])
            pred_mask_path = os.path.join(out_dir, pred_mask_filenames[idx])

            # All values are 0 or 1, dtype: int
            gt_mask = load_mask(gt_mask_path)
            # All values are 0 or 1, dtype: int
            pred_mask = load_mask(pred_mask_path)

            iou = compute_iou(pred_mask, gt_mask)
            test_iou_sum += iou

        average_test_iou = test_iou_sum / num_samples_to_evaluate
        print(f"[INFO]: IoU: {average_test_iou}")

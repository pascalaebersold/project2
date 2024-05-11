"""Utility functions."""

import numpy as np
import torch

from PIL import Image
from torch import Tensor


IMAGE_SIZE = (252, 378)


def dice_coeff(
    pred: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    """Average of Dice coefficient for all batches, or for a single mask"""
    assert pred.size() == target.size()
    assert pred.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if pred.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (pred * target).sum(dim=sum_dim)
    sets_sum = pred.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def load_mask(mask_path):
    """Loads the segmentation mask from the specified path.

    Inputs:
        mask_path (str): the path from which the segmentation mask will be read.
        It should have the format "/PATH/TO/LOAD/DIR/XXXX_mask.png".

    Outputs:
        mask (np.array): segmentation mask as a numpy array.
    """
    mask = np.asarray(Image.open(mask_path)).astype(int)
    if mask.max() > 1:
        mask = mask // 255
    return mask


def compute_iou(pred_mask, gt_mask, eps=1e-6):
    """Computes the IoU between two numpy arrays: pred_mask and gt_mask.

    Inputs:
        pred_mask (np.array): dtype:int, shape:(image_height, image_width), values are 0 or 1.
        gt_mask (np.array): dtype:int, shape:(image_height, image_width), values are 0 or 1.
        eps (float): epsilon to smooth the division in order to avoid 0/0.

    Outputs:
        iou_score (float)
    """
    intersection = (
        (pred_mask & gt_mask).astype(float).sum()
    )  # will be zero if gt=0 or pred=0
    union = (pred_mask | gt_mask).astype(float).sum()  # will be zero if both are 0
    iou = (intersection + eps) / (
        union + eps
    )  # we smooth our division by epsilon to avoid 0/0
    iou_score = iou.mean()
    return iou_score

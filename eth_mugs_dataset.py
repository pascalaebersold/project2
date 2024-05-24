"""ETH Mugs Dataset."""

import os
from PIL import Image
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from utils import IMAGE_SIZE, load_mask


# This is only an example - you DO NOT have to use it exactly in this form!
class ETHMugsDataset(Dataset):
    """Torch dataset template shared as an example."""

    def __init__(self, root_dir, mode="train"):
        """This dataset class loads the ETH Mugs dataset.

        It will return the resized image according to the scale and mask tensors
        in the original resolution.

        Args:
            root_dir (str): Path to the root directory of the dataset.
            mode (str): Mode of the dataset. It can be "train", "val" or "test"
        """
        self.mode = mode
        self.root_dir = root_dir

        # TODO: get image and mask paths
        self.rgb_dir = os.path.join(self.root_dir, "rgb")
        self.mask_dir = os.path.join(self.root_dir, "masks")
        self.image_paths = sorted(os.listdir(self.rgb_dir))

        if mode == "train":
            self.image_paths = [img for img in self.image_paths if img.endswith(".jpg")]
        # In val mode, only include the validation set
        elif mode == "val":
            self.image_paths = [img for img in self.image_paths if img.endswith(".jpg")]

        # TODO: set image transforms - these transforms will be applied to pre-process the data before passing it through the model transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])

        print("[INFO]: Dataset mode:", mode)
        print(
            "[INFO]: Number of images in the ETHMugDataset: {}".format(len(self.image_paths))
        )

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Get an item from the dataset."""
        # TODO: load image and gt mask (unless when in test mode), apply transforms if necessary
        # Load image
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.rgb_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        # Apply transformations
        image = self.transform(image)

        # Load mask (if not in test mode)
        if self.mode != "test":
            mask_name = img_name.replace("_rgb.jpg", "_mask.png")  # Assuming mask file names are similar to image file names
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = load_mask(mask_path)  # Implement load_mask function according to your needs
            mask = torch.as_tensor(mask, dtype=torch.float)
            mask = (mask > 0.3).float() # Mask IS NOT BOOL WHAT THE FUCK round gray scales to 1.0
            return image, mask
        else:
            return image

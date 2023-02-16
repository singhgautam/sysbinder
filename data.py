import glob
import json

import torch

from torchvision import transforms
from torch.utils.data import Dataset

import numpy as np

from PIL import Image

from pathlib import Path


class GlobDataset(Dataset):
    def __init__(self, root, phase, img_size):
        self.root = root
        self.img_size = img_size
        self.total_imgs = sorted(glob.glob(root))

        if phase == 'train':
            self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
        elif phase == 'val':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
        elif phase == 'test':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
        else:
            pass

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = self.total_imgs[idx]
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        tensor_image = self.transform(image)
        return tensor_image


class CLEVREasyWithAnnotations(Dataset):
    def __init__(self, root, phase, img_size, max_num_objs=3, num_attributes=3):
        self.root = root
        self.img_size = img_size
        self.total_imgs = sorted(glob.glob(root))
        self.max_num_objs = max_num_objs
        self.num_attributes = num_attributes

        if phase == 'train':
            self.total_imgs = self.total_imgs[:int(len(self.total_imgs) * 0.7)]
        elif phase == 'val':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.7):int(len(self.total_imgs) * 0.85)]
        elif phase == 'test':
            self.total_imgs = self.total_imgs[int(len(self.total_imgs) * 0.85):]
        else:
            pass

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        # paths
        img_loc = self.total_imgs[idx]
        p = Path(img_loc)
        mask_loc = p.parent / (p.stem + "_mask.png")
        json_loc = p.parent.parent / "scenes" / (p.stem + ".json")

        # codes
        color_codes = {
            "gray": 0,
            "red": 1,
            "blue": 2,
            "green": 3,
            "brown": 4,
            "purple": 5,
            "cyan": 6,
            "yellow": 7,
        }
        shape_codes = {
            "cube": 0,
            "sphere": 1,
            "cylinder": 2,
        }

        # mask colors
        object_mask_colors = torch.Tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])  # N, C
        max_num_objs = object_mask_colors.shape[0]

        # image
        image = Image.open(img_loc).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        image = self.transform(image)  # C, H, W

        eps = 0.001

        # masks
        mask_image = Image.open(mask_loc).convert("RGB")
        mask_image = mask_image.resize((self.img_size, self.img_size))
        mask_image = self.transform(mask_image)  # C, H, W
        masks = (mask_image[None, :, :, :] < object_mask_colors[:, :, None, None] + eps) & \
                (mask_image[None, :, :, :] > object_mask_colors[:, :, None, None] - eps)
        masks = masks.float().prod(1, keepdim=True)  # N, 1, H, W

        # annotations
        annotations = torch.zeros(max_num_objs, self.num_attributes)  # N, G

        with open(json_loc) as f:
            data = json.load(f)
            object_list = data["objects"]
            for i, object in enumerate(object_list):
                # shape
                annotations[i, 0] = shape_codes[object["shape"]]

                # color
                annotations[i, 1] = color_codes[object["color"]]

                # position
                K = 3
                annotations[i, 2] = np.digitize(object['3d_coords'][0], np.linspace(-4 - eps, 4 + eps, K + 1)) - 1
                annotations[i, 2] = annotations[i, 2] * K + np.digitize(object['3d_coords'][1], np.linspace(-3 - eps, 4 + eps, K + 1)) - 1

        return (
            image,  # C, H, W
            masks,  # N, 1, H, W
            annotations,  # N, G
        )

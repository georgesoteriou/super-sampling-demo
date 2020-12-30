import random
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from utils import crop_random


class DIV2K(Dataset):
    def __init__(self, folder_path, scaling_factor=4, mode="RGB"):
        super(DIV2K, self).__init__()
        self.folder_path = folder_path
        self.scaling_factor = scaling_factor
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])

    def __getitem__(self, idx):
        idx = idx+1
        path = f"{self.folder_path}/HR/{format(idx, '04d')}.png"
        hr_img = Image.open(path).convert("RGB")
        hr_img = crop_random(hr_img, 96)
        lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)),
                               Image.BICUBIC)
        lr_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)
        if self.mode == "y":
            hr_img, _, _ = hr_img.split()
            lr_img, _, _ = lr_img.split()
        lr_tensor = self.transform(lr_img)
        hr_tensor = self.transform(hr_img)
        return lr_tensor, hr_tensor

    def __len__(self):
        return len(os.listdir(f'{self.folder_path}/HR'))

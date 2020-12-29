import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class DIV2K(Dataset):
    def __init__(self, folder_path, lr_type):
        super(DIV2K, self).__init__()

        self.folder_path = folder_path
        if lr_type not in [2, 3, 4]:
            print("Invalid lr_type. Defaulting to x2")
            self.lr_type = 2
        else:
            self.lr_type = lr_type

    def __getitem__(self, idx):
        idx = idx+1
        lr_path = f"{self.folder_path}/X{self.lr_type}/{format(idx, '04d')}x{self.lr_type}.png"
        hr_path = f"{self.folder_path}/HR/{format(idx, '04d')}.png"
        lr_pic = Image.open(lr_path).convert("YCbCr").split()[0]
        hr_pic = Image.open(hr_path).convert("YCbCr").split()[0]
        # Transform LR picture to size of HR picture using BICUBIC method
        lr_transform = transforms.Compose([
            transforms.Resize(
                (hr_pic.size[1], hr_pic.size[0]), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])
        hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, ], std=[0.5, ])
        ])
        lr_tensor = lr_transform(lr_pic)
        hr_tensor = hr_transform(hr_pic)
        return lr_tensor, hr_tensor

    def __len__(self):
        return len(os.listdir(f'{self.folder_path}/HR'))

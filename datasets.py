import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class DIV2K(Dataset):
    def __init__(self, folder_path, lr_type):
        super(DIV2K, self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.folder_path = folder_path
        if lr_type not in [2, 3, 4]:
            print("Invalid lr_type. Defaulting to x2")
            self.lr_type = 2
        else:
            self.lr_type = lr_type

    def __getitem__(self, idx):
        idx = idx+1
        hr = self.transform(Image.open(
            f"{self.folder_path}/HR/{format(idx, '04d')}.png").convert("RGB"))
        lr = self.transform(Image.open(
            f"{self.folder_path}/X{self.lr_type}/{format(idx, '04d')}x{self.lr_type}.png").convert("RGB"))
        return hr, lr

    def __len__(self):
        return len(os.listdir(f'{self.folder_path}/HR'))

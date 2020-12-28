import numpy as np
from torch.utils.data import Dataset
from PIL import Image


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
        hr = Image.open(f"{self.folder_path}/HR/{format(idx, '04d')}.png")
        lr = Image.open(
            f"{self.folder_path}/X{self.lr_type}/{format(idx, '04d')}x{self.lr_type}.png")
        return np.expand_dims(lr[:, :] / 255., 0), np.expand_dims(hr[:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

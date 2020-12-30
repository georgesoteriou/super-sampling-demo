import torch
import random


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # get sequence lengths
    lengths = torch.tensor([t.shape[0] for t in batch]).to(device)
    # padd
    batch = [torch.Tensor(t).to(device) for t in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    # compute mask
    mask = (batch != 0).to(device)
    return batch, lengths, mask


def crop_random(img, crop_size):
    left = random.randint(1, img.width - crop_size)
    top = random.randint(1, img.height - crop_size)
    right = left + crop_size
    bottom = top + crop_size
    return img.crop((left, top, right, bottom))


def crop_offset(img, crop_size, left, top):
    right = left + crop_size
    bottom = top + crop_size
    return img.crop((left, top, right, bottom))


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

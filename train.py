import numpy as np
import torch
from models import SRCNN
from dataset import TrainDataset, EvalDataset


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)

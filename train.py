import math
import torch
from torch import nn, optim
from tqdm import tqdm
from models import SRCNN
from datasets import DIV2K
from torch.cuda import amp

batch_size = 1
epochs = 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SRCNN().to(device)
train_dataset = DIV2K(folder_path=f"data/DIV2K/train", lr_type=2)
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=3)

criterion = nn.MSELoss()
optimizer = optim.Adam([
    {'params': model.conv1.parameters()},
    {'params': model.conv2.parameters()},
    {'params': model.conv3.parameters(), 'lr': 0.001}
], lr=0.001)
scaler = amp.GradScaler()

for epoch in range(epochs):  # loop over the dataset multiple times
    progress_bar = tqdm(enumerate(train_dataloader),
                        total=len(train_dataloader))
    for iteration, (input, target) in progress_bar:
        # Set model gradients to zero
        optimizer.zero_grad()

        lr = input.to(device)
        hr = target.to(device)

        with amp.autocast():
            sr = model(lr)
            loss = criterion(sr, hr)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        # Updates the scale for next iteration.
        scaler.update()

        psnr_value = 10 * math.log10((hr.max() ** 2) / loss)

        progress_bar.set_description(f"[{epoch + 1}/{epochs}]"
                                     f"[{iteration + 1}/{len(train_dataloader)}] "
                                     f"MSE: {loss.item():.4f} "
                                     f"PSNR: {psnr_value:.2f}dB")

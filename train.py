import math
import torch
from torch import nn, optim
from tqdm import tqdm
from models import SRCNN
from datasets import DIV2K
from torch.cuda import amp


batch_size = 16

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SRCNN(num_channels=3).to(device)
train_dataset = DIV2K(folder_path=f"data/DIV2K/train", scaling_factor=4)
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=4)

criterion = nn.MSELoss()
optimizer = optim.Adam([
    {"params": model.features.parameters()},
    {"params": model.map.parameters()},
    {"params": model.reconstruction.parameters(), "lr": 0.00001}
], lr=0.00001)
scaler = amp.GradScaler()

epochs = int(1e8 // len(train_dataloader))
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

        if (len(train_dataloader) * epoch + iteration + 1) % 1000 == 0:
            torch.save({"epoch": epoch + 1,
                        "optimizer": optimizer.state_dict(),
                        "state_dict": model.state_dict()
                        }, f"./weights/SRCNN_checkpoint.pth")

torch.save(model.state_dict(), f"./weights/SRCNN.pth")
print(f"[*] Training model done! Saving model weight to `./weights/SRCNN.pth`.")

from torch import nn
from torch.cuda import amp


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        # Patch extraction and representation.
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True)
        )

        # Non-linear mapping.
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True)
        )

        # Reconstruction image.
        self.reconstruction = nn.Conv2d(
            32, num_channels, kernel_size=5, padding=5 // 2)

    @amp.autocast()
    def forward(self, input):
        out = self.features(input)
        out = self.map(out)
        out = self.reconstruction(out)
        return out

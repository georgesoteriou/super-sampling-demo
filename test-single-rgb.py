import torch
from models import SRCNN
from PIL import Image
from torchvision import transforms
from utils import crop_offset, UnNormalize
import sys

# if len(sys.argv) != 3:
#     print("Not enough parameters. Correct way to use:")
#     print("python3 test-single.py [path/to/weights.pth] [path/to/picture.png]")
#     exit()

scaling_factor = 4
crop_size = 94

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SRCNN(num_channels=3).to(device)
# model.load_state_dict(torch.load(sys.argv[1], map_location=device))
model.load_state_dict(torch.load(
    "weights/SRCNN_checkpoint.pth", map_location=device)["state_dict"])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, ], std=[0.5, ])
])

img = Image.open("data/DIV2K/valid/0867.png").convert("RGB")
lr_img = img.resize((int(img.width / scaling_factor),
                     int(img.height / scaling_factor)), Image.BICUBIC)
lr_img = lr_img.resize((img.width, img.height), Image.BICUBIC)
final_sr_img = Image.new('RGB', (img.width, img.height), (250, 250, 250))

for left in range(0, img.width, crop_size):
    for top in range(0, img.height, crop_size):
        lr_img_crop = crop_offset(img, crop_size, left, top)
        lr_tensor = transform(lr_img_crop)
        inputs = lr_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(inputs).squeeze(0)
        out = out.cpu().detach()
        out_img = transforms.ToPILImage()(out)
        final_sr_img.paste(out_img, (left, top))

all_images = Image.new('RGB', (img.width*3, img.height), (250, 250, 250))
all_images.paste(lr_img, (0, 0))
all_images.paste(final_sr_img, (img.width, 0))
all_images.paste(img, (img.width*2, 0))
all_images.show()

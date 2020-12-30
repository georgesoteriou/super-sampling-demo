import numpy as np
import torch
from models import SRCNN
from PIL import Image
from torchvision import transforms
from utils import crop_offset
import sys


# if len(sys.argv) != 3:
#     print("Not enough parameters. Correct way to use:")
#     print("python3 test-single.py path/to/weights.pth path/to/picture.png")
#     exit()
scaling_factor = 3
crop_size = 94

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SRCNN().to(device)
model.load_state_dict(torch.load(
    "weights/SRCNN_y_checkpoint.pth", map_location=device)["state_dict"])

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
        lr_img_crop = crop_offset(lr_img, crop_size, left, top)
        y, cb, cr = lr_img.convert("YCbCr").split()
        inputs = transform(y)
        inputs = inputs.unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inputs).squeeze(0)
        out = out.cpu()
        out_image_y = out[0].detach().numpy()
        out_image_y *= 255.0
        out_image_y = out_image_y.clip(0, 255)
        out_image_y = Image.fromarray(np.uint8(out_image_y[0]), mode="L")
        out_img_cb = cb.resize(out_image_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_image_y.size, Image.BICUBIC)
        out_img = Image.merge(
            "YCbCr", [out_image_y, out_img_cb, out_img_cr]).convert("RGB")
        final_sr_img.paste(out_img, (left, top))

all_images = Image.new('RGB', (img.width*3, img.height), (250, 250, 250))
all_images.paste(lr_img, (0, 0))
all_images.paste(final_sr_img, (img.width, 0))
all_images.paste(img, (img.width*2, 0))
all_images.show()

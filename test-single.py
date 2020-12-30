import numpy as np
import torch
from models import SRCNN
from PIL import Image
from torchvision import transforms
import sys

if len(sys.argv) != 3:
    print("Not enough parameters. Correct way to use:")
    print("python3 test-single.py path/to/weights.pth path/to/picture.png")
    exit()

batch_size = 1
epochs = 2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = SRCNN().to(device)
model.load_state_dict(torch.load(sys.argv[1], map_location=device))

# Open image
image = Image.open(sys.argv[2]).convert("YCbCr")
y, cb, cr = image.split()

preprocess = transforms.ToTensor()
inputs = preprocess(y).view(1, -1, y.size[1], y.size[0])

inputs = inputs.to(device)

with torch.no_grad():
    out = model(inputs)
out = out.cpu()
out_image_y = out[0].detach().numpy()
out_image_y *= 255.0
out_image_y = out_image_y.clip(0, 255)
out_image_y = Image.fromarray(np.uint8(out_image_y[0]), mode="L")

out_img_cb = cb.resize(out_image_y.size, Image.BICUBIC)
out_img_cr = cr.resize(out_image_y.size, Image.BICUBIC)
out_img = Image.merge(
    "YCbCr", [out_image_y, out_img_cb, out_img_cr]).convert("RGB")
out_img.save(f"srcnn.png")

import argparse
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms as transforms
import numpy as np
from train import GeneratorResNet
from torchvision.utils import save_image, make_grid

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

lr_transform = transforms.Compose(
            [
                transforms.Resize((256 // 4, 256 // 4), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

hr_transform = transforms.Compose(
            [
                transforms.Resize((256, 256), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='generator.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

TEST_MODE = True if opt.test_mode == 'GPU' else False
IMAGE_NAME = opt.image_name
MODEL_NAME = opt.model_name

model = GeneratorResNet()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load('saved_models/' + MODEL_NAME))
else:
    model.load_state_dict(torch.load('saved_models/' + MODEL_NAME, map_location=lambda storage, loc: storage))

image = Image.open(IMAGE_NAME)
imgs_hr = hr_transform(image)
image = lr_transform(image).unsqueeze(0)
imgs_lr = torch.nn.functional.interpolate(image, scale_factor=4)

if TEST_MODE:
    image = image.cuda()
model.eval()
start = time.time()
out = model(image)
elapsed = (time.time() - start)
print('cost' + str(elapsed) + 's')
imgs_lr = make_grid(imgs_lr[0], nrow=1, normalize=True)
gen_hr = make_grid(out[0], nrow=1, normalize=True)
imgs_hr = make_grid(imgs_hr, nrow=1, normalize=True)
img_grid = torch.cat((imgs_lr.cuda(), gen_hr,imgs_hr.cuda()), -1)
save_image(img_grid, f"/root/{IMAGE_NAME[:-4]}_com.png", normalize=False)

"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import yaml
import time
import numpy as np
import tifffile
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from torch import Tensor

from data import ImageLabelFilelist, default_loader, tiff_loader
import torchvision.transforms.functional as F
from pathlib import Path


class tiff_normalize(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, tensor: Tensor) -> Tensor:
        old_tensor = tensor.clone()
        means = tensor.mean(axis=(1,2))
        stds = tensor.std(axis=(1,2))
        stds = torch.Tensor([s if s > 0 else 1 for s in stds])
        tensor.sub_(means.view(-1,1,1)).div_(stds.view(-1,1,1))
        if np.isnan(tensor).sum() > 1:
            print('nonnn')
            torch.save(old_tensor, 'old_tensor.pth')
        return tensor


def main():
    transform_list = [transforms.ToTensor(),
                        tiff_normalize()]
    transform_list = [transforms.Resize(128)] + transform_list \
        if True is not None else transform_list
    transform = transforms.Compose(transform_list)
    for class_num in range(5):
        for path in Path(f'/home/ubuntu/coco_scripts/data/class_{class_num}/').glob('*.tiff'):
            img = tifffile.imread(path)
            img = Image.fromarray(img.transpose(1,2,0))
            img = transform(img)
            if np.isnan(img).sum() > 0:
                print(path)

if __name__ == '__main__':
    main()

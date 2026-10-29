import glob
import os
import numpy as np
import torch
from PIL import Image


class Crops(torch.utils.data.Dataset):
    """Crops written by satdistill.prepare: images as float tensors in [0, 1] (C, H, W) and label maps (H, W) as
    int64."""

    def __init__(self, folder, labels=False, max_items=0):
        self.images = sorted(glob.glob(os.path.join(folder, "images", "*.jpg")))
        if not self.images:
            raise FileNotFoundError("no crops in %s/images" % folder)
        if max_items:
            self.images = self.images[:max_items]
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(Image.open(self.images[i]).convert("RGB"))).permute(2, 0, 1).float() / 255.0
        if not self.labels:
            return x
        path = self.images[i].replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)[:-4] + ".png"
        return x, torch.from_numpy(np.array(Image.open(path))).long()

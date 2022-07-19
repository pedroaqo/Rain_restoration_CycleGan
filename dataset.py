from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class RainyCleanDataset(Dataset):
    def __init__(self, root_clean, root_rain, transform=None):
        self.root_clean = root_clean
        self.root_rain = root_rain
        self.transform = transform

        self.clean_images = os.listdir(root_clean)
        self.rainy_images = os.listdir(root_rain)
        self.length_dataset = max(len(self.clean_images), len(self.rainy_images)) # 1000, 1500
        self.clean_len = len(self.clean_images)
        self.rainy_len = len(self.rainy_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        clean_img = self.clean_images[index % self.clean_len]
        rainy_img = self.rainy_images[index % self.rainy_len]

        clean_path = os.path.join(self.root_clean, clean_img)
        rainy_path = os.path.join(self.root_rain, rainy_img)

        clean_img = np.array(Image.open(clean_path).convert("RGB"))
        rainy_img = np.array(Image.open(rainy_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=clean_img, image0=rainy_img)
            clean_img = augmentations["image"]
            rainy_img = augmentations["image0"]

        return clean_img, rainy_img

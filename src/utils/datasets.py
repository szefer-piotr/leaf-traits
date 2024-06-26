from torch.utils.data import Dataset
from typing import List, Dict
import imageio.v3 as imageio
import albumentations as A
import numpy as np

class LTDataset(Dataset):
    def __init__(
        self, 
        image_bytes: List, 
        targets: np.array, 
        features: np.array, 
        transforms: A.Compose = None       
    ):
        self.image_bytes = image_bytes
        self.targets = targets
        self.features = features
        self.transforms = transforms

    def __len__(self):
        return len(self.image_bytes)
    
    def __getitem__(self, index) -> Dict:
        predictor = {
            'image': self.transforms(
                image=imageio.imread(self.image_bytes[index]),
            )['image'],
            'feature': self.features[index],
        }
        target = self.targets

        return predictor, target
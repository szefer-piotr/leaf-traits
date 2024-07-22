from torch.utils.data import Dataset, DataLoader
import imageio.v3 as iio
from pathlib import Path
import numpy as np

class LTDataset(Dataset):
    def __init__(
        self, 
        file_path: np.ndarray, 
        targets: np.ndarray, 
        features: np.ndarray, 
        transforms: A.Compose = None       
	):
        self.file_path = file_path
        self.targets = targets
        self.features = features
        self.transforms = transforms

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        X_sample = {
            'image': self.transforms(
                    image=iio.imread(Path("..", self.file_path[index])),
                )['image'],
            'feature': self.features[index],
        }
        y_sample = self.targets[index]
            
        return X_sample, y_sample
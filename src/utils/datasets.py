from torch.utils.data import Dataset, DataLoader
import imageio.v3 as iio
from pathlib import Path
import numpy as np

class LTDataset(Dataset):
    """Custom dataset class that for each datapoint returns image, 
    corresponding `n` features form the tabular data, and six target traits.

    Returns:
        The __getitem__ metod returns:
        X_sample - dictionary containig the transformed image (what format?), and features from the tabular data.
        y_sample - six target traits.
    
    """
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
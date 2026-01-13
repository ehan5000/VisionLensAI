import numpy as np
import torch
from torch.utils.data import Dataset

class VisionDataset(Dataset):
    """
    Loads vision_data.npz and provides normalized features for training.

    Normalization is done using dataset mean/std to stabilize training.
    """
    def __init__(self, path: str = "vision_data.npz"):
        data = np.load(path)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.float32)

        self.x_mean = X.mean(axis=0, keepdims=True)
        self.x_std = X.std(axis=0, keepdims=True) + 1e-6
        Xn = (X - self.x_mean) / self.x_std

        self.X = torch.from_numpy(Xn)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

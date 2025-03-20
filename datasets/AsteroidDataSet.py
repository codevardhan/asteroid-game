import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class AsteroidDataSet(Dataset):
    def __init__(self,data_file):
        self.data = np.load(data_file, allow_pickle=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, action = self.data[idx]  # Each entry is (state, action)
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)  # Assuming discrete actions
        return state, action
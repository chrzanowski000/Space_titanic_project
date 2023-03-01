#Dataset class
from torch.utils.data import Dataset # Dataset module
import torch
class Dataset_maker(Dataset):
    def __init__(self, data ,transform=False):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index][0], requires_grad=True, dtype=torch.float64)
        y_label = torch.tensor(self.data[index][1], requires_grad=True, dtype=torch.float64)

        if self.transform:
            x = self.transform(x)

        return (x, y_label)
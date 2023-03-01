#Dataset class
from torch.utils.data import Dataset # Dataset module
class BloodCell_train(Dataset):
    def __init__(self, data ,transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        y_label = self.data[index][1]

        if self.transform:
            x = self.transform(x)

        return (x, y_label)
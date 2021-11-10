import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class TabularToTextDataset(Dataset):

    def __init__(self, data, label):
        self.data = data 
        self.label = torch.tensor(label.to_numpy(), dtype=torch.long)
        self.keys = list(self.data.keys())

    def __getitem__(self, idx):
        instance = {key: self.data[key][idx] for key in self.keys}
        label = self.label[idx]
        return instance, label

    def __len__(self):
        return len(self.label)

    
class TabularToTextDM(pl.LightningDataModule):
  
    def __init__(self, train_encodings, y_train, test_encodings, y_test):
        super().__init__()
        self.train_encodings = train_encodings
        self.y_train = y_train 
        self.test_encodings = test_encodings
        self.y_test = y_test

    def train_dataloader(self):
        train_dataset = TabularToTextDataset(self.train_encodings, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=12)
        return train_loader
  
    def test_dataloader(self):
        test_dataset = TabularToTextDataset(self.test_encodings, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=12)
        return test_loader

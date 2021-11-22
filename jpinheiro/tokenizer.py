import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer


# This model is a fine-tune checkpoint of DistilBERT-base-uncased, fine-tuned on SST-2.
MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'


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
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train 
        self.X_test = X_test
        self.y_test = y_test
        self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME) # define a tokenizer object

    def train_dataloader(self):
        train_encodings = self.tokenize_train()
        train_dataset = TabularToTextDataset(train_encodings, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=12)
        return train_loader

    def test_dataloader(self):
        test_encodings = self.tokenize_test()
        test_dataset = TabularToTextDataset(test_encodings, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=12)
        return test_loader

    def tokenize_train(self):
        train_encodings = self.tokenizer(list(self.X_train.values), truncation=True, padding=True)
        return train_encodings

    def tokenize_test(self):
        test_encodings = self.tokenizer(list(self.X_test.values), truncation=True, padding=True)
        return test_encodings

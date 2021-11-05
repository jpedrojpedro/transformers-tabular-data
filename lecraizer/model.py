import torch.nn as nn

import pytorch_lightning as pl
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# import tensorflow as tf
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from transformers import DistilBertTokenizer, DistilBertForTokenClassification

pd.set_option('display.max_colwidth', None)
MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'  # This model is a fine-tune checkpoint of DistilBERT-base-uncased, fine-tuned on SST-2.
# MODEL_NAME = 'distilbert-base-multilingual-cased'
BATCH_SIZE = 60
N_EPOCHS = 3  # we can put more, because evaluation of the model shows big difference in loss with accuracy 1.0


def join_columns(row):
    final = []
    for col in df.columns[:4]:
        aux = []
        aux.append(col)
        aux.append(str(row[col]))
        final.append(' '.join(aux))
    return ', '.join(final)


iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

df['text'] = df.apply(join_columns, axis=1)
df_text = df[['text', 'target']].copy()

print(df_text['text'][0])

text = df_text['text']
labels = df_text['target']

X_train, X_test, y_train, y_test = train_test_split(text, labels,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=labels)


#define a tokenizer object
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


#tokenize the text (padding to max sequence in batch)
train_encodings = tokenizer(list(X_train.values), truncation=True, padding=True)
test_encodings = tokenizer(list(X_test.values), truncation=True, padding=True)


#print the first paragraph and its transformation
# print(f'First paragraph: {X_train[0]}')
# print(f'Input ids: {train_encodings["input_ids"][0]}')
# print(f'Attention mask: {train_encodings["attention_mask"][0]}')


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

    def train_dataloader(self,):
        train_dataset = TabularToTextDataset(self.train_encodings, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=12)
        return train_loader
  
    def test_dataloader(self,):
        test_dataset = TabularToTextDataset(self.test_encodings, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=12)
        return test_loader

    def val_dataloader(self,):
        test_dataset = TabularToTextDataset(self.test_encodings, self.y_test)
        test_loader = DataLoader(test_dataset, batch_size=12)
        return test_loader
    
    
data_module = TabularToTextDM(train_encodings, y_train, test_encodings, y_test)
data_module.train_dataloader()




import torch
from torch.utils.data import DataLoader, random_split


class SimpleDataLoaderBuilder:
    def __init__(self, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.train_size = None
        self.test_dataset = test_dataset
        self.test_size = None
        self.loader_train = None
        self.loader_test = None

    def build(self):
        self.loader_test = DataLoader(self.test_dataset, batch_size=10, shuffle=True, drop_last=False)
        self.loader_train = DataLoader(self.train_dataset, batch_size=24, shuffle=True, drop_last=False)


class DataLoaderBuilder:
    def __init__(self, dataset, location_folder, train_percent=0.8, test_percent=0.2):
        self.dataset = dataset
        self.train_size = None
        self.test_size = None
        self.loader_train = None
        self.loader_test = None
        self.location_folder = location_folder
        self._validate_sizes(train_percent, test_percent)

    def _validate_sizes(self, train_percent, test_percent):
        if train_percent + test_percent != 1:
            raise Exception("Train and Test percentages should sum 1.00")
        self.train_size = round(len(self.dataset) * train_percent)
        self.test_size = round(len(self.dataset) * test_percent)

    def build(self):
        iris_test, iris_train = random_split(
            self.dataset, [self.test_size, self.train_size], generator=torch.Generator().manual_seed(3)
        )
        self.loader_test = DataLoader(iris_test, batch_size=10, shuffle=True, drop_last=False)
        self.loader_train = DataLoader(iris_train, batch_size=24, shuffle=True, drop_last=False)

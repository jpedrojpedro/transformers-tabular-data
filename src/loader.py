import torch
from torch.utils.data import DataLoader, random_split


class DataLoaderBuilder:
    def __init__(self, dataset, location_folder, train_percent=0.8, validation_percent=0.2):
        self.dataset = dataset
        self.train_size = None
        self.validation_size = None
        self.loader_train = None
        self.loader_validation = None
        self.location_folder = location_folder
        self._validate_sizes(train_percent, validation_percent)

    def _validate_sizes(self, train_percent, validation_percent):
        if train_percent + validation_percent != 1:
            raise Exception("Train and Validation percentages should sum 1.00")
        self.train_size = round(len(self.dataset) * train_percent)
        self.validation_size = round(len(self.dataset) * validation_percent)

    def build(self):
        iris_validation, iris_train = random_split(
            self.dataset, [self.validation_size, self.train_size], generator=torch.Generator().manual_seed(3)
        )
        self.loader_validation = DataLoader(iris_validation, batch_size=10, shuffle=True, drop_last=False)
        self.loader_train = DataLoader(iris_train, batch_size=24, shuffle=True, drop_last=False)
        
        import os
        os.chdir(os.path.dirname(__file__))        
        
        
        with open(self.location_folder + 'train.data', 'w') as f:
            for x in self.loader_train:
                f.write(str(x))
        with open(self.location_folder + 'test.data', 'w') as f:
            for x in self.loader_validation:
                f.write(str(x))

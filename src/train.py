import time
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from torch import nn
from torch import optim
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# from transformers import T5Tokenizer, T5ForConditionalGeneration


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class IrisDataset(Dataset):
    def __init__(self, src_file, root_dir, transform=None):
        # data like: 5.0, 3.5, 1.3, 0.3, 0
        self.data = np.loadtxt(src_file, usecols=range(0, 5), delimiter=",", skiprows=0, dtype=np.float32)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # preds = torch.from_numpy(self.data[idx, 0:4]).to(device)
        preds = tokenizer.encode(
            '  '.join([str(round(float(pred), 2)) for pred in self.data[idx, 0:4]]), return_tensors='pt', padding=True
        )
        preds = torch.reshape(preds, (-1,)).to(device)
        spcs = torch.tensor(self.data[idx, 4:], dtype=torch.long).to(device)
        sample = (preds, spcs)
        if self.transform:
            sample = self.transform(sample)
        return sample


MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'
# model_ft = T5ForConditionalGeneration.from_pretrained('t5-small')
# tokenizer = T5Tokenizer.from_pretrained('t5-small')
model_ft = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

datasets_folder = Path(__file__).parent.parent / "datasets"
iris_ds = IrisDataset(datasets_folder / "iris" / "iris.data", root_dir=datasets_folder / "iris")
train_size = round(len(iris_ds) * 0.75)
validation_size = round(len(iris_ds) * 0.25)
iris_validation, iris_train = random_split(
    iris_ds, [validation_size, train_size], generator=torch.Generator().manual_seed(3)
)
loader_validation = DataLoader(iris_validation, batch_size=10, shuffle=True, drop_last=False)
loader_train = DataLoader(iris_train, batch_size=10, shuffle=True, drop_last=False)

data_loaders = {
    'train': loader_train,
    'val': loader_validation
}
dataset_sizes = {
    'train': train_size,
    'val': validation_size
}
class_names = ['setosa', 'versicolor', 'virginica']


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # 'batch_sampler', 'batch_size', 'check_worker_number_rationality', 'collate_fn', 'dataset','drop_last',
            # 'generator', 'multiprocessing_context', 'num_workers', 'persistent_workers', 'pin_memory',
            # 'prefetch_factor', 'sampler', 'timeout', 'worker_init_fn'
            for inputs, labels in data_loaders[phase]:
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                one_hot_labels = one_hot(labels, num_classes=3)
                one_hot_labels = torch.squeeze(one_hot_labels)
                one_hot_labels = one_hot_labels.float()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).logits
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, one_hot_labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    print(model_ft)
    # BERT modifications
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, len(class_names))
    model_ft.dropout = nn.Identity()

    # T5 modifications
    # num_ftrs = model_ft.lm_head.in_features
    # model_ft.lm_head = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Start training
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)

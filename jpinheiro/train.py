import time
import torch
from copy import deepcopy
from torch.nn.functional import one_hot


def normalize_predictions(outputs):
    result_tensor = torch.max(outputs, 1).values
    new_tensor = torch.zeros(result_tensor.shape)
    for idx, t in enumerate(result_tensor):
        new_tensor[idx] = one_hot(torch.argmax(t), num_classes=len(t))
    return torch.argmax(new_tensor, 1, keepdim=True)


class TrainAndValidate:
    def __init__(self, data_loader, model, criterion, optimizer, scheduler=None, num_epochs=25):
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs

    def run(self):
        since = time.time()
        best_model_wts = deepcopy(self.model.state_dict())
        best_acc = 0.0
        data_loaders = {
            'train': self.data_loader.loader_train,
            'val': self.data_loader.loader_validation
        }
        dataset_sizes = {
            'train': self.data_loader.train_size,
            'val': self.data_loader.validation_size
        }

        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in data_loaders[phase]:
                    # inputs = inputs.to(device)
                    # labels = labels.to(device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    one_hot_labels = one_hot(labels, num_classes=self.data_loader.dataset.num_classes())
                    one_hot_labels = torch.squeeze(one_hot_labels)
                    with torch.set_grad_enabled(phase == 'train'):
                        if 'bert' in self.model.base_model_prefix:
                            one_hot_labels = one_hot_labels.float()
                            outputs = self.model(inputs.long()).logits
                            _, predictions = torch.max(outputs, 1, keepdim=True)
                        elif 'transformer' in self.model.base_model_prefix:
                            one_hot_labels = one_hot_labels.long()
                            outputs = self.model(input_ids=inputs.long(), labels=one_hot_labels).logits
                            predictions = normalize_predictions(outputs)
                        else:
                            raise NotImplementedError('Undefined model')
                        loss = self.criterion(outputs, one_hot_labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(predictions == labels)
                if phase == 'train' and self.scheduler:
                    self.scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = deepcopy(self.model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model

import time
import torch
import datetime as dt
from pathlib import Path
from torch.nn.functional import one_hot
from torch import nn
from torch import optim
import torchmetrics as tm


def model_state_path(model_prefix) -> Path:
    base_path = Path(__file__)
    state_dir = base_path.parent.parent / "model_state" / model_prefix
    return state_dir


class TrainAndValidate:
    def __init__(self,
                 data_loader,
                 model,
                 loss_fn=nn.CrossEntropyLoss,
                 opt_fn=optim.Adam,
                 acc_fn=tm.Accuracy,
                 scheduler=optim.lr_scheduler.StepLR,
                 scheduler_step_size=25,
                 scheduler_factor=0.1,
                 num_epochs=75,
                 learning_rate=1e-5
                 ):
        self.data_loader = data_loader
        self.model = model
        self._set_model_prefix()
        self.loss_fn = loss_fn()
        self.optimizer = opt_fn(model.parameters(), lr=learning_rate)
        self.train_acc = acc_fn()
        self.val_acc = acc_fn()
        self.scheduler = scheduler(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_factor)
        self.num_epochs = num_epochs

    def train(self):
        since = time.time()
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, self.num_epochs))
            print('-' * 10)
            running_loss = 0.0
            training_total = 0
            for inputs, labels in self.data_loader.loader_train:
                if self.model_prefix == 't5':
                    full_outputs = self.model(
                        input_ids=inputs['encoded_inputs_ids'],
                        attention_mask=inputs['attention_mask_inputs'],
                        labels=labels['encoded_outputs_ids'],
                        decoder_attention_mask=labels['attention_mask_outputs'],
                    )
                    batch_len = inputs['encoded_inputs_ids'].size(0)
                    outputs = torch.argmax(full_outputs.logits, dim=2)
                    loss = full_outputs.loss
                    one_hot_labels = labels['encoded_outputs_ids']
                else:
                    batch_len, outputs, one_hot_labels = self._boilerplate(inputs, labels)
                    loss = self.loss_fn(outputs, one_hot_labels)
                training_total += batch_len
                running_loss += loss.item() * inputs['encoded_inputs_ids'].size(0)
                # Backward pass
                loss.backward()
                self.optimizer.step()
                self.train_acc(outputs, one_hot_labels.int())
            # self.scheduler.step()
            epoch_acc = self.train_acc.compute()
            epoch_loss = running_loss / training_total
            print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            print()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.persist_model()

    def validate(self, model_state='20211130-201153_state.pt'):
        self.load_model(model_state)
        since = time.time()
        print('Starting validation')
        print('-' * 10)
        running_loss = 0.0
        validation_total = 0
        for inputs, labels in self.data_loader.loader_validation:
            if self.model_prefix == 't5':
                full_outputs = self.model(
                    input_ids=inputs['encoded_inputs_ids'],
                    attention_mask=inputs['attention_mask_inputs'],
                    labels=labels['encoded_outputs_ids'],
                    decoder_attention_mask=labels['attention_mask_outputs'],
                )
                batch_len = inputs['encoded_inputs_ids'].size(0)
                outputs = full_outputs.logits
                loss = full_outputs.loss
                one_hot_labels = one_hot(
                    labels['encoded_outputs_ids'],
                    num_classes=outputs.size(2)
                )
            else:
                batch_len, outputs, one_hot_labels = self._boilerplate(inputs, labels)
                loss = self.loss_fn(outputs, one_hot_labels)
            validation_total += batch_len
            running_loss += loss.item() * inputs['encoded_inputs_ids'].size(0)
            self.val_acc(outputs, one_hot_labels.int())
        final_acc = self.val_acc.compute()
        final_loss = running_loss / validation_total
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(final_loss, final_acc))
        print()
        time_elapsed = time.time() - since
        print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def _boilerplate(self, inputs, labels):
        batch_len = len(inputs)
        one_hot_labels = one_hot(labels, num_classes=self.data_loader.dataset.num_classes())
        one_hot_labels = torch.squeeze(one_hot_labels)

        if 'bert' in self.model_prefix:
            one_hot_labels = one_hot_labels.float()
            outputs = self.model(inputs.long()).logits
        elif 't5' == self.model_prefix or 't0' == self.model_prefix:
            one_hot_labels = one_hot_labels.long()
            outputs = self.model(input_ids=inputs.long(), labels=one_hot_labels).logits
        elif 'transformer' == self.model_prefix or 'gpt2' == self.model_prefix:
            one_hot_labels = one_hot_labels.float()
            outputs = self.model(inputs.long()).logits
        else:
            raise NotImplementedError('Undefined model')

        return batch_len, outputs, one_hot_labels

    def _set_model_prefix(self):
        model_prefix = self.model.name_or_path
        if 'bert' in model_prefix.lower():
            self.model_prefix = 'bert'
        elif 't5' in model_prefix.lower():
            self.model_prefix = 't5'
        elif 't0' in model_prefix.lower():
            self.model_prefix = 't0'
        elif 'gpt2' in model_prefix.lower():
            self.model_prefix = 'gpt2'
        else:
            self.model_prefix = 'undefined'

    def persist_model(self):
        now = dt.datetime.now()
        state_filename = "{}-{}.pt".format(now.strftime("%Y%m%d-%H%M%S"), self.data_loader.dataset.name())
        full_path = model_state_path(self.model_prefix) / state_filename
        with open(full_path, 'w'):
            torch.save(self.model.state_dict(), full_path)

    def load_model(self, state_filename):
        full_path = model_state_path(self.model_prefix) / state_filename
        self.model.load_state_dict(torch.load(full_path))
        self.model.eval()

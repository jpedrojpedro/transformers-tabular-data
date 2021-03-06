import time
# import ipdb
import torch
import datetime as dt
from pathlib import Path
from torch.nn.functional import one_hot
from torch import nn
from torch import optim
import torchmetrics as tm
from torchmetrics.functional import f1, recall
from src.metrics import accuracy


def model_state_path(model_prefix, log=False) -> Path:
    base_path = Path(__file__)
    state_dir = base_path.parent.parent / "model_state" / model_prefix
    if log:
        state_dir = state_dir / "logs"
    return state_dir


def naming(dataset_name, extension='pt'):
    now = dt.datetime.now()
    return "{}-{}.{}".format(now.strftime("%Y%m%d-%H%M%S"), dataset_name, extension)


class TrainAndValidate:
    def __init__(self,
                 data_loader,
                 model,
                 device,
                 loss_fn=nn.CrossEntropyLoss,
                 opt_fn=optim.Adam,
                 acc_fn=tm.Accuracy,
                 scheduler=optim.lr_scheduler.StepLR,
                 scheduler_step_size=25,
                 scheduler_factor=0.1,
                 num_epochs=75,
                 learning_rate=1e-5,
                 ):
        self.data_loader = data_loader
        self.data_train = self.data_loader.loader_train
        self.data_test = self.data_loader.loader_test
        self.num_classes = self.data_loader.train_dataset.num_classes()
        self.model = model
        self._set_model_prefix()
        self.loss_fn = loss_fn()
        self.optimizer = opt_fn(model.parameters(), lr=learning_rate)
        self.train_acc = acc_fn().to(device)
        self.val_acc = acc_fn().to(device)
        self.scheduler = scheduler(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_factor)
        self.num_epochs = num_epochs
        self.train_log_file = None

        
    def train(self):
        since = time.time()        
        for epoch in range(self.num_epochs):
            self.logging('Epoch {}/{}'.format(epoch + 1, self.num_epochs))
            self.logging('-' * 10)
            
            len_total = 0
            loss_total = 0.0
            f1_total = 0.0
            recall_total = 0.0
            acc_total = 0.0
            for inputs, labels in self.data_train:
                len_total += len(inputs)
                if self.model_prefix == 't5':                    
                    full_outputs = self.model(
                        input_ids=inputs['encoded_inputs_ids'],
                        attention_mask=inputs['attention_mask_inputs'],
                        labels=labels['encoded_outputs_ids'],
                        decoder_attention_mask=labels['attention_mask_outputs'],
                    )
                    y_pred = torch.argmax(full_outputs.logits, dim=2)
                    y_true_one_hot = labels['encoded_outputs_ids']
                    # ipdb.set_trace()
                    loss = full_outputs.loss
                    loss_total += loss.item() * inputs['encoded_inputs_ids'].size(0)
                    acc = self.calculate_acc(y_pred, y_true_one_hot)
                    acc_total += acc * inputs['encoded_inputs_ids'].size(0)
                else:
                    y_pred = self.model(inputs.long()).logits
                    y_true_one_hot = one_hot(labels, num_classes=self.num_classes)
                    y_true_one_hot = y_true_one_hot.float()
                    loss = self.loss_fn(y_pred, y_true_one_hot)
                    loss_total += loss.item() * len(inputs)
                    # BERT: acc and loss are calculated the same way
                    # y_pred = [-0.06, 0.984, 0.12]
                    # y_true = [0, 1, 0]
                    self.train_acc(y_pred, y_true_one_hot.int())

                if self.model_prefix != 't5':
                    f1_score = f1(y_pred, y_true_one_hot.int(), self.num_classes)
                    f1_total += f1_score * len(inputs)
                    
                    recall_score = recall(y_pred, y_true_one_hot.int(), average='macro', num_classes=self.num_classes)
                    recall_total += recall_score * len(inputs)

                # Backward pass
                loss.backward()
                self.optimizer.step()

            epoch_acc = self.train_acc.compute() if self.model_prefix != 't5' else acc_total / len_total
            epoch_loss = loss_total / len_total
            epoch_f1 = f1_total / len_total
            epoch_recall = recall_total / len_total

            self.logging('Train Loss: {:.4f} Acc: {:.4f} F1: {:.4f} Recall: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1, epoch_recall))
            self.logging('')
        
        # Total time
        time_elapsed = time.time() - since
        self.logging('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.persist_model()


    def validate(self, model_state='20211130-201153_state.pt'):
        self.load_model(model_state)
        since = time.time()
        print('Starting test')
        print('-' * 10)

        loss_total = 0.0
        len_total = 0
        for inputs, labels in self.data_test:
            len_total += len(inputs)
            if self.model_prefix == 't5':
                full_outputs = self.model(
                    input_ids=inputs['encoded_inputs_ids'],
                    attention_mask=inputs['attention_mask_inputs'],
                    labels=labels['encoded_outputs_ids'],
                    decoder_attention_mask=labels['attention_mask_outputs'],
                )
                y_pred = full_outputs.logits
                y_true_one_hot = one_hot( labels['encoded_outputs_ids'], num_classes=y_pred.size(2) )
                
                loss = full_outputs.loss
                loss_total += loss.item() * inputs['encoded_inputs_ids'].size(0)
            else:
                y_pred = self.model(inputs.long()).logits
                y_true_one_hot = one_hot(labels, num_classes=self.num_classes)
                y_true_one_hot = y_true_one_hot.float()
                
                loss = self.loss_fn(y_pred, y_true_one_hot)
                loss_total += loss.item() * len(inputs)

            self.val_acc(y_pred, y_true_one_hot.int())

        final_acc = self.val_acc.compute()
        final_loss = loss_total / len_total
        print('Validation Loss: {:.4f} Acc: {:.4f}'.format(final_loss, final_acc))
        print()
        time_elapsed = time.time() - since
        print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


    ########## ------------ FUNCTIONS FOR LOGGING RESULTS ------------ ##########

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
        if not self.train_log_file:
            self.train_log_file = naming(self.data_loader.train_dataset.name())
        else:
            self.train_log_file = self.train_log_file[:-3] + 'pt'
        full_path = model_state_path(self.model_prefix) / self.train_log_file
        with open(full_path, 'w'):
            torch.save(self.model.state_dict(), full_path)
        self.train_log_file = None

    def load_model(self, state_filename):
        full_path = model_state_path(self.model_prefix) / state_filename
        self.model.load_state_dict(torch.load(full_path))
        self.model.eval()

    def logging(self, msg):
        log_folder = model_state_path(self.model_prefix, log=True)
        if not self.train_log_file:
            self.train_log_file = naming(self.data_loader.train_dataset.name(), extension='txt')
        with open(log_folder / self.train_log_file, 'a') as fp:
            print(msg, file=fp)
            print(msg)

    def calculate_acc(self, y_pred_batch, y_true_batch):
        fn = self.data_train.dataset.tokenizer.decode
        y_pred = [fn(p, skip_special_tokens=True) for p in y_pred_batch]
        y_true = [fn(t, skip_special_tokens=True) for t in y_true_batch]
        return accuracy(y_true, y_pred)['accuracy']

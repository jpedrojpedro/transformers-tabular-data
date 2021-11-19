import torch 
from torch import nn 
from torch import optim 
import torchmetrics as tm
import pytorch_lightning as pl
from transformers import DistilBertForSequenceClassification

MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'


class DistilBertTabular(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.classifier = nn.Linear(768, num_classes)
        self.model.dropout = nn.Identity()
        
        self.train_acc = tm.Accuracy()
        self.val_acc = tm.Accuracy()
        self.loss_fn = nn.CrossEntropyLoss()
  

    def forward(self, **kwargs):
        preds = self.model(**kwargs)
        return preds.logits

    
    def training_step(self, batch, batch_id):
        batch, labels = batch
        keys = batch.keys()
        instance = {}
        for key in keys:
            instance[key] = torch.stack(batch[key], dim=0).transpose(1, 0).cuda()
        pred = self.forward(**instance)
        loss_value = self.loss_fn(pred, labels)
        acc = self.train_acc(pred, labels)
        self.log("train_loss", loss_value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_value
  

    def training_epoch_end(self, training_step_outputs):
        train_acc = self.train_acc.compute()
        self.log("train_acc", train_acc, on_epoch=True, prog_bar=True, logger=True)
  

    def validation_step(self, batch, batch_id):
        batch, labels = batch
        keys = batch.keys()
        instance = {}
        for key in keys:
            instance[key] = torch.stack(batch[key], dim=0).transpose(1, 0).cuda()
        pred = self.forward(**instance)
        loss_value = self.loss_fn(pred, labels)
        acc = self.val_acc(pred, labels)
        self.log('val_loss', loss_value, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return pred
    

    def test_step(self, batch, batch_id):
        return self.validation_step(batch, batch_id)

    
    def validation_epoch_end(self, validation_step_outputs):
        acc = self.val_acc.compute()
        print('validation acc', acc)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        return acc 
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

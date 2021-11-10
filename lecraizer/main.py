
from transformers import DistilBertTokenizer
# from transformers import DistilBertForSequenceClassification
import torch

# pd.set_option('display.max_colwidth', None)
MODEL_NAME = 'distilbert-base-uncased-finetuned-sst-2-english'  # This model is a fine-tune checkpoint of DistilBERT-base-uncased, fine-tuned on SST-2.
# BATCH_SIZE = 60
# N_EPOCHS = 3
import pytorch_lightning as pl



from loader import Loader
from tabtext import TabularToTextDM
from train import DistilBertTabular


if __name__ == '__main__':

    D = Loader()
    X_train, X_test, y_train, y_test = D.load_data()
        
           
    #define a tokenizer object
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


    #tokenize the text (padding to max sequence in batch)
    train_encodings = tokenizer(list(X_train.values), truncation=True, padding=True)
    test_encodings = tokenizer(list(X_test.values), truncation=True, padding=True)
    
    
    data_module = TabularToTextDM(train_encodings, y_train, test_encodings, y_test)
        
        
    model = DistilBertTabular(3)
    model = model.cuda()
    trainer = pl.Trainer(gpus=1)
    trainer.test(model, data_module.test_dataloader())
    trainer.fit(model=model, datamodule=data_module)
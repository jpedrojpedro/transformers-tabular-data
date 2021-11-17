import pytorch_lightning as pl

from loader import Loader
from tokenizer import TabularToTextDM
from train import DistilBertTabular


if __name__ == '__main__':

    # Load data
    L = Loader()
    X_train, X_test, y_train, y_test = L.load_data()
    
    # Convert input data to the proper form for training
    data_module = TabularToTextDM(X_train, y_train, X_test, y_test)    

    # Load model
    bert_model = DistilBertTabular(len(set(y_train)))
    bert_model = bert_model.cuda()
    
    # Train model
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model=bert_model, datamodule=data_module)
    trainer.test(bert_model, data_module.test_dataloader())
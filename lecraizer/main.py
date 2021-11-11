import pytorch_lightning as pl


from loader import Loader
from tokenizer import TabularToTextDM
from train import DistilBertTabular


if __name__ == '__main__':

    L = Loader()
    X_train, X_test, y_train, y_test = L.load_data()
                  
    data_module = TabularToTextDM(X_train, y_train, X_test, y_test)    

    num_classes = len(set(y_train))
    model = DistilBertTabular(num_classes)
    model = model.cuda()
    
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model, data_module.test_dataloader())
import pytorch_lightning as pl

from loader import Loader
from tokenizer import TabularToTextDM
from train import DistilBertTabular


if __name__ == '__main__':

    # Load data
    L = Loader()
    X_train, X_test, y_train, y_test = L.load_data()
    print(X_train.shape)
    
    # Convert input data to the proper form for training
    data_module = TabularToTextDM(X_train, y_train, X_test, y_test)
#     print(data_module)

    # Load model
    num_classes = len(set(y_train))
    model = DistilBertTabular(num_classes)
    model = model.cuda()
    
    
#     # freezing layers
#     for i, (name, param) in enumerate(model.named_parameters()):
#         if i < 102:


    for name, param in model.named_parameters():
        if 'layer_norm' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
                   
                              
    for name, param in model.named_parameters():
        print(name)
        print(param.requires_grad)
        print('\n')
    
    
    # Train model
    trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model, data_module.test_dataloader())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     Print model summary
#     print(model)
    

    


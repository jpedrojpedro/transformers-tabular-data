import torch
from pathlib import Path
from torch import nn
from torch import optim
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import IrisDataset
from loader import DataLoaderBuilder
from train import TrainAndValidate


def load_bert(num_classes):
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    model_ft = DistilBertForSequenceClassification.from_pretrained(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    model_ft.dropout = nn.Identity()
    return model_ft, tokenizer


def load_t5(num_classes):
    model_name = 't5-small'
    model_ft = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    num_ftrs = model_ft.lm_head.in_features
    model_ft.lm_head = nn.Linear(num_ftrs, num_classes)
    return model_ft, tokenizer


def main():
    datasets_folder = Path(__file__).parent.parent / "datasets"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iris_ds = IrisDataset(datasets_folder / "iris" / "iris.data", datasets_folder / "iris", device)
    model_ft, tokenizer = load_bert(iris_ds.num_classes())
    # model_ft, tokenizer = load_t5(iris_ds.num_classes())
    iris_ds.tokenizer = tokenizer
    data_loader = DataLoaderBuilder(iris_ds)
    data_loader.build()

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    # Decay LR by a factor of 0.1 every 10 epochs
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

    tv = TrainAndValidate(
        data_loader, model_ft.to(device), criterion, optimizer_ft, num_epochs=100
    )
    tv.run()


if __name__ == '__main__':
    main()

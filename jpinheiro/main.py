import torch
from pathlib import Path
from torch import nn
from torch import optim
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import IrisDataset
from loader import DataLoaderBuilder
from train import TrainAndValidate


def load_bert():
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    return model, tokenizer


def customize_bert(model_ft, num_classes=3):
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    model_ft.dropout = nn.Identity()
    return model_ft


def load_t5():
    model_name = 't5-small'
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    return model, tokenizer


def customize_t5(model_ft, num_classes=3):
    num_ftrs = model_ft.lm_head.in_features
    model_ft.lm_head = nn.Linear(num_ftrs, num_classes)
    return model_ft


def main():
    datasets_folder = Path(__file__).parent.parent / "datasets"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft, tokenizer = load_t5()
    # model_ft, tokenizer = load_bert()
    iris_ds = IrisDataset(datasets_folder / "iris" / "iris.data", datasets_folder / "iris", tokenizer, device)
    data_loader = DataLoaderBuilder(iris_ds)
    data_loader.build()

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = customize_t5(model_ft)
    # model_ft = customize_bert(model_ft)
    tv = TrainAndValidate(data_loader, model_ft.to(device), criterion, optimizer_ft)
    tv.run()


if __name__ == '__main__':
    main()
    # print(model_ft)

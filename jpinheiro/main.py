import torch
from pathlib import Path
from torch import nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import IrisDataset
from loader import DataLoaderBuilder
from train import TrainAndValidate


def load_bert(num_classes, freeze=False):
    model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
    model_ft = DistilBertForSequenceClassification.from_pretrained(model_name)
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    model_ft.dropout = nn.Identity()
    if freeze:
        # freezing all parameters
        for param in model_ft.parameters():
            param.requires_grad = False
        # activating specific layers
        for i in range(6):
            model_ft.distilbert.transformer.layer[i].sa_layer_norm.requires_grad_(True)
            model_ft.distilbert.transformer.layer[i].ffn.dropout.requires_grad_(True)
            model_ft.distilbert.transformer.layer[i].ffn.lin1.requires_grad_(True)
            model_ft.distilbert.transformer.layer[i].ffn.lin2.requires_grad_(True)
            model_ft.distilbert.transformer.layer[i].output_layer_norm.requires_grad_(True)
        model_ft.classifier.requires_grad_(True)
        model_ft.dropout.requires_grad_(True)
    return model_ft, tokenizer


def load_t5(num_classes, freeze=False):
    model_name = 't5-small'
    model_ft = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    num_ftrs = model_ft.lm_head.in_features
    model_ft.lm_head = nn.Linear(num_ftrs, num_classes)
    if freeze:
        # freezing all parameters
        for param in model_ft.parameters():
            param.requires_grad = False
        # activating specific layers
        for i in range(6):
            for j in range(3):
                model_ft.decoder.block[i].layer[j].layer_norm.requires_grad_(True)
                model_ft.decoder.block[i].layer[j].dropout.requires_grad_(True)
        model_ft.lm_head.requires_grad_(True)
    return model_ft, tokenizer


def load_t0(num_classes, freeze=False):
    model_name = 'bigscience/T0_3B'
    model_ft = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_ftrs = model_ft.lm_head.in_features
    model_ft.lm_head = nn.Linear(num_ftrs, num_classes)
    if freeze:
        # freezing all parameters
        for param in model_ft.parameters():
            param.requires_grad = False
        # activating specific layers
        for i in range(24):
            for j in range(3):
                model_ft.decoder.block[i].layer[j].layer_norm.requires_grad_(True)
                model_ft.decoder.block[i].layer[j].dropout.requires_grad_(True)
        model_ft.decoder.final_layer_norm.requires_grad_(True)
        model_ft.decoder.dropout.requires_grad_(True)
        model_ft.lm_head.requires_grad_(True)
    return model_ft, tokenizer


def main():
    datasets_folder = Path(__file__).parent.parent / "datasets"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    iris_ds = IrisDataset(datasets_folder / "iris" / "iris.data", datasets_folder / "iris", device)
    # model_ft, tokenizer = load_bert(iris_ds.num_classes())
    # model_ft, tokenizer = load_t5(iris_ds.num_classes())
    model_ft, tokenizer = load_t0(iris_ds.num_classes(), freeze=True)
    iris_ds.tokenizer = tokenizer
    data_loader = DataLoaderBuilder(iris_ds)
    data_loader.build()

    tv = TrainAndValidate(data_loader, model_ft.to(device))
    tv.train()
    # tv.validate()


if __name__ == '__main__':
    main()

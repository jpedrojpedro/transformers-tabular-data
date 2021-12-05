import torch
from pathlib import Path
from torch import nn
from datasets import IrisDataset, AbaloneDataset, concat_table_values, written_form_table_values
from loader import DataLoaderBuilder
from train import TrainAndValidate


def load_roberta(num_classes):
    from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaTokenizer

    config = RobertaConfig.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model_ft = RobertaForSequenceClassification(config)
    num_ftrs = model_ft.classifier.out_proj.in_features
    model_ft.classifier.out_proj = nn.Linear(num_ftrs, num_classes)
    model_ft.dropout = nn.Identity()
    return model_ft, tokenizer


def load_bert(num_classes, freeze=False):
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

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
    from transformers import T5Tokenizer, T5ForConditionalGeneration

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
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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


def load_gpt2(num_classes, freeze=False):
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification, GPT2Config

    model_name = 'gpt2-medium'
    model_ft = GPT2ForSequenceClassification.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    num_ftrs = model_ft.score.in_features
    model_ft.score = nn.Linear(num_ftrs, num_classes)

    # model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name, num_labels=num_classes)
    # Adding padding left and right
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Loading model
    # model_ft = GPT2ForSequenceClassification.from_pretrained(model_name)
    # model_ft.resize_token_embeddings(len(tokenizer))
    model_ft.config.pad_token_id = model_ft.config.eos_token_id

    if freeze:
        # freezing all parameters
        for param in model_ft.parameters():
            param.requires_grad = False
        # activating specific layers
        for i in range(24):
            model_ft.transformer.h[i].ln_1.requires_grad_(True)
            model_ft.transformer.h[i].ln_2.requires_grad_(True)
        model_ft.score.requires_grad_(True)
    return model_ft, tokenizer


def select_process_combination():
    opts = {
        1: ("bert", "iris"),
        2: ("t5", "iris"),
        3: ("gpt2", "iris"),
        4: ("bert", "abalone"),
        5: ("t5", "abalone"),
        6: ("gpt2", "abalone"),
        99: ("exit", ""),
    }
    print("What combination do you want?")
    print("-" * 10)
    for key, val in opts.items():
        print(f"{key} - {' + '.join(val)}")
    sel = int(input("> "))
    print("")
    return opts[sel]


def main():
    model, dataset = select_process_combination()
    if model == 'exit':
        return

    datasets_folder = Path(__file__).parent.parent / "datasets"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == 'iris':
        ds = IrisDataset(
            datasets_folder / "iris" / "iris.data", datasets_folder / "iris", device,
            build_input_fn=written_form_table_values, max_encoded_len=15
        )
    elif dataset == 'abalone':
        ds = AbaloneDataset(
            datasets_folder / "abalone" / "abalone_str.data", datasets_folder / "abalone", device,
            build_input_fn=concat_table_values, max_encoded_len=30
        )
    else:
        raise FileNotFoundError("Invalid Dataset selection")

    if model == 'bert':
        model_fn = load_bert
    elif model == 't5':
        model_fn = load_t5
    elif model == 'gpt2':
        model_fn = load_gpt2
    else:
        raise ModuleNotFoundError("Invalid Model selection")

    model_ft, tokenizer = model_fn(ds.num_classes(), freeze=True)
    ds.tokenizer = tokenizer
    data_loader = DataLoaderBuilder(ds)
    data_loader.build()
    tv = TrainAndValidate(data_loader, model_ft)
    tv.train()


if __name__ == '__main__':
    main()

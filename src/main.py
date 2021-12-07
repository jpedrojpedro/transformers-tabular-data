import torch
from pathlib import Path
from datasets import IrisWrittenDataset,    \
                     IrisConcatDataset,     \
                     AbaloneWrittenDataset, \
                     AbaloneConcatDataset,  \
                     IrisT5Dataset

from torch import nn
from loader import DataLoaderBuilder
from train import TrainAndValidate
from models import load_bert, load_t5, load_gpt2


def select_process_combination():
    opts = {
        1: ("bert", "iris-concat"),
        2: ("bert", "iris-written"),
        3: ("bert", "abalone-concat"),
        4: ("bert", "abalone-written"),
        5: ("t5", "iris-concat"),
        6: ("t5", "iris-written"),
        7: ("t5", "abalone-concat"),
        8: ("t5", "abalone-written"),
        9: ("gpt2", "iris-concat"),
        10: ("gpt2", "iris-written"),
        11: ("gpt2", "abalone-concat"),
        12: ("gpt2", "abalone-written"),
        13: ("t5", "iris-t5"),
        99: ("exit", "exit"),
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
    iris_data_file = datasets_folder / "iris" / "iris.data"
    abalone_data_file = datasets_folder / "abalone" / "abalone_str.data"
    corona_data_file = datasets_folder / "corona" / "corona.data"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == 'iris-written':
        ds = IrisWrittenDataset(iris_data_file, iris_data_file.parent, device)
    elif dataset == 'iris-concat':
        ds = IrisConcatDataset(iris_data_file, iris_data_file.parent, device)
    elif dataset == 'iris-t5':
        ds = IrisT5Dataset(iris_data_file, iris_data_file.parent, device)
    elif dataset == 'abalone-written':
        ds = AbaloneWrittenDataset(abalone_data_file, abalone_data_file.parent, device)
    elif dataset == 'abalone-concat':
        ds = AbaloneConcatDataset(abalone_data_file, abalone_data_file.parent, device)
    elif dataset == 'corona-concat':
        ds = CoronaDataset(corona_data_file, corona_data_file.parent, device)
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

    model_ft, tokenizer = model_fn(ds.num_classes(), freeze=False)
    ds.tokenizer = tokenizer
    data_loader = DataLoaderBuilder(ds)
    data_loader.build()
    tv = TrainAndValidate(data_loader, model_ft, num_epochs=10)
    tv.train()
    # tv.validate(model_state='20211207-134546-iris-t5.pt')


if __name__ == '__main__':
    main()

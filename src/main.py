import torch
from pathlib import Path
from datasets import *
from loader import SimpleDataLoaderBuilder
from train import TrainAndValidate
from models import load_bert, load_t5, load_gpt2


def select_process_combination():
    opts = {
        1: ("bert", "iris-concat"),
        3: ("bert", "abalone-concat"),

        # 9: ("gpt2", "iris-concat"),
        # 11: ("gpt2", "abalone-concat"),

        13: ("t5", "iris-t5"),
        15: ("t5", "abalone-t5"),

        80: ("bert", "adult-concat"),
        81: ("t5", "adult-concat"),
        82: ("gpt2", "adult-concat"),

        87: ("bert", "pulsar-concat"),
        88: ("t5", "pulsar-concat"),
        89: ("gpt2", "pulsar-concat"),

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == 'iris-concat':
        ds_train = IrisConcatDataset(datasets_folder / "iris" / "iris_train.data", device)
        ds_test = IrisConcatDataset(datasets_folder / "iris" / "iris_test.data", device)
    elif dataset == 'iris-t5':
        ds_train = IrisT5Dataset(datasets_folder / "iris" / "iris_train.data", device)
        ds_test = IrisT5Dataset(datasets_folder / "iris" / "iris_test.data", device)
    elif dataset == 'abalone-t5':
        ds_train = AbaloneT5Dataset(datasets_folder / "abalone" / "abalone_train.data", device)
        ds_test = AbaloneT5Dataset(datasets_folder / "abalone" / "abalone_test.data", device)
    elif dataset == 'abalone-concat':
        ds_train = AbaloneConcatDataset(datasets_folder / "abalone" / "abalone_train.data", device)
        ds_test = AbaloneConcatDataset(datasets_folder / "abalone" / "abalone_test.data", device)
    elif dataset == 'adult-concat':
        ds_train = AdultConcatDataset(datasets_folder / "adult" / "adult_train.data", device)
        ds_test = AdultConcatDataset(datasets_folder / "adult" / "adult_test.data", device)
    elif dataset == 'pulsar-concat':
        ds_train = PulsarConcatDataset(datasets_folder / "pulsar" / "pulsar_train.data", device)
        ds_test = PulsarConcatDataset(datasets_folder / "pulsar" / "pulsar_test.data", device)
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

    model_ft, tokenizer = model_fn(ds_train.num_classes(), freeze=True)
    ds_train.tokenizer = tokenizer
    ds_test.tokenizer = tokenizer
    data_loader = SimpleDataLoaderBuilder(ds_train, ds_test)
    data_loader.build()
    tv = TrainAndValidate(data_loader, model_ft, num_epochs=50, learning_rate=1e-5)
    tv.train()
    # tv.validate(model_state='20211207-134546-iris-t5.pt')


if __name__ == '__main__':
    main()

import torch
from pathlib import Path
from datasets import *
from loader import SimpleDataLoaderBuilder
from train import TrainAndValidate
from models import load_bert, load_t5, load_gpt2


def select_process_combination():
    opts = {
        1: ("bert", "iris text-to-label"),
        2: ("bert", "abalone text-to-label"),
        3: ("bert", "adult text-to-label"),
        4: ("bert", "pulsar text-to-label"),

        5: ("t5", "iris text-to-text"),
        6: ("t5", "abalone text-to-text"),
        7: ("t5", "adult text-to-text"),
        8: ("t5", "pulsar text-to-text"),
        
        9: ("gpt2", "iris text-to-label"),
        10: ("gpt2", "abalone text-to-label"),
        11: ("gpt2", "adult text-to-label"),
        12: ("gpt2", "pulsar text-to-label"),
        
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

    if dataset == 'iris text-to-label':
        ds_train = IrisConcatDataset(datasets_folder / "iris" / "iris_train.csv", device)
        ds_test = IrisConcatDataset(datasets_folder / "iris" / "iris_test.csv", device)
    elif dataset == 'iris text-to-text':
        ds_train = IrisT5Dataset(datasets_folder / "iris" / "iris_train.csv", device)
        ds_test = IrisT5Dataset(datasets_folder / "iris" / "iris_test.csv", device)
        
    elif dataset == 'abalone text-to-label':
        ds_train = AbaloneConcatDataset(datasets_folder / "abalone" / "abalone_train.csv", device)
        ds_test = AbaloneConcatDataset(datasets_folder / "abalone" / "abalone_test.csv", device)
    elif dataset == 'abalone text-to-text':
        ds_train = AbaloneT5Dataset(datasets_folder / "abalone" / "abalone_train.csv", device)
        ds_test = AbaloneT5Dataset(datasets_folder / "abalone" / "abalone_test.csv", device)
        
    elif dataset == 'adult text-to-label':
        ds_train = AdultConcatDataset(datasets_folder / "adult" / "adult_train.csv", device)
        ds_test = AdultConcatDataset(datasets_folder / "adult" / "adult_test.csv", device)
    elif dataset == 'adult text-to-text':
        ds_train = AdultT5Dataset(datasets_folder / "adult" / "adult_train.csv", device)
        ds_test = AdultT5Dataset(datasets_folder / "adult" / "adult_test.csv", device)
       
    elif dataset == 'pulsar text-to-label':
        ds_train = PulsarConcatDataset(datasets_folder / "pulsar" / "pulsar_train.csv", device)
        ds_test = PulsarConcatDataset(datasets_folder / "pulsar" / "pulsar_test.csv", device)        
    elif dataset == 'pulsar text-to-text':
        ds_train = PulsarT5Dataset(datasets_folder / "pulsar" / "pulsar_train.csv", device)
        ds_test = PulsarT5Dataset(datasets_folder / "pulsar" / "pulsar_test.csv", device)
        
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
#     tv.validate(model_state='20220119-020420-iris-concat.pt')


if __name__ == '__main__':
    main()

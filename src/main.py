import torch
from pathlib import Path
from datasets import *
from loader import SimpleDataLoaderBuilder
from train import TrainAndValidate
from models import load_bert, load_t5, load_gpt2


def select_process_combination():
    opts = {
        # 1: ("bert", "iris-concat"),
        # 2: ("bert", "iris-written"),
        # 3: ("bert", "abalone-concat"),
        # 4: ("bert", "abalone-written"),
        # 5: ("t5", "iris-concat"),
        # 6: ("t5", "iris-written"),
        # 7: ("t5", "abalone-concat"),
        # 8: ("t5", "abalone-written"),
        # 9: ("gpt2", "iris-concat"),
        # 10: ("gpt2", "iris-written"),
        # 11: ("gpt2", "abalone-concat"),
        # 12: ("gpt2", "abalone-written"),
        # 13: ("t5", "iris-t5"),
        # 14: ("t5", "iris-t5-written"),
        # 15: ("t5", "abalone-t5"),

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

    # if dataset == 'iris-written':
    #     ds = IrisWrittenDataset(iris_data_file, iris_data_file.parent, device)
    # elif dataset == 'iris-concat':
    #     ds = IrisConcatDataset(iris_data_file, iris_data_file.parent, device)
    # elif dataset == 'iris-t5':
    #     ds = IrisT5Dataset(iris_data_file, iris_data_file.parent, device)
    # elif dataset == 'iris-t5-written':
    #     ds = IrisT5WrittenDataset(iris_data_file, iris_data_file.parent, device)
    # elif dataset == 'abalone-t5':
    #     ds = AbaloneT5Dataset(abalone_data_file, abalone_data_file.parent, device)
    # elif dataset == 'abalone-written':
    #     ds = AbaloneWrittenDataset(abalone_data_file, abalone_data_file.parent, device)
    # elif dataset == 'abalone-concat':
    #     ds = AbaloneConcatDataset(abalone_data_file, abalone_data_file.parent, device)
    if dataset == 'adult-concat':
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
    data_loader = SimpleDataLoaderBuilder(ds_train, ds_test)
    data_loader.build()
    tv = TrainAndValidate(data_loader, model_ft, num_epochs=50, learning_rate=1e-5)
    tv.train()
    # tv.validate(model_state='20211207-134546-iris-t5.pt')


if __name__ == '__main__':
    main()

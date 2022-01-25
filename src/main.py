import torch
from pathlib import Path
from datasets import *
from loader import SimpleDataLoaderBuilder
from train import TrainAndValidate
from models import *


def select_process_combination():
    opts = {
        1: ("bert", "iris text-to-label"),
        2: ("bert", "abalone text-to-label"),
        3: ("bert", "adult text-to-label"),
        4: ("bert", "pulsar text-to-label"),
        5: ("bert", "nursery text-to-label"),

        11: ("t5", "iris text-to-text"),
        12: ("t5", "abalone text-to-text"),
        13: ("t5", "adult text-to-text"),
        14: ("t5", "pulsar text-to-text"),
        15: ("t5", "nursery text-to-text"),
        
        21: ("gpt2", "iris text-to-label"),
        22: ("gpt2", "abalone text-to-label"),
        23: ("gpt2", "adult text-to-label"),
        24: ("gpt2", "pulsar text-to-label"),
        25: ("gpt2", "nursery text-to-label"),
                
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
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # percentil of data in training_set
    percentils = (1, 10, 80) 
    perc = percentils[0]
    print('{}% training set'.format(perc))
    
    if dataset == 'iris text-to-label':
        train_string = "iris_train_perc" + str(perc) + ".csv"
        ds_train = IrisConcatDataset(datasets_folder / "iris" / str(perc) / train_string, device)
        test_string = "iris_test_perc" + str(perc) + ".csv"
        ds_test = IrisConcatDataset(datasets_folder / "iris" / str(perc) / test_string, device)
    elif dataset == 'iris text-to-text':
        train_string = "iris_train_perc" + str(perc) + ".csv"
        ds_train = IrisT5Dataset(datasets_folder / "iris" / str(perc) / train_string, device)
        test_string = "iris_test_perc" + str(perc) + ".csv"
        ds_test = IrisT5Dataset(datasets_folder / "iris" / str(perc) / test_string, device)
        
    elif dataset == 'abalone text-to-label':
        train_string = "abalone_train_perc" + str(perc) + ".csv"
        ds_train = AbaloneConcatDataset(datasets_folder / "abalone" / str(perc) / train_string, device)
        test_string = "abalone_test_perc" + str(perc) + ".csv"
        ds_test = AbaloneConcatDataset(datasets_folder / "abalone" / str(perc) / test_string, device)
    elif dataset == 'abalone text-to-text':
        train_string = "abalone_train_perc" + str(perc) + ".csv"
        ds_train = AbaloneT5Dataset(datasets_folder / "abalone" / str(perc) / train_string, device)
        test_string = "abalone_test_perc" + str(perc) + ".csv"
        ds_test = AbaloneT5Dataset(datasets_folder / "abalone" / str(perc) / test_string, device)
        
    elif dataset == 'adult text-to-label':
        train_string = "adult_train_perc" + str(perc) + ".csv"
        ds_train = AdultConcatDataset(datasets_folder / "adult" / str(perc) / train_string, device)
        test_string = "adult_test_perc" + str(perc) + ".csv"
        ds_test = AdultConcatDataset(datasets_folder / "adult" / str(perc) / test_string, device)
        test_string = "adult_test_perc" + str(perc) + ".csv"
    elif dataset == 'adult text-to-text':
        train_string = "adult_train_perc" + str(perc) + ".csv"
        ds_train = AdultT5Dataset(datasets_folder / "adult" / str(perc) / train_string, device)
        test_string = "adult_test_perc" + str(perc) + ".csv"
        ds_test = AdultT5Dataset(datasets_folder / "adult" / str(perc) / test_string, device)
       
    elif dataset == 'pulsar text-to-label':
        train_string = "pulsar_train_perc" + str(perc) + ".csv"
        ds_train = PulsarConcatDataset(datasets_folder / "pulsar" / str(perc) / train_string, device)
        test_string = "pulsar_test_perc" + str(perc) + ".csv"
        ds_test = PulsarConcatDataset(datasets_folder / "pulsar" / str(perc) / test_string, device)        
    elif dataset == 'pulsar text-to-text':
        train_string = "pulsar_train_perc" + str(perc) + ".csv"
        ds_train = PulsarT5Dataset(datasets_folder / "pulsar" / str(perc) / train_string, device)
        test_string = "pulsar_test_perc" + str(perc) + ".csv"
        ds_test = PulsarT5Dataset(datasets_folder / "pulsar" / str(perc) / test_string, device)
        
    elif dataset == 'nursery text-to-label':
        train_string = "nursery_train_perc" + str(perc) + ".csv"
        ds_train = NurseryConcatDataset(datasets_folder / "nursery" / str(perc) / train_string, device)
        test_string = "nursery_test_perc" + str(perc) + ".csv"
        ds_test = NurseryConcatDataset(datasets_folder / "nursery" / str(perc) / test_string, device)
    elif dataset == 'nursery text-to-text':
        train_string = "nursery_train_perc" + str(perc) + ".csv"
        ds_train = NurseryT5Dataset(datasets_folder / "nursery" / str(perc) / train_string, device)
        test_string = "nursery_test_perc" + str(perc) + ".csv"
        ds_test = NurseryT5Dataset(datasets_folder / "nursery" / str(perc) / test_string, device)
        
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

    # freeze options: 'ft', 'norm' or 'linear'
    model_ft, tokenizer = model_fn(ds_train.num_classes(), freeze='norm')
    model_ft.to(device)
    ds_train.tokenizer = tokenizer
    ds_test.tokenizer = tokenizer
    data_loader = SimpleDataLoaderBuilder(ds_train, ds_test)
    data_loader.build()
    tv = TrainAndValidate(data_loader, model_ft, device, num_epochs=50, learning_rate=1e-5)
    
    print('\nGPU:', torch.cuda.is_available())
    print('Device:', torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))
    print('\n')
    
    tv.train()
#     tv.validate(model_state='20220125-195921-iris-t5.pt')


if __name__ == '__main__':
    main()

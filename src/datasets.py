import torch
import numpy as np
from torch.utils.data import Dataset
from hashlib import md5
from inflect import engine


def _normalizer(inputs, max_len=10, device=None):
    complement = max_len - len(inputs)
    if complement <= 0:
        inputs = inputs[:max_len]
    else:
        inputs = torch.cat((inputs, torch.zeros(complement)))
    if device:
        inputs = inputs.to(device)
    return inputs


def concat_table_values(table_data, delimiter='  '):
    aux = [str(round(float(i), 2)) for i in table_data]
    result = delimiter.join(aux)
    return result


def hash_table_values(table_data, delimiter='  '):
    aux = [md5(str(round(float(i), 2)).encode()).hexdigest() for i in table_data]
    result = delimiter.join(aux)
    return result


def written_form_table_values(table_data, delimiter='  '):
    eng = engine()
    aux = [eng.number_to_words(str(round(float(i), 2))) for i in table_data]
    result = delimiter.join(aux)
    return result


class BaseDataset(Dataset):
    def __init__(self, src_file, root_dir, device, build_input_fn, max_encoded_len):
        self.data = np.loadtxt(src_file, delimiter=",", skiprows=0)
        self.root_dir = root_dir
        self.device = device
        self.tokenizer = None
        self.build_input_fn = build_input_fn
        self.max_encoded_len = max_encoded_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.tokenizer:
            raise AssertionError("Tokenizer should be set")

        if torch.is_tensor(idx):
            idx = idx.tolist()
        text = self.build_input_fn(self.data[idx, :-1])
        encoded_inputs = self.tokenizer.encode(text, return_tensors='pt', padding=True)
        encoded_inputs = torch.reshape(encoded_inputs, (-1,))
        encoded_inputs = _normalizer(encoded_inputs, max_len=self.max_encoded_len)

        outputs = torch.tensor(self.data[idx, -1:], dtype=torch.long)

        return encoded_inputs, outputs

    def name(self):
        raise NotImplementedError

    def num_classes(self):
        return len(set([row[-1] for row in self.data]))


class IrisConcatDataset(BaseDataset):
    def __init__(self,
                 src_file,
                 root_dir,
                 device,
                 build_input_fn=concat_table_values,
                 max_encoded_len=15
                 ):
        super().__init__(src_file, root_dir, device, build_input_fn, max_encoded_len)

    def name(self):
        return 'iris-concat'


class IrisWrittenDataset(BaseDataset):
    def __init__(self,
                 src_file,
                 root_dir,
                 device,
                 build_input_fn=written_form_table_values,
                 max_encoded_len=15
                 ):
        super().__init__(src_file, root_dir, device, build_input_fn, max_encoded_len)

    def name(self):
        return 'iris-written'


class AbaloneConcatDataset(BaseDataset):
    def __init__(self,
                 src_file,
                 root_dir,
                 device,
                 build_input_fn=concat_table_values,
                 max_encoded_len=30
                 ):
        super().__init__(src_file, root_dir, device, build_input_fn, max_encoded_len)

    def name(self):
        return 'abalone-concat'


class AbaloneWrittenDataset(BaseDataset):
    def __init__(self,
                 src_file,
                 root_dir,
                 device,
                 build_input_fn=written_form_table_values,
                 max_encoded_len=60
                 ):
        super().__init__(src_file, root_dir, device, build_input_fn, max_encoded_len)

    def name(self):
        return 'abalone-written'

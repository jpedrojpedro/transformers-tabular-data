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
        # [PAD] token is represented with index 0
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


class IrisT5Dataset(BaseDataset):
    def __init__(self,
                 src_file,
                 root_dir,
                 device,
                 max_encoded_len=128
                 ):
        super().__init__(src_file, root_dir, device, build_input_fn=None, max_encoded_len=max_encoded_len)

    def __getitem__(self, idx):
        if not self.tokenizer:
            raise AssertionError("Tokenizer should be set")

        task = 'multilabel classification:'
        features_numeric = self.data[idx, :-1]
        features_full = [f"{feature} {features_numeric[idx]}" for idx, feature in enumerate(self.features())]
        features_full = [task] + features_full
        text = '  '.join(features_full)

        encoded_inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_encoded_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        encoded_inputs_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask_inputs = encoded_inputs['attention_mask'].squeeze()

        label_numeric = self.data[idx, -1:]
        outputs = self.classes()[int(label_numeric[0])]
        encoded_outputs = self.tokenizer.encode_plus(
            outputs,
            max_length=self.max_encoded_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        encoded_outputs_ids = encoded_outputs['input_ids'].squeeze()
        attention_mask_outputs = encoded_outputs['attention_mask'].squeeze()

        final_inputs = {
            'encoded_inputs_ids': encoded_inputs_ids,
            'attention_mask_inputs': attention_mask_inputs,
        }
        final_outpus = {
            'encoded_outputs_ids': encoded_outputs_ids,
            'attention_mask_outputs': attention_mask_outputs
        }
        return final_inputs, final_outpus

    def name(self):
        return 'iris-t5'

    def classes(self):
        return [
            'setosa',
            'versicolour',
            'virginica',
        ]

    def features(self):
        return [
            'sepal length',
            'sepal width',
            'petal length',
            'petal width',
        ]


class IrisT5WrittenDataset(BaseDataset):
    def __init__(self,
                 src_file,
                 root_dir,
                 device,
                 max_encoded_len=128
                 ):
        super().__init__(src_file, root_dir, device, build_input_fn=None, max_encoded_len=max_encoded_len)

    def __getitem__(self, idx):
        if not self.tokenizer:
            raise AssertionError("Tokenizer should be set")

        task = 'multilabel classification:'
        features_numeric = self.data[idx, :-1]
        eng = engine()
        features_str = [eng.number_to_words(str(round(float(i), 2))) for i in features_numeric]
        features_full = [f"{feature} {features_str[idx]}" for idx, feature in enumerate(self.features())]
        features_full = [task] + features_full
        text = '  '.join(features_full)

        encoded_inputs = self.tokenizer.encode_plus(
            text,
            max_length=self.max_encoded_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )

        encoded_inputs_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask_inputs = encoded_inputs['attention_mask'].squeeze()

        label_numeric = self.data[idx, -1:]
        outputs = self.classes()[int(label_numeric[0])]
        encoded_outputs = self.tokenizer.encode_plus(
            outputs,
            max_length=self.max_encoded_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt'
        )
        encoded_outputs_ids = encoded_outputs['input_ids'].squeeze()
        attention_mask_outputs = encoded_outputs['attention_mask'].squeeze()

        final_inputs = {
            'encoded_inputs_ids': encoded_inputs_ids,
            'attention_mask_inputs': attention_mask_inputs,
        }
        final_outpus = {
            'encoded_outputs_ids': encoded_outputs_ids,
            'attention_mask_outputs': attention_mask_outputs
        }
        return final_inputs, final_outpus

    def name(self):
        return 'iris-t5-written'

    def classes(self):
        return [
            'setosa',
            'versicolour',
            'virginica',
        ]

    def features(self):
        return [
            'sepal length',
            'sepal width',
            'petal length',
            'petal width',
        ]


class IrisConcatDataset(BaseDataset):
    def __init__(self,
                 src_file,
                 root_dir,
                 device,
                 build_input_fn=concat_table_values,
                 max_encoded_len=64
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
                 max_encoded_len=64
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
                 max_encoded_len=128
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
                 max_encoded_len=128
                 ):
        super().__init__(src_file, root_dir, device, build_input_fn, max_encoded_len)

    def name(self):
        return 'abalone-written'


class AbaloneT5Dataset(IrisT5Dataset):
    def __init__(self,
                 src_file,
                 root_dir,
                 device,
                 max_encoded_len=128
                 ):
        super().__init__(src_file, root_dir, device, max_encoded_len=max_encoded_len)

    def name(self):
        return 'abalone-t5'

    def classes(self):
        return [
            '1 year old',
            '2 years old',
            '3 years old',
            '4 years old',
            '5 years old',
            '6 years old',
            '7 years old',
            '8 years old',
            '9 years old',
            '10 years old',
            '11 years old',
            '12 years old',
            '13 years old',
            '14 years old',
            '15 years old',
            '16 years old',
            '17 years old',
            '18 years old',
            '19 years old',
            '20 years old',
            '21 years old',
            '22 years old',
            '23 years old',
            '24 years old',
            '25 years old',
            '26 years old',
            '27 years old',
            '29 years old',
        ]

    def features(self):
        return [
            'sex',
            'length',
            'diameter',
            'height',
            'whole weight',
            'shucked weight',
            'viscera weight',
            'shell weight',
            # 'rings',  => value to be predicted
        ]

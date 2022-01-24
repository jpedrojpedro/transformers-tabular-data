from abc import ABC
import torch
import numpy as np
from torch.utils.data import Dataset
from hashlib import md5
from inflect import engine
from random import shuffle


########## ------------ AUXILIAR FUNCTIONS ------------ ##########

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


def create_intervals(aux, col_min, col_max, col_names):
    new_aux = []
    for i, value in enumerate(aux):
        diff = col_max[int(i)] - col_min[int(i)]
        factor = diff/5
        num = int(value)
        if num < col_min[i] + factor:
            c = 'very low'
        elif (num > col_min[i] + factor) & (num < col_min[i] + 2*factor):
            c = 'little low'
        elif (num > col_min[i] + 2*factor) & (num < col_min[i] + 3*factor):
            c = 'in average'
        elif (num > col_min[i] + 3*factor) & (num < col_min[i] + 4*factor):
            c = 'little high'
        else:
            c = 'very high'        
        
        d = col_names[i] + ': ' + aux[i] + ' (' + c + ')'
        new_aux.append(d)
        
    shuffle(new_aux)
    delimiter = ', '
    return delimiter.join(new_aux)


def concat_table_values(table_data, delimiter='  '):
#     aux = [str(int(round(float(i), 2)*10)) for i in table_data]
    aux = [str(i) for i in table_data]

    # col_min = [43, 20, 10, 1]
    # col_max = [79, 44, 69, 25]
    # col_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
    # result = create_intervals(aux, col_min, col_max, col_names)

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


########## ------------ __getitem__ implementations ------------ ##########

def getitem_text_to_label(idx, data, tokenizer, max_encoded_len, classes=None, features=None):
    if torch.is_tensor(idx):
        idx = idx.tolist()
    
    text = concat_table_values(data[idx])[:-1].strip()
#     print('\n\nText:', text)
    encoded_inputs = tokenizer.encode(text, return_tensors='pt', padding=True)
#     print('Encoded', encoded_inputs)
    decoded_inputs = [tokenizer.decode(i) for i in encoded_inputs]
#     print('Decoded:', decoded_inputs)
    encoded_inputs = torch.reshape(encoded_inputs, (-1,))
    encoded_inputs = _normalizer(encoded_inputs, max_len=max_encoded_len)
#     print('Normalized:', encoded_inputs)
    if classes:
        try:
            outputs = torch.tensor(classes()[data[idx][-1].strip()], dtype=torch.long)
        except:
            outputs = torch.tensor(classes()[data[idx][-1]], dtype=torch.long)
    else:
        outputs = torch.tensor(data[idx][-1], dtype=torch.long)

    return encoded_inputs, outputs


def getitem_text_to_text(idx, data, tokenizer, max_encoded_len, classes, features):
    task = 'multilabel classification:'
        
    features_numeric = data[idx].tolist()[:-1]
    features_full = [f"{feature} {features_numeric[idx]}" for idx, feature in enumerate(features())]
#     features_full = [f"{feature} {int(features_numeric[idx] * 10)}" for idx, feature in enumerate(features())]
    features_full = [task] + features_full
    text = '  '.join(features_full)

    encoded_inputs = tokenizer.encode_plus(
        text,
        max_length=max_encoded_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt'
    )

    encoded_inputs_ids = encoded_inputs['input_ids'].squeeze()
    attention_mask_inputs = encoded_inputs['attention_mask'].squeeze()

    label_numeric = data[idx].tolist()[-1:][0]
  
    if type(label_numeric) == str:
        outputs = label_numeric.strip()
    else:        
        outputs = classes()[int(label_numeric)]
        
        
    encoded_outputs = tokenizer.encode_plus(
        outputs,
        max_length=max_encoded_len,
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


########## ------------ GENERIC CLASS TO HANDLE DATASETS ------------ ##########

class BaseDataset(Dataset):
    def __init__(self, getitem_fn, src_file, device, max_encoded_len):
        self.getitem_fn = getitem_fn
        self.data = np.genfromtxt(src_file, dtype=None, delimiter=",", encoding='utf-8', skip_header=0, names=True)
        self.device = device
        self.tokenizer = None
        self.max_encoded_len = max_encoded_len

    def __getitem__(self, index):
        if not self.tokenizer:
            raise AssertionError("Tokenizer should be set")
        return self.getitem_fn(index, self.data, self.tokenizer, self.max_encoded_len, self.classes, self.features)

    def __len__(self):
        return len(self.data)

    def name(self):
        raise NotImplementedError

    def num_classes(self):
        return len(set([row[-1] for row in self.data]))

    def max_min_column(self, col):
        return max([row[col] for row in self.data]), min([row[col] for row in self.data])
    
    
########## ------------ DATASETS CLASSES ------------ ##########


### --- IRIS --- ###

class IrisConcatDataset(BaseDataset):
    def __init__(self,
                 src_file,
                 device,
                 getitem_fn=getitem_text_to_label,
                 max_encoded_len=10
                 ):
        super().__init__(getitem_fn, src_file, device, max_encoded_len)
        # FIXME: same name for method and attribute
        self.classes = None
        self.features = None

    def name(self):
        return 'iris-concat'
    
    
class IrisT5Dataset(BaseDataset):
    def __init__(self,
                 src_file,
                 device,
                 getitem_fn=getitem_text_to_text,
                 max_encoded_len=40
                 ):
        super().__init__(getitem_fn, src_file, device, max_encoded_len)

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

    
### --- ABALONE --- ###

class AbaloneConcatDataset(BaseDataset):
    def __init__(self,
                 src_file,
                 device,
                 getitem_fn=getitem_text_to_label,
                 max_encoded_len=64
                 ):
        super().__init__(getitem_fn, src_file, device, max_encoded_len)
        self.features = None
        
    def name(self):
        return 'abalone-concat'

    def num_classes(self):
        return len(self.classes())

    def classes(self):
        return {
            3: 0,
            4: 1,
            5: 2,
            6: 3,
            7: 4,
            8: 5,
            9: 6,
            10: 7,
            11: 8,
            12: 9,
            13: 10,
            14: 11,
            15: 12,
            16: 13,
            17: 14,
            18: 15,
            19: 16,
            20: 17,
            21: 18,
            22: 19,
            23: 20,
        }


class AbaloneT5Dataset(BaseDataset):
    def __init__(self,
                 src_file,
                 device,
                 getitem_fn=getitem_text_to_text,
                 max_encoded_len=128
                 ):
        super().__init__(getitem_fn, src_file, device, max_encoded_len)

    def name(self):
        return 'abalone-t5'

    def classes(self):
        return {
            3: '3 years old',
            4: '4 years old',
            5: '5 years old',
            6: '6 years old',
            7: '7 years old',
            8: '8 years old',
            9: '9 years old',
            10: '10 years old',
            11: '11 years old',
            12: '12 years old',
            13: '13 years old',
            14: '14 years old',
            15: '15 years old',
            16: '16 years old',
            17: '17 years old',
            18: '18 years old',
            19: '19 years old',
            20: '20 years old',
            21: '21 years old',
            22: '22 years old',
            23: '23 years old',
        }

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


### --- ADULT --- ###

class AdultConcatDataset(BaseDataset):
    def __init__(self,
                 src_file,
                 device,
                 getitem_fn=getitem_text_to_label,
                 max_encoded_len=64
                 ):
        super().__init__(getitem_fn, src_file, device, max_encoded_len)

        self.features = None
        
    def classes(self):
        return {
            "<=50K": 0,
            ">50K": 1
        }
    
    def name(self):
        return 'adult-concat'

    
class AdultT5Dataset(BaseDataset):
    def __init__(self,
                 src_file,
                 device,
                 getitem_fn=getitem_text_to_text,
                 max_encoded_len=128
                 ):
        super().__init__(getitem_fn, src_file, device, max_encoded_len)

    def name(self):
        return 'adult-t5'


    def classes(self):
        return {
            0: " <=50K",
            1: " >50K"
        }
    
    def features(self):
        return [
            'age',
            'workclass',
            'fnlwgt',
            'education',
            'education-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country'
        ]
    
    


### --- PULSAR --- ###
    
class PulsarConcatDataset(BaseDataset):
    def __init__(self,
                 src_file,
                 device,
                 getitem_fn=getitem_text_to_label,
                 max_encoded_len=64
                 ):
        super().__init__(getitem_fn, src_file, device, max_encoded_len)

        self.classes = None
        self.features = None

    def name(self):
        return 'pulsar-concat'
    
    
class PulsarT5Dataset(BaseDataset):
    def __init__(self,
                 src_file,
                 device,
                 getitem_fn=getitem_text_to_text,
                 max_encoded_len=128
                 ):
        super().__init__(getitem_fn, src_file, device, max_encoded_len)

    def name(self):
        return 'pulsar-t5'

    def classes(self):
        return [
            '0',
            '1',
        ]

    def features(self):
        return [
            ' Mean of the integrated profile',
            ' Standard deviation of the integrated profile',
            ' Excess kurtosis of the integrated profile',
            ' Skewness of the integrated profile', 
            ' Mean of the DM-SNR curve',
            ' Standard deviation of the DM-SNR curve',
            ' Excess kurtosis of the DM-SNR curve', 
            ' Skewness of the DM-SNR curve'
        ]


import torch
import numpy as np
from torch.utils.data import Dataset


class IrisDataset(Dataset):
    def __init__(self, src_file, root_dir, device, both_ids=False):
#     def __init__(self, src_file, root_dir, both_ids=False):
        # data like: 5.0, 3.5, 1.3, 0.3, 0
        self.data = np.loadtxt(src_file, delimiter=",", skiprows=0)
        self.root_dir = root_dir
        self.device = device
        self.both_ids = both_ids
        self.tokenizer = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.tokenizer:
            raise AssertionError("Tokenizer should be set")
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
            
        text = '  '.join([str(round(float(i), 2)) for k, i in enumerate(self.data[idx, 0:-1])])        
        
        encoded_inputs = self.tokenizer.encode(text, return_tensors='pt', padding=True)

        encoded_inputs = torch.reshape(encoded_inputs, (-1,))
        encoded_inputs = self._normalizer(encoded_inputs)

        outputs = torch.tensor(self.data[idx, -1:], dtype=torch.long)

        return encoded_inputs, outputs

    def _normalizer(self, inputs, max_len=10):
        complement = max_len - len(inputs)
        if complement <= 0:
            inputs = inputs[:max_len]
        else:
            inputs = torch.cat((inputs, torch.zeros(complement)))
#         inputs = inputs.to(self.device)
        return inputs

    def num_classes(self):
        return len(set([row[-1] for row in self.data]))


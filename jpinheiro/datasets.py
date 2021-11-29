import torch
import numpy as np
from torch.utils.data import Dataset


class IrisDataset(Dataset):
    def __init__(self, src_file, root_dir, device, both_ids=False):
        # data like: 5.0, 3.5, 1.3, 0.3, 0
        self.data = np.loadtxt(src_file, usecols=range(0, 5), delimiter=",", skiprows=0, dtype=np.float32)
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
        # inputs = torch.from_numpy(self.data[idx, 0:4]).to(self.device)
        encoded_inputs = self.tokenizer.encode(
            '  '.join([str(round(float(i), 2)) for i in self.data[idx, 0:4]]), return_tensors='pt', padding=True
        )
        encoded_inputs = torch.reshape(encoded_inputs, (-1,))
        encoded_inputs = self._normalizer(encoded_inputs)

        outputs = torch.tensor(self.data[idx, 4:], dtype=torch.long).to(self.device)
        # encoded_outputs = self.tokenizer.encode(
        #     '  '.join([str(int(i)) for i in self.data[idx, 4:]]), return_tensors='pt', padding=True
        # )
        # encoded_outputs = torch.reshape(encoded_outputs, (-1,))
        # encoded_outputs = self._normalizer(encoded_outputs, max_len=3)

        return encoded_inputs, outputs

    def _normalizer(self, inputs, max_len=10):
        complement = max_len - len(inputs)
        if complement <= 0:
            inputs = inputs[:max_len]
        else:
            inputs = torch.cat((inputs, torch.zeros(complement)))
        inputs = inputs.to(self.device)
        return inputs

    def class_names(self):
        # FIXME: generate automatically
        return ['setosa', 'versicolor', 'virginica']

    def num_classes(self):
        # FIXME: generate automatically
        return 3

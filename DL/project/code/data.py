import os
import random
from typing import List, Dict
import numpy as np
import torch


class Transformer:
    """Base transformer class"""
    def __call__(self, data):
        """Apply the transformation.

        Args:
            data (array like): Input to transform

        Returns:
            array like: The transformed input
        """
        return data


class Normalizer(Transformer):
    """Normalize data.

    X' = (X' - mean)/std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        return (data - self.mean)/self.std

    def reverse(self, data):
        return data * self.std + self.mean


class NinaProDataset(torch.utils.data.Dataset):
    """Basic dataset for a NinaPro Database.

    Load the Ninapro database in RAM from the disk. (After the preprocessing step)

    Attrs:
        METADATA (List[str]): List of metadata key to keep
        DATA (List[str]): List of data key to keep
    """
    METADATA = ["subject", "exercise"]
    DATA = ["emg", "stimulus", "glove"]

    def __init__(self, data_folder: str):
        """Constructor:

        Args:
            data_folder (str): The directory where the mat files are stored
        """
        self.data: List[Dict]
        self.size: List[int]

        self._load(data_folder)

        assert len(self.size) > 0, "No data has been loaded."

    def _load(self, data_folder: str):
        self.data = []
        self.size = []

        files = os.listdir(data_folder)
        for file in sorted(filter(lambda file: file[-4:] == ".npz", files)):
            print(file)
            npz_file = np.load(os.path.join(data_folder, file))

            self.data.append({
                key: npz_file[key] for key in self.DATA + self.METADATA
            })

            self.size.append(len(self.data[-1][self.DATA.copy().pop()]))

    def __getitem__(self, idx):
        cum_size = 0
        for i, size in enumerate(self.size):
            if idx < cum_size + size:
                break
            cum_size += size
        idx -= cum_size
        return [self.data[i][key][idx] for key in self.DATA]

    def __len__(self):
        return sum(self.size)

    def transform(self, transformer: Dict[str, Transformer]):
        """Apply a dictionnary of transformer to the data.

        Valid key: self.DATA

        Args:
            transformer (Dict[Transformer]): Apply transformer[key] to data[key]
        """
        for key in transformer:
            for file in self.data:
                file[key] = transformer[key](file[key])


class SequenceDataset(NinaProDataset):
    """Sequence dataset over a NinaProDataset for RNN training

    Each item is a sequence extracted from one of the exercise: For each exercise
    of N time steps, then sequences extracted are:
        [i: i + seq_len] for 0 <= i < N - seq_len

    If seq_len is None, then the sequences simply are the total exercise sequence (for testing purpose)
    """
    DATA = ["emg", "stimulus", "glove"]

    def __init__(self, data_folder: str, seq_len: int = None):
        """Constructor:

        Args:
            data_folder (str): The directory where the Dataset is stored
            seq_len (int): The length of sequences to extract in seconds
        """
        super().__init__(data_folder)
        self.seq_len = seq_len
        if self.seq_len is None:  # One sequence by exercise
            self.size = [1 for size in self.size]
        else:  # Otherwise just take inbound sequences.
            self.size = [size - self.seq_len for size in self.size]

    def __getitem__(self, idx):
        if self.seq_len is None:
            return [self.data[idx][key] for key in self.DATA]

        cum_size = 0
        for i, size in enumerate(self.size):
            if idx < cum_size + size:
                break
            cum_size += size
        idx -= cum_size

        return [self.data[i][key][idx: idx + self.seq_len] for key in self.DATA]

    def random_split(self):
        """Basic random splitter.

        Useful to train on some user and test over the same user.
        """
        step = 54 * 400 // 2  # ~ 54seconds for each ex.

        train_idx = []
        test_idx = []
        cum_size = 0
        for s in self.size:
            i = cum_size
            cum_size += s
            while i < cum_size:
                train = (random.random() > 0.5)
                j = min(i + step, cum_size)
                k = min(j + step, cum_size)
                if train:
                    train_idx.extend(range(i, j - self.seq_len))
                    test_idx.extend(range(j, k - self.seq_len))
                else:
                    train_idx.extend(range(j, k - self.seq_len))
                    test_idx.extend(range(i, j - self.seq_len))
                i = k

        return SubSequenceDataset(self, train_idx), SubSequenceDataset(self, test_idx)


class SubSequenceDataset(torch.utils.data.Dataset):
    """Contains only a part of the sequences of SequenceDataset"""
    def __init__(self, dataset, idx):
        super().__init__()
        self.idx = idx
        self.dataset = dataset

    def __getitem__(self, idx):
        return self.dataset[self.idx[idx]]

    def __len__(self):
        return len(self.idx)

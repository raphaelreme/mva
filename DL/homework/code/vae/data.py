import torch
from torchvision import datasets
from torchvision import transforms


class AutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, idx):
        data, _ = self.dataset[idx]
        return (data, data)

    def __len__(self):
        return len(self.dataset)


def load_datasets(root):
    train_set = datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST(root, train=False, download=True, transform=transforms.ToTensor())

    return AutoEncoderDataset(train_set), AutoEncoderDataset(test_set)

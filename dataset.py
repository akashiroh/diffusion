# Make a dataset class for the mnist dataset

import torch
import torchvision
from torch.utils.data import Dataset

class MNISTDataset(Dataset):
    def __init__(self, train_dev_test, transform):

        if train_dev_test == 'train':
            self.data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        else:
            self.data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return (img, label)

    def __len__(self):
        return len(self.data)

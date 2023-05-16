from dataset import MNISTDataset
from model import DiffusionModel
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch

def load_data():
    #transform
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # dataset
    train_dataset = MNISTDataset('train', transform)
    dev_dataset = MNISTDataset('dev', transform)

    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=64, shuffle=True)

    return train_dataset, dev_dataset, train_dataloader, dev_dataloader

def train(model, train_dataloader, dev_dataloader, optimizer, criterion):
    # train loop
    for epoch in range(10):
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # dev loop
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dev_dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()




def main():
    _, _, train_datalaoder, dev_dataloader = load_data()
    model = DiffusionModel(784, 128, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train(model, train_datalaoder, dev_dataloader, optimizer, criterion)

    return

if __name__ == '__main__':
    main()
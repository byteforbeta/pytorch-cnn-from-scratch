import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def dataprep(path):

    basic_transform = transforms.ToTensor() # scales the raw pixels from 0-255 down to 0.0-1.0 so our math is stable.

    train_set = torchvision.datasets.FashionMNIST(
        root=path, 
        train=True, 
        transform=basic_transform, 
        download=True
    )

    loader = DataLoader(train_set, batch_size=64, shuffle=False)

    
    channel_sum = torch.zeros(1) # 1- grey scale and 3 - RGB imgs
    channel_sq_sum = torch.zeros(1)
    total_batches = 0

    # Iterate through the dataset
    for images, labels in loader:
        channel_sum += torch.mean(images, dim=[0, 2, 3])
        channel_sq_sum += torch.mean(images ** 2, dim=[0, 2, 3])
        
        total_batches += 1


    mean = channel_sum / total_batches
    variance = (channel_sq_sum / total_batches) - (mean ** 2)
    std = torch.sqrt(variance)

    # Standard Transform
    test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((mean.item(),), (std.item(),)) # Centers the data around zero
    ])

    # Augmented Transform
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5), # 50% chance to flip the image
        transforms.ToTensor(),
        transforms.Normalize((mean.item(),), (std.item(),))
    ])

    train_set = torchvision.datasets.FashionMNIST(root=path, train=True, transform=train_transform, download=True)
    test_set = torchvision.datasets.FashionMNIST(root=path, train=False, transform=test_transform, download=True)

    return train_set, test_set



if __name__=='__main__':
    train_set, test_set = dataprep(path)
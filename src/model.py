import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CnnNet(nn.Module):
    def __init__(self, in_channels, hidden_size, num_classes, dropout_rate):
        super(CnnNet,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.dropout = nn.Dropout(p=dropout_rate)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=7*7*64, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        p1 = self.pool(torch.relu(self.conv1(x)))
        p2 = self.pool(torch.relu(self.conv2(p1)))

        # Flatten 3D - 1D vector
        f = torch.flatten(p2, start_dim=1)

        fc1_out = torch.relu(self.fc1(f))
        dropped_out = self.dropout(fc1_out)
        prediction = self.fc2(dropped_out)
        
        return prediction

if __name__=='__main__':
    model = CnnNet()
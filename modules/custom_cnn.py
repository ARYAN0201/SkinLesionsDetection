import torch
import torch.nn as nn
import torch.nn.functional as F


class Custom_CNN(nn.Module):
    def __init__(self, input_channels = 3, num_classes=7, dropout = 0.25):
        super(Custom_CNN, self).__init__()

        # First Conv Layer: 16 kernels, kernel_size=3, padding=1 
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)

        # Second layer: maxpool 2x2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(dropout)

        # Third Conv layer: 32 kernels, kernel_size=3, padding=1 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Fourth layer: maxpool 2x2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(dropout)

        # Fifth conv layer: 64 kernels, kernel_size=3, padding=1 
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # Sixth layer: maxpool 2x2
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(dropout)
                # Fully connected layer input size calculated for 224x224 input images
        self.fc = nn.Linear(64 * 28 * 28, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop2(x)

        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.drop3(x)

        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)

        return x
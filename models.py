import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, hidden=(512, 256), dropout=0.0, input_size=32 * 32 * 3):
        super().__init__()
        sizes = [input_size] + list(hidden)
        self.layers = nn.ModuleList()
        for i in range(len(hidden)):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        self.out = nn.Linear(sizes[-1], 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        return self.out(x)


class CNN(nn.Module):
    # 3 blocks of conv 3x3 -> batch norm -> relu -> max pool, then 2 fully connected layers
    def __init__(self, width=32, hidden=256, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, 2 * width, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * width)
        self.conv3 = nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * width)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4 * width * 4 * 4, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4
        x = x.flatten(1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

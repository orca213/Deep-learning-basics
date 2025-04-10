import torch.nn as nn
import torch.nn.functional as F

class GenreCNN(nn.Module):
    def __init__(self, num_classes):
        super(GenreCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # output shape fixed
        self.fc1 = nn.Linear(32 * 4 * 4, 128)  # matches adaptive pooled shape
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> (B, 16, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))  # -> (B, 32, H/4, W/4)
        x = self.adaptive_pool(x)             # -> (B, 32, 4, 4)
        x = x.view(x.size(0), -1)             # -> (B, 512)
        x = self.dropout(F.relu(self.fc1(x))) # -> (B, 128)
        x = self.fc2(x)                       # -> (B, num_classes)
        return x


class GenreCNN_v2(nn.Module):
    def __init__(self, num_classes):
        super(GenreCNN_v2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # output shape fixed
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # -> (B, 32, H/2, W/2)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # -> (B, 64, H/4, W/4)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # -> (B, 128, H/8, W/8)
        x = self.adaptive_pool(x)                       # -> (B, 128, 4, 4)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

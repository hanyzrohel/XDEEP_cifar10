import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # --- Feature extractor ---
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.act2 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(2, 2)

        # --- Classifier ---
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.act_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.act1(self.conv1(x)))  # -> (32, 16, 16)
        x = self.pool(self.act2(self.conv2(x)))  # -> (64, 8, 8)

        x = x.view(x.size(0), -1)              # flatten
        x = self.act_fc1(self.fc1(x))
        x = self.fc2(x)
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

"""

class SimpleBrainTumorCNN(nn.Module):
    def __init__(self):
        super(SimpleBrainTumorCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 30 * 30, 128)  # Après 3 pools, 240/2/2/2 = 30
        self.fc2 = nn.Linear(128, 2)  # 2 classes : MGMT methylated or not

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))

        x = torch.flatten(x, 1)  # Flatten toutes les dimensions sauf batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBrainTumorCNN(nn.Module):
    def __init__(self):
        super(SimpleBrainTumorCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=20, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 30 * 30, 128)  # 240 → 120 → 60 → 30
        self.fc2 = nn.Linear(128, 2)  # 2 classes : methylated / unmethylated

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

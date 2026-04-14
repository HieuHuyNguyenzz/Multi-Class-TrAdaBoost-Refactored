import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import NUM_CLASSES

class CNNModel(nn.Module):
    def __init__(self, input_shape, num_classes=NUM_CLASSES):
        """
        CNN Architecture for traffic classification.
        
        Args:
            input_shape (tuple): (packet_num, num_features)
            num_classes (int): Number of target classes.
        """
        super(CNNModel, self).__init__()
        
        # input_shape = (20, 256)
        # Conv Layers
        self.conv1 = nn.Conv2d(1, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        
        self.flatten_dim = self._get_flatten_dim(input_shape)
        
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256, num_classes)
        
    def _get_flatten_dim(self, input_shape):
        with torch.no_grad():
            # dummy input: (batch, channel, height, width)
            x = torch.zeros(1, 1, *input_shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.pool(F.relu(self.conv4(x)))
            x = F.relu(self.conv5(x))
            x = self.pool(F.relu(self.conv6(x)))
            return x.numel()
            
    def forward(self, x):
        # Add channel dimension if missing: (batch, h, w) -> (batch, 1, h, w)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

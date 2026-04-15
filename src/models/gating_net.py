import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import NUM_CLASSES, DEVICE

class GatingNetwork(nn.Module):
    def __init__(self, input_shape, num_learners):
        """
        CNN-based Gating Network to select top-k weak learners.
        
        Args:
            input_shape (tuple): (packet_num, num_features)
            num_learners (int): Number of weak learners T in AdaBoost.
        """
        super(GatingNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.flatten_dim = self._get_flatten_dim(input_shape)
        
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_learners)
        
    def _get_flatten_dim(self, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, 1, *input_shape)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            return x.numel()
            
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

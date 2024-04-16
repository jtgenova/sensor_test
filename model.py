import torch.nn as nn
import torch.nn.functional as F
import torch

class MultiOutputCNN(nn.Module):
    def __init__(self):
        super(MultiOutputCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Task-specific layers
        self.fc_segmentation = nn.Linear(64 * 128, 1024)  # Output layer for localization (segmentation-like task)
        self.fc_width = nn.Linear(64 * 128, 1)  # Output layer for width estimation
        self.fc_depth = nn.Linear(64 * 128, 1)  # Output layer for depth estimation


    def forward(self, x):
        # x = x.unsqueeze(1)  # Ensure channel dimension is pres
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 128)  # Corrected flatten operation
        x = self.dropout(x)
        
        # Obtain outputs for each task
        segmentation = self.fc_segmentation(x)
        width = self.fc_width(x)
        depth = self.fc_depth(x)

        return segmentation, width, depth


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Convert logits to probabilities
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

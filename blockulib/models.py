import torch
import torch.nn as nn

import torch.nn.functional as F

class CustomPadConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1):
        super(CustomPadConv2d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0)
    
    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=-1)
        return self.conv(x)
    
class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(), 
        nn.Linear(32 * 9 * 9, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
        
    def forward(self, x):
        return self.model(x)
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
    )
        self.SkipC = nn.Sequential(
        nn.Flatten(),
        nn.Linear(9*9, 64)
    )
        self.head = nn.Linear(64, 1)
    def forward(self, x):
        y1 = self.model(x)
        y2 = self.SkipC(x)
        return self.head(y1 + y2)
    
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(9*9, 1)
    )
        
    def forward(self, x):
        return self.model(x)
    
class TrikowyModel(nn.Module):
    def __init__(self):
        super(TrikowyModel, self).__init__()
        self.pipe1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3), # (1, 9, 9) -> (2, 7, 7)
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3), # (2, 7, 7) -> (4, 5, 5)
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(4*5*5, 7)
    )
        self.pipe2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=4), # (1, 9, 9) -> (2, 6, 6)
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3), # (2, 5, 5) -> (4, 4, 4)
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(4*4*4, 7)
    )
        self.pipe3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5), # (1, 9, 9) -> (2, 5, 5)
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3), # (2, 5, 5) -> (4, 3, 3)
            nn.Flatten(),
            nn.GELU(),
            nn.Linear(4*3*3, 7)
    )
        self.final = nn.Sequential(
            nn.GELU(),
            nn.Linear(21, 1)
        )
        
        
    def forward(self, x):
        A = self.pipe1(x)
        B = self.pipe2(x)
        C = self.pipe3(x)
        inp = torch.cat([A, B, C], dim=1) 
        return self.final(inp)
    

        
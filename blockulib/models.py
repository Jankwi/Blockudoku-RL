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
    
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(9*9, 1),
    )
        
    def forward(self, x):
        return self.model(x)   
    
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.model = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride = 3, padding=0),
        nn.Flatten(),
        nn.ReLU(),
        nn.Linear(4*3*3, 10),
        nn.GELU(),
        nn.Linear(10, 1),
        nn.Tanh()
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


# class EmbeddedConvModel(nn.Module):
#     def __init__(self, emb_dim=8):
#         super(EmbeddedConvModel, self).__init__()
#         # Embedding layer that maps each board value (0 or 1) to a dense vector.
#         self.embedding = nn.Embedding(num_embeddings=2, embedding_dim=emb_dim)
        
#         # Convolution branch:
#         # After embedding, the board shape is (batch, 9, 9, emb_dim).
#         # We permute the dimensions to (batch, emb_dim, 9, 9) for convolution.
#         self.conv_branch = nn.Sequential(
#             nn.Conv2d(in_channels=emb_dim, out_channels=16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Flatten(),  # Output will be (32 * 9 * 9)
#             nn.Linear(32 * 9 * 9, 64),
#             nn.ReLU()
#         )
        
#         # Skip branch:
#         # Flattens the embedded board (without convolution) directly and linearly transforms it.
#         self.skip_branch = nn.Sequential(
#             nn.Flatten(),  # Shape: (batch, 9*9*emb_dim)
#             nn.Linear(9 * 9 * emb_dim, 64),
#             nn.ReLU()
#         )
        
#         # Final head: Combines the outputs of both pipelines.
#         self.head = nn.Linear(64, 1)
    
#     def forward(self, x):
#         # x should be a LongTensor with shape (batch, 9, 9) containing only 0's and 1's.
#         # Convert the board values to dense embeddings.
#         x_emb = self.embedding(x)  # Shape: (batch, 9, 9, emb_dim)
        
#         # Rearrange dimensions to (batch, emb_dim, 9, 9) for the convolutional layers.
#         x_emb = x_emb.permute(0, 3, 1, 2)
        
#         # Process through the convolution branch.
#         out_conv = self.conv_branch(x_emb)
        
#         # Process through the skip branch: flatten directly the embedded board.
#         skip_in = x_emb.flatten(start_dim=1)  # Shape: (batch, 9*9*emb_dim)
#         out_skip = self.skip_branch(skip_in)
        
#         # Combine the outputs of both branches elementwise.
#         combined = out_conv + out_skip
        
#         # Final scalar output.
#         return self.head(combined)

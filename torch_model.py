import torch
from torch import nn
import torchvision.transforms as transforms

# simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self, input_shape: int = 784, output_shape: int = 10):
        super().__init__()
        # Convert input_shape (e.g., 784 for 28x28 images) to 2D shape
        # Input is 28x28 grayscale image (784 = 28*28)
        self.input_shape = (1, 28, 28)  # (channels, height, width)

        self.layer_stack = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, output_shape)
        )
  
    def forward(self, x):
        return self.layer_stack(x)

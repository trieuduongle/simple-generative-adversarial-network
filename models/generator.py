import torch
from torch import nn

# Generator Network
class Generator(nn.Sequential):
    def __init__(self, sample_size: int):
        super().__init__(
            nn.Linear(sample_size, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 784),
            nn.Sigmoid())

        # Random value vector size
        self.sample_size = sample_size

    def forward(self, batch_size: int):
        # Generate randon values
        z = torch.randn(batch_size, self.sample_size)

        # Generator output
        output = super().forward(z)

        # Convert the output into a greyscale image (1x28x28)
        generated_images = output.reshape(batch_size, 1, 28, 28)
        return generated_images

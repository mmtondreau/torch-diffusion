import torch.nn as nn
import pytorch_lightning as pl
from torch_diffusion.model.residual_conv_block import ResidualConvBlock


class UnetDown(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, down_scale=2):
        super(UnetDown, self).__init__()
        self.output_dim = out_channels
        self.input_dim = in_channels

        # Create a list of layers for the downsampling block
        # Each block consists of two ResidualConvBlock layers, followed by a MaxPool2d layer for downsampling
        layers = [
            ResidualConvBlock(in_channels, out_channels, kernel_size=kernel_size),
            ResidualConvBlock(out_channels, out_channels, kernel_size=kernel_size),
            nn.MaxPool2d(down_scale),
            nn.Dropout(0.2),
        ]

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Pass the input through the sequential model and return the output
        return self.model(x)

import torch.nn as nn
import pytorch_lightning as pl
import torch
from torch_diffusion.model.residual_conv_block import ResidualConvBlock


class UnetUp(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, upscale=2):
        super(UnetUp, self).__init__()

        # Create a list of layers for the upsampling block
        # The block consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, upscale, upscale),
            nn.Dropout(0.2),
            ResidualConvBlock(out_channels, out_channels, kernel_size=kernel_size),
            ResidualConvBlock(out_channels, out_channels, kernel_size=kernel_size),
        ]

        # Use the layers to create a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        # Concatenate the input tensor x with the skip connection tensor along the channel dimension
        x = torch.cat((x, skip), 1)

        # Pass the concatenated tensor through the sequential model and return the output
        x = self.model(x)
        return x

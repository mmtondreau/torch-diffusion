import torch.nn as nn
import pytorch_lightning as pl
import torch


class ResidualConvBlock(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_res: bool = False,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        # Check if input and output channels are the same for the residual connection
        self.same_channels = in_channels == out_channels

        # Flag for whether or not to use residual connection
        self.is_res = is_res
        if (kernel_size - 1) % 2 != 0:
            raise ValueError("kernel size must be odd")

        padding = (kernel_size - 1) // 2
        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=padding
            ),  # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.GELU(),  # GELU activation function
        )

        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels, out_channels, kernel_size, stride=1, padding=padding
            ),  # 3x3 kernel with stride 1 and padding 1
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.GELU(),  # GELU activation function
        )

        # if self.is_res and not self.same_channels:
        #     self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If using residual connection
        if self.is_res:
            # Apply first convolutional layer
            x1 = self.conv1(x)

            # Apply second convolutional layer
            x2 = self.conv2(x1)

            # If input and output channels are the same, add residual connection directly
            if self.same_channels:
                out = x + x2
            else:
                # If not, apply a 1x1 convolutional layer to match dimensions before adding residual connection
                shortcut = nn.Conv2d(
                    x.shape[1], x2.shape[1], kernel_size=1, stride=1, padding=0
                ).to(self.device)
                out = shortcut(x) + x2
            # print(f"resconv forward: x {x.shape}, x1 {x1.shape}, x2 {x2.shape}, out {out.shape}")

            # Normalize output tensor
            return out / 1.414

        # If not using residual connection, return output of second convolutional layer
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

    # Method to get the number of output channels for this block
    def get_out_channels(self):
        return self.conv2[0].out_channels

    # Method to set the number of output channels for this block
    def set_out_channels(self, out_channels):
        self.conv1[0].out_channels = out_channels
        self.conv2[0].in_channels = out_channels
        self.conv2[0].out_channels = out_channels

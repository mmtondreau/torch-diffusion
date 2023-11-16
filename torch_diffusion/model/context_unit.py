from dataclasses import dataclass
import torch.nn as nn
import pytorch_lightning as pl
import torch
from torch_diffusion.model.embedfc import EmbedFC
from torch_diffusion.model.residual_conv_block import ResidualConvBlock
from torch_diffusion.model.unet_down import UnetDown
from torch_diffusion.model.unet_up import UnetUp
from typing import List, TypedDict


@dataclass
class ContextUnitLayerCOnfig:
    kernel_size: int
    features: int


@dataclass
class ContextUnitConfig:
    features: int
    layers: list[ContextUnitLayerCOnfig]


class ContextUnet(pl.LightningModule):
    def __init__(self, in_channels, config: ContextUnitConfig, height=192, width=128):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = config.features
        self.layers = config.layers
        num_layers = len(self.layers)
        self.h = height
        self.w = width

        self.init_conv = ResidualConvBlock(in_channels, self.n_feat, is_res=True)

        feature_kernel = (height // (2**num_layers), width // (2**num_layers))

        up0_dim = self.layers[-1].features
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(
                up0_dim,
                up0_dim,
                feature_kernel,
                feature_kernel,
            ),
            nn.GroupNorm(8, up0_dim),
            nn.ReLU(),
        )

        self.down = nn.ModuleList()
        self.timeembed = nn.ModuleList()
        self.up = nn.ModuleList()
        prev_layer: int = self.n_feat
        for layer in self.layers:
            features = layer.features
            down = UnetDown(
                prev_layer,
                features,
                layer.kernel_size,
            )
            embed = EmbedFC(1, features)

            up = UnetUp(
                2 * features,  # takes down cat embed as inputs
                prev_layer,
                kernel_size=layer.kernel_size,
            )

            self.timeembed.append(embed)
            self.down.append(down)
            self.up.append(up)
            prev_layer = features

        self.to_vec = nn.Sequential(nn.AvgPool2d(feature_kernel), nn.GELU())

        self.out = nn.Sequential(
            nn.Conv2d(2 * self.n_feat, self.n_feat, 3, 1, 1),
            nn.GroupNorm(8, self.n_feat),
            nn.ReLU(),
            nn.Conv2d(self.n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        c : (batch, n_classes)    : context label
        """
        x = self.init_conv(x)
        x_orig = x

        down_out = []
        for module in self.down:
            x = module(x)
            down_out.append(x)

        hiddenvec = self.to_vec(x)

        temb_out = []
        for index, module in enumerate(self.timeembed):
            temb_out.append(module(t).view(-1, self.n_feat * (2 ** (index + 1)), 1, 1))

        x = self.up0(hiddenvec)

        for module in reversed(self.up):
            x = module(x + temb_out.pop(), down_out.pop())

        out = self.out(torch.cat((x, x_orig), 1))
        return out

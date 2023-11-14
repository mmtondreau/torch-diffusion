import torch.nn as nn
import pytorch_lightning as pl
import torch
from torch_diffusion.model.embedfc import EmbedFC
from torch_diffusion.model.residual_conv_block import ResidualConvBlock
from torch_diffusion.model.unet_down import UnetDown
from torch_diffusion.model.unet_up import UnetUp
from typing import TypedDict


class ContextUnitConfig(TypedDict):
    features: int
    scale: list[int]


class ContextUnet(pl.LightningModule):
    def __init__(self, in_channels, config: ContextUnitConfig, height=192, width=128):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = config["features"]
        self.scale = config["scale"]
        num_layers = len(self.scale)
        self.h = height
        self.w = width

        self.init_conv = ResidualConvBlock(in_channels, self.n_feat, is_res=True)

        self.down = nn.ModuleList()
        for layer, kernel in enumerate(self.scale):
            self.down.append(
                UnetDown(
                    (2**layer) * self.n_feat, (2 ** (layer + 1)) * self.n_feat, kernel
                )
            )

        feature_kernel = (height // (2**num_layers), width // (2**num_layers))
        self.to_vec = nn.Sequential(nn.AvgPool2d(feature_kernel), nn.GELU())

        timeembed = []
        for layer, _ in enumerate(self.scale):
            self.timeembed.append(EmbedFC(1, (2**layer) * self.n_feat))
        timeembed.reverse()
        self.timeembed = nn.ModuleList(timeembed)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(
                4 * self.n_feat,
                4 * self.n_feat,
                feature_kernel,
                feature_kernel,
            ),
            nn.GroupNorm(8, 4 * self.n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(8 * self.n_feat, 2 * self.n_feat, kernel_size=3)
        self.up2 = UnetUp(4 * self.n_feat, self.n_feat, kernel_size=3)
        self.up3 = UnetUp(2 * self.n_feat, self.n_feat, kernel_size=3)

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
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        hiddenvec = self.to_vec(down3)

        temb1 = self.timeembed1(t).view(-1, self.n_feat * 4, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 2, 1, 1)
        temb3 = self.timeembed3(t).view(-1, self.n_feat * 1, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1 + temb1, down3)
        up3 = self.up2(up2 + temb2, down2)
        up4 = self.up3(up3 + temb3, down1)
        out = self.out(torch.cat((up4, x), 1))
        return out

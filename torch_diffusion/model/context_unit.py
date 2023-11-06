import torch.nn as nn
import pytorch_lightning as pl
import torch
from torch_diffusion.model.embedfc import EmbedFC
from torch_diffusion.model.residual_conv_block import ResidualConvBlock
from torch_diffusion.model.unet_down import UnetDown
from torch_diffusion.model.unet_up import UnetUp


class ContextUnet(pl.LightningModule):
    def __init__(self, in_channels, n_feat=256, n_cfeat=10, height=192, width=128):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height
        self.w = width

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat, 3)
        self.down2 = UnetDown(n_feat, 2 * n_feat, 3)
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat, 3)

        feature_kernel = (height // 8, width // 8)
        self.to_vec = nn.Sequential(nn.AvgPool2d(feature_kernel), nn.GELU())

        self.timeembed1 = EmbedFC(1, 4 * n_feat)
        self.timeembed2 = EmbedFC(1, 2 * n_feat)
        self.timeembed3 = EmbedFC(1, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(
                4 * n_feat,
                4 * n_feat,
                feature_kernel,
                feature_kernel,
            ),
            nn.GroupNorm(8, 4 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(8 * n_feat, 2 * n_feat, kernel_size=3)
        self.up2 = UnetUp(4 * n_feat, n_feat, kernel_size=3)
        self.up3 = UnetUp(2 * n_feat, n_feat, kernel_size=3)

        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
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

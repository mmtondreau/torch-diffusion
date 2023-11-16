import torch.nn as nn
import pytorch_lightning as pl


class EmbedFC(pl.LightningModule):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        """
        This class defines a generic one layer feed-forward neural network for embedding input data of
        dimensionality input_dim to an embedding space of dimensionality emb_dim.
        """
        self.input_dim = input_dim
        self.output_dim = emb_dim
        # define the layers for the network
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]

        # create a PyTorch sequential model consisting of the defined layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # flatten the input tensor
        x = x.view(-1, self.input_dim)
        # apply the model layers to the flattened tensor
        return self.model(x)

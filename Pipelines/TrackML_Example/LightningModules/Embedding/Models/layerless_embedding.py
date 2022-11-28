# System imports
import sys
import os

# 3rd party imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from ..embedding_base import EmbeddingBase
from torch.nn import Linear
import torch.nn as nn
from torch_cluster import radius_graph
import torch.nn.functional as F
import torch
from torch_geometric.loader import DataLoader

# Local imports
from ..utils import graph_intersection, make_mlp,make_quantized_mlp


class LayerlessEmbedding(EmbeddingBase):
    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different embedding training regimes
        """
        # Construct the MLP architecture
        in_channels = hparams["spatial_channels"] + hparams["cell_channels"]
      
        if(hparams["quantized_network"]==False):
            
            self.network = make_mlp(
                in_channels,
                [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
                hidden_activation=hparams["activation"],
                output_activation=None,
                layer_norm=True,
            )
        else:

            self.network = make_quantized_mlp(
                in_channels,
                [hparams["emb_hidden"]] * hparams["nb_layer"] + [hparams["emb_dim"]],
                hidden_activation=hparams["hidden_activation_qnn"],
                output_activation=hparams["output_activation_qnn"],
                weight_bit_width = hparams["weight_bit_width"],
                activation_bit_width = hparams["activation_bit_width"],
                input_layer_quantization = hparams["input_layer_quantization"],
                layer_norm = True

            )




        self.save_hyperparameters()

    def forward(self, x):

        x_out = self.network(x)

        if "norm" in self.hparams["regime"]:
            return F.normalize(x_out)
        else:
            return x_out
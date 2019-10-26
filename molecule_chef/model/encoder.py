

import torch
from torch import nn
from torch.nn.utils import rnn

from autoencoders.dist_parameterisers import shallow_distributions
from autoencoders.dist_parameterisers import nn_paramterised_dists

from . import embedders
from .. import torch_utils


class SumEmbedMapEncoderNet(nn.Module):
    def __init__(self, gnn_embedder: embedders.SymbolGNNEmbedder, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp
        self.gnn_embedder = gnn_embedder

    def forward(self, symbol_seq_packed):
        symbol_seq_packed = torch_utils.remove_last_from_packed_seq(symbol_seq_packed)
        # ^ we remove the stop symbol as we are not interested in it here.
        embedded_seq = self.gnn_embedder.forward_on_packed_sequence(symbol_seq_packed, stops_pre_filtered_flag=True)
        padded_seq, _ = rnn.pad_packed_sequence(embedded_seq, padding_value=0.0)  #[T, b, h]
        # ^ pad with 0's so do not contribute to sum below.
        summed = torch.sum(padded_seq, dim=0)  # [b, h]
        mapped = self.mlp(summed)  # [b, h'] -> [b, h'] for encoding a dist
        return mapped


def get_encoder(gnn_embedder: embedders.SymbolGNNEmbedder, mlp: nn.Module):
    network = SumEmbedMapEncoderNet(gnn_embedder, mlp)
    encoder_top = shallow_distributions.IndependentGaussianDistribution()
    return nn_paramterised_dists.NNParamterisedDistribution(network, encoder_top)


import typing
from dataclasses import dataclass

import torch
from torch import nn

from graph_neural_networks.core import mlp
from graph_neural_networks.core.utils import CudaDetails
from graph_neural_networks.sparse_pattern import graph_as_adj_list

from autoencoders.dist_parameterisers import shallow_distributions
from autoencoders import wasserstein
from autoencoders import similarity_funcs

from ..aux_models import graph_embedder
from .. import mchef_config
from . import embedders
from . import encoder
from . import decoder


@dataclass
class MChefParams:
    cuda_details: CudaDetails
    index_to_graph_lookup: typing.Callable[[torch.Tensor], graph_as_adj_list.GraphAsAdjList]
    total_number_of_graphs: int
    stop_indx: int
    latent_dim: int

    # GGNN specific
    gnn_hidden_size: int = 101  # our molecule features have this dimensionality.
    edge_names = ['single', 'double', 'triple']
    ggnn_time_steps = 4

    # Embedding Specific
    embedding_dim = 50

    # Property specific
    property_dim:int = 1

    # Decoder Specific
    decd_layers: int = 2
    decd_max_steps:int = 5



def get_mol_chef(params: MChefParams):
    # Create embedder
    ggnn = graph_embedder.GraphEmbedder(params.gnn_hidden_size, params.edge_names, params.embedding_dim, params.cuda_details,
                                        params.ggnn_time_steps)
    embedr = embedders.SymbolGNNEmbedder(ggnn, params.index_to_graph_lookup, params.total_number_of_graphs, params.stop_indx)

    # Create the encoder
    mlp_top = mlp.MLP(mlp.MlpParams(params.embedding_dim, params.latent_dim * 2, hidden_sizes=[200]))
    encd = encoder.get_encoder(embedr, mlp_top)

    # Create the decoder
    gru_hsize = params.embedding_dim
    mlp_decdr_gru_to_h = mlp.MLP(mlp.MlpParams(gru_hsize, params.embedding_dim, hidden_sizes=[128]))
    mlp_proj_latents_to_hidden = nn.Linear(params.latent_dim, gru_hsize)
    decoder_params = decoder.DecoderParams(gru_insize=params.embedding_dim, gru_hsize=gru_hsize,
                                           num_layers=params.decd_layers, gru_dropout=0.,
                                           mlp_out=mlp_decdr_gru_to_h, mlp_parameterise_hidden=mlp_proj_latents_to_hidden,
                                           max_steps=params.decd_max_steps)
    decd = decoder.get_decoder(embedr, decoder_params)

    # Create the latent prior
    latent_prior = shallow_distributions.IndependentGaussianDistribution(
        nn.Parameter(torch.zeros(1, params.latent_dim * 2,
                                 device=params.cuda_details.device_str, dtype=mchef_config.PT_FLOAT_TYPE),
                     requires_grad=False))

    # Create the kernel
    c = 2*params.latent_dim*(1**2)
    # ^ see section 4 of Wasserstein Auto-Encoders by Tolstikhin et al. for motivation behind this value.
    kernel = similarity_funcs.InverseMultiquadraticsKernel(c=c)

    # create the autoencoder
    wae = wasserstein.WAEnMMD(encd, decd, latent_prior, kernel=kernel)

    # Add the regressor from latent space to property prediction
    wae.prop_predictor_ = mlp.MLP(mlp.MlpParams(params.latent_dim, params.property_dim, [40, 40]))

    return wae

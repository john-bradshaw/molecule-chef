
from torch import nn

from . import graph_embedder
from graph_neural_networks.core import mlp


class GNNThenMLP(nn.Module):
    def __init__(self, output_dim, hidden_layer_size, edge_names, embedding_dim, cuda_details, T):
        super().__init__()
        self.ggnn = graph_embedder.GraphEmbedder(hidden_layer_size, edge_names, embedding_dim, cuda_details, T)
        self.mlp = mlp.MLP(mlp.MlpParams(embedding_dim, output_dim, [125, 80, 50]))

    def forward(self, graphs_in):
        embedded_graphs = self.ggnn(graphs_in)
        prediction = self.mlp(embedded_graphs)
        return prediction.squeeze()


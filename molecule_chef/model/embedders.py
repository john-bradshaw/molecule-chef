
import typing

import torch
from torch import nn
from torch.nn.utils import rnn
import numpy as np

from graph_neural_networks.sparse_pattern import graph_as_adj_list

from ..aux_models import graph_embedder
from .. import mchef_config


class SymbolGNNEmbedder(nn.Module):
    """
    Symbols are reactants/stop
    We label the symbols with consecutive integers (from 0) with all but the final one representing the reactant graphs.
    The final integer (number_graphs, as idx from 0) represents the stop sequence symbol.

    The graph symbols have embeddings calculated by embedding their associated graphs using a Graph Neural Network (GNN)
    The stop symbol is associated with a learnt embedding (stored as a parameter in this class)
    """
    def __init__(self, gnn_graph_embedder: graph_embedder.GraphEmbedder,
                 index_to_graph_lookup: typing.Callable[[torch.Tensor], graph_as_adj_list.GraphAsAdjList],
                 total_number_of_graphs: int,
                 stop_symbol_indx: int
                 ):
        """
        :param gnn_graph_embedder: takes a graph and computes its GNN embedding.
        :param index_to_graph_lookup: Takes the index of a symbol and looks up its associated graph
        :param total_number_of_graphs: total number of graphs in vocabulary.
        :param stop_symbol_indx: should be equal to `total_number_of_graphs` and represents the integer label for the
        stop symbol.
        """

        super().__init__()
        self.graph_embedder = gnn_graph_embedder
        self.index_to_graph_lookup = index_to_graph_lookup
        self._total_number_of_graphs = total_number_of_graphs
        self._stop_symbol_indx = stop_symbol_indx

        # Some sanity checks to make sure we are using sensible symbols
        assert stop_symbol_indx == total_number_of_graphs, "stop symbol shoulb be after all graphs"
        assert mchef_config.PAD_VALUE not in set(range(total_number_of_graphs +1)), "pad symbol also used for graph/stop"

        # To embed the stop symbol then we will just learn an embedding.
        self.stop_embedding = nn.Parameter(torch.Tensor(self.embedding_dim).to(mchef_config.PT_FLOAT_TYPE))
        bound = 1 / np.sqrt(self.embedding_dim)
        nn.init.uniform_(self.stop_embedding, -bound, bound)

    def forward(self, symbol_tensor_in: torch.Tensor, stops_pre_filtered_flag=False) -> torch.Tensor:
        """
        :param stops_pre_filtered_flag: set to avoid having to check whether we have stop symbols (and dealing
        with these differently)
        """
        return self._zip_graph_embeddings_and_stop_symbols(symbol_tensor_in, stops_pre_filtered_flag)

    def forward_on_packed_sequence(self, packed_sequence, stops_pre_filtered_flag=False):
        data, *other = packed_sequence
        new_data = self.forward(data, stops_pre_filtered_flag)
        new_packed = rnn.PackedSequence(new_data, *other)
        return new_packed

    def _zip_graph_embeddings_and_stop_symbols(self, indcs_to_convert: torch.Tensor, stops_pre_filtered_flag):
        if stops_pre_filtered_flag:
            # Feed everything into the GGNN
            graphs = self.index_to_graph_lookup(indcs_to_convert)
            embeddings = self.graph_embedder(graphs)
        else:
            # Feed the graphs into the GGNN and add the learnt stop embeddings elsewhere.
            embeddings = torch.zeros(indcs_to_convert.shape[0], self.embedding_dim,
                                     device=str(self._device_to_use),
                                     dtype=mchef_config.PT_FLOAT_TYPE)
            stop_locations = indcs_to_convert == self._stop_symbol_indx
            graph_locations = stop_locations == False

            embeddings[stop_locations, :] = self.stop_embedding[None, :]

            symbols_for_graphs_needed = indcs_to_convert[graph_locations]
            graphs = self.index_to_graph_lookup(symbols_for_graphs_needed)
            embeddings_of_graphs = self.graph_embedder(graphs)

            embeddings[graph_locations, :] = embeddings_of_graphs
        return embeddings

    @property
    def embedding_dim(self):
        return self.graph_embedder.embedding_dim

    @property
    def total_num_of_embeddingsincl_stop(self):
        return self._total_number_of_graphs + 1  # plus one is for the stop embedding.

    @property
    def all_embeddings(self):  # [V, h]
        # range to cover all the elements including stop:
        return self(torch.arange(self.total_num_of_embeddingsincl_stop), stops_pre_filtered_flag=False)

    @property
    def stop_symbol_indx(self):
        return self._stop_symbol_indx

    @property
    def _device_to_use(self):
        return str(self.stop_embedding.device)

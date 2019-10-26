
import pickle

import numpy as np
import torch
from torch.utils import data

from graph_neural_networks.sparse_pattern import graph_as_adj_list

from ..chem_ops import rdkit_featurization_ops
from .. import mchef_config


class PickledIndexableDataset(data.Dataset):
    """
    Generic dataset that reads from a pickle files, indexes it with integers and applies a transform before returning
    the result.
    """
    def __init__(self, path_to_read, transforms=None):
        """
        :param path_to_read: str to where a pickle should be read
        """
        with open(path_to_read, 'rb') as fo:
            data = pickle.load(fo)
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        if self.transforms is not None:
            data_item = self.transforms(data_item)
        return data_item


class PickledGraphDataset:
    """
    This class is used for assessing the graph features that are stored in a Pickled List.
    """
    def __init__(self, path_to_read, cuda_details):
        self.pickled_indexable_dataset = PickledIndexableDataset(path_to_read,
                                                                 trfm_mol_as_adj_list_to_graph_as_adj_list_trsfm)
        self.cuda_details = cuda_details
        self.total_number = len(self.pickled_indexable_dataset)

    def __call__(self, indices: torch.Tensor):
        graphs = [self.pickled_indexable_dataset[i] for i in indices.cpu().numpy()]
        graphs_concatenated = graphs[0].concatenate(graphs)
        graphs_concatenated = graphs_concatenated.to_torch(cuda_details=self.cuda_details)
        return graphs_concatenated

    def __len__(self):
        return len(self.pickled_indexable_dataset)



def trfm_mol_as_adj_list_to_graph_as_adj_list_trsfm(mol_in: rdkit_featurization_ops.MolAsAdjList) -> graph_as_adj_list.GraphAsAdjList:
    graph = graph_as_adj_list.GraphAsAdjList(mol_in.atom_feats,
                                             {k: np.array(v).T for k, v in mol_in.adj_lists_for_each_bond_type.items()},
                                              np.zeros(mol_in.atom_feats.shape[0], dtype=mchef_config.INT_TYPE))
    return graph


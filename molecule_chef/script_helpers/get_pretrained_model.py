
import torch

from graph_neural_networks.core import utils as gnn_utils

from ..data import atom_features_dataset
from .. import mchef_config
from ..model import get_mchef

def load_in_mchef(chkpt_path, cuda_details: gnn_utils.CudaDetails, path_molecule_details):
    indices_to_graphs = atom_features_dataset.PickledGraphDataset(path_molecule_details, cuda_details)

    chkpt = torch.load(chkpt_path, map_location=cuda_details.device_str)
    # ^ Load the checkpoint as we can get some model details from there.

    latent_dim = chkpt['wae_state_dict']['latent_prior._params'].shape[1] // 2
    print(f"Inferred a latent dimensionality of {latent_dim}")
    assert chkpt['stop_symbol_idx'] == mchef_config.get_num_graphs()

    mol_chef_params = get_mchef.MChefParams(cuda_details, indices_to_graphs, len(indices_to_graphs),
                                            chkpt['stop_symbol_idx'], latent_dim)
    molchef_wae = get_mchef.get_mol_chef(mol_chef_params)
    molchef_wae = cuda_details.return_cudafied(molchef_wae)

    # == Lets load in some good weights! ==
    molchef_wae.load_state_dict(chkpt['wae_state_dict'])
    return molchef_wae, latent_dim, chkpt['stop_symbol_idx']

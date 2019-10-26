""" Create the reactant bags (tokenized) using the trained regressor and Molecule Chef's decoder. Ie retrosyntheis.

Usage:
  try_trained_network_on_products.py <input_weights_mchef> <input_weights_regressor>


"""

from os import path
import math

from docopt import docopt
import torch
import tqdm

from graph_neural_networks.core import utils as gnn_utils

from molecule_chef.data import atom_features_dataset
from molecule_chef import mchef_config
from molecule_chef.model import get_mchef
from molecule_chef.script_helpers import molecular_transformer_helper as mt
from molecule_chef.aux_models import graph_regressors
from molecule_chef.chem_ops import rdkit_general_ops
from molecule_chef.chem_ops import rdkit_featurization_ops
from molecule_chef.script_helpers.get_pretrained_model import load_in_mchef


class Params:
    def __init__(self):
        self.cuda_details = gnn_utils.CudaDetails(use_cuda=torch.cuda.is_available())
        # GNN details
        self.gnn_args = dict(output_dim=25, hidden_layer_size=101, edge_names=['single', 'double', 'triple'],
                             embedding_dim=50, T=4)

        # Data Paths
        processed_data_dir = mchef_config.get_processed_data_dir()
        self.path_mol_details = path.join(processed_data_dir, 'reactants_feats.pick')

        self.product_files_to_try = [('test_reachable', path.join(processed_data_dir, 'test_products.txt')),
                                     ('test_unreachable', path.join(processed_data_dir, 'test_unreachable_products.txt'))]

        # Command line arguments.
        arguments = docopt(__doc__)
        self.weights_to_use_mchef = arguments['<input_weights_mchef>']
        self.weights_to_use_regressor = arguments['<input_weights_regressor>']


@torch.no_grad()
def _run_through(product_path, regressor, mol_chef_wae, seq_to_smi_list_func, cuda_details):
    # == Get the products we want to retrosyntheize  ==
    with open(product_path, 'r') as fo:
        products = [x.strip() for x in fo.readlines()]

    # == Get graph representation of the ones we care about ==
    graphs = []
    for mol_smi in tqdm.tqdm(products, desc="creating graphs"):
        mol = rdkit_general_ops.get_molecule(mol_smi, kekulize=True)
        mol, am_to_indx_map = rdkit_general_ops.add_atom_mapping(mol)
        mol_as_adj_list = rdkit_featurization_ops.mol_to_atom_feats_and_adjacency_list(mol, am_to_indx_map)
        graph = atom_features_dataset.trfm_mol_as_adj_list_to_graph_as_adj_list_trsfm(mol_as_adj_list)
        graphs.append(graph)

    # == Now regress to latent space & run decoder ==
    batch_size = 500
    predicted_latents = []
    resultant_reactants = []
    for i in tqdm.tqdm(range(math.ceil(len(graphs) / batch_size)), desc="to_z_and_then_bag"):
        graphs_of_batch = graphs[i*batch_size:(i+1)*batch_size]
        graphs_of_batch = graphs_of_batch[0].concatenate(graphs_of_batch)
        graphs_of_batch = graphs_of_batch.to_torch(cuda_details)
        latents_ = regressor(graphs_of_batch)
        predicted_latents.append(latents_.cpu().numpy())

        seq_, _ = mol_chef_wae.decode_from_z_no_grad(latents_)
        predicted_seqs_batch_first_np = seq_.cpu().numpy().T
        for seq in predicted_seqs_batch_first_np:
            seq_as_mols = seq_to_smi_list_func(seq)
            reactant_str = '.'.join(sorted(seq_as_mols))
            resultant_reactants.append(reactant_str)

    return resultant_reactants


def _load_regressor(chkpt_path, cuda_details: gnn_utils.CudaDetails, gnn_args):
    chkpt_prod_to_latent = torch.load(chkpt_path, map_location=cuda_details.device_str)

    model = graph_regressors.GNNThenMLP(**gnn_args, cuda_details=cuda_details)
    model = cuda_details.return_cudafied(model)
    model.load_state_dict(chkpt_prod_to_latent['model_state_dict'])
    return model




def main(params: Params):
    print("Starting...")

    # == Load in the regressor model. ==
    regressor = _load_regressor(params.weights_to_use_regressor, params.cuda_details, params.gnn_args)

    # == Now load in Molecule Chef ==
    molchef_wae, *_ = load_in_mchef(params.weights_to_use_mchef, cuda_details=params.cuda_details,
                                 path_molecule_details=params.path_mol_details)

    # == Get the mapping from molecules to ID ==
    seq_to_smi_list = mt.MapSeqsToReactants()

    # == Run predictions on the various datasets! ==
    seen_reactants_set = set()
    for name, product_path in params.product_files_to_try:
        reactants = _run_through(product_path, regressor, molchef_wae, seq_to_smi_list, params.cuda_details)

        with open(f'op/{name}_retrosynthesized_reactants.txt', 'w') as fo:
            fo.writelines('\n'.join(reactants))

        seen_reactants_set.update(set(reactants))

    tokenized_reac = [mt.tokenization(smi_str) for smi_str in set(seen_reactants_set) if len(smi_str)]
    with open("op/retro.tokenized-reactant.txt", 'w') as fo:
        fo.writelines('\n'.join(tokenized_reac))


if __name__ == '__main__':
    main(Params())


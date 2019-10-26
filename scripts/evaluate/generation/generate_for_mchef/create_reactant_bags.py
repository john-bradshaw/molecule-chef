"""Create Reactant Bags

Usage:
  create_reactant_bags.py <input_weights> <output_name>


"""

import math
from os import path

import torch
import tqdm
from docopt import docopt

from graph_neural_networks.core import utils as gnn_utils

from molecule_chef.data import atom_features_dataset
from molecule_chef import mchef_config
from molecule_chef.model import get_mchef
from molecule_chef.script_helpers import molecular_transformer_helper as mt
from molecule_chef.script_helpers.get_pretrained_model import load_in_mchef



class Params:
    def __init__(self):
        self.num_to_generate = 20000
        self.batch_size = 2000

        processed_data_dir = mchef_config.get_processed_data_dir()

        self.path_mol_details = path.join(processed_data_dir, 'reactants_feats.pick')

        self.cuda_details = gnn_utils.CudaDetails(use_cuda=torch.cuda.is_available())

        arguments = docopt(__doc__)
        self.weights_to_use = arguments['<input_weights>']
        self.location_for_tokenized_reactants = arguments['<output_name>']


def main(params: Params):
    # == Lets get the model! ==
    molchef_wae, latent_dim, _ = load_in_mchef(params.weights_to_use, cuda_details=params.cuda_details,
                                path_molecule_details=params.path_mol_details)

    # == Sample ==
    sampled_seqs = []
    batch_size = params.batch_size

    for i in tqdm.tqdm(range(math.ceil(params.num_to_generate / batch_size))):
        random_z = torch.randn(batch_size, latent_dim,  device=params.cuda_details.device_str)
        # ^ a normal prior.
        x, _ = molchef_wae.decode_from_z_no_grad(random_z)
        sampled_seqs += list(x.unbind(dim=1))

    # == Let's get the mapping from id to SMILES ==
    seq_to_smi_list = mt.MapSeqsToReactants()

    # == Turn the symbol id samples into reactant strings ==
    def map_function(tensor_array):
        sequence = tensor_array.detach().cpu().numpy()
        reactants = seq_to_smi_list(sequence)
        reactants = '.'.join(reactants)
        return reactants

    sampled_reactants = [map_function(seq) for seq in sampled_seqs]

    # == We now remove the empty reactant strings -- these should be automatically called invalid later  ==
    # (this needs to be done as the OpenNMT Transformer implementation being used cannot work on empty output)
    n_sampled_before = sampled_reactants
    sampled_reactants = [elem for elem in sampled_reactants if elem != '']
    print(f"{len(n_sampled_before) - len(sampled_reactants)} reactant bags removed from tokenzied set as were empty")

    # == Finally we tokenize and write the reactants out  ==
    tokenized_sampled_reactants = [mt.tokenization(smi_str) for smi_str in sampled_reactants]
    with open(params.location_for_tokenized_reactants, 'w') as fo:
        fo.writelines('\n'.join(tokenized_sampled_reactants))


if __name__ == '__main__':
    main(Params())


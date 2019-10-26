""" WAE optimize locally based on property predictor network.

Usage:
  wae_latent_space_optimize.py <input_weights>


"""


import collections
from os import path
import pickle

import torch
import tqdm
import numpy as np

from torch.nn.utils import rnn
from torch.nn import functional as F
from docopt import docopt

from graph_neural_networks.core import utils as gnn_utils

from molecule_chef.data import symbol_sequence_data
from molecule_chef import mchef_config
from molecule_chef.script_helpers import molecular_transformer_helper as mt
from molecule_chef.script_helpers.get_pretrained_model import load_in_mchef


class Params:
    def __init__(self):
        self.num_molecules_to_optimize: int = 250
        self.num_distinct_molecule_steps: int = 10
        self.epsilon: float = 0.5

        self.cuda_details = gnn_utils.CudaDetails(use_cuda=torch.cuda.is_available())

        processed_data_dir = mchef_config.get_processed_data_dir()

        self.path_mol_details = path.join(processed_data_dir, 'reactants_feats.pick')
        self.path_react_bags_train = path.join(processed_data_dir, 'train_react_bags.txt')

        # Command line arguments.
        arguments = docopt(__doc__)
        self.weights_to_use = arguments['<input_weights>']


class LocalSearchRunner:
    def __init__(self, use_random_dir_flag, nn_predictor, ae, seq_to_smi_list_func, params: Params):
        self.random_dir_flag = use_random_dir_flag
        self.nn_predictor = nn_predictor
        self.ae = ae
        self.seq_to_smi_list = seq_to_smi_list_func
        self.params = params

        self.block_size = 100

    def go_from_batched_zs_to_predicted_products(self, z_samples):
        symbol_samples_seq_first_torch, _ = self.ae.decode_from_z_no_grad(z_samples)
        symbol_samples_batch_first_numpy = [arr.cpu().numpy() for arr in symbol_samples_seq_first_torch.transpose(0, 1)]
        reactant_strings = ['.'.join(sorted(self.seq_to_smi_list(seq)))
                            for seq in symbol_samples_batch_first_numpy]
        return reactant_strings

    def optimize_z(self, initial_z, num_of_diverse_molecule_sets_to_be_found, epsilon):
        initial_z = initial_z.clone().detach()

        initial_z_np = initial_z.cpu().numpy()
        assert initial_z_np.shape[0] == 1, "should have batch on leading dimension"
        all_zs = [initial_z_np]
        all_val_preds = [self.nn_predictor(initial_z).item()]
        all_reactant_strings = self.go_from_batched_zs_to_predicted_products(initial_z)

        last_z = initial_z
        while True:

            # So we will form a batch of z points to query next (to feed through decoder all at once):
            zs_on_optimisation_path = []
            for i in range(self.block_size):
                new_dir = self.get_direction(last_z)
                last_z = last_z + epsilon * new_dir
                last_z = last_z.detach()
                zs_on_optimisation_path.append(last_z)

            # We will then query these z points by running them through as one whole batch:
            zs_on_optimisation_path = torch.cat(zs_on_optimisation_path, dim=0)
            with torch.no_grad():
                predicted_values = self.nn_predictor(zs_on_optimisation_path)
                reactant_strings = self.go_from_batched_zs_to_predicted_products(zs_on_optimisation_path)

            # We add the results to our storage outside the while loop
            all_zs.append(zs_on_optimisation_path.detach().cpu().numpy())
            all_val_preds += predicted_values.detach().cpu().numpy().tolist()
            all_reactant_strings += reactant_strings
            if len(all_reactant_strings) > 500:
                print("Over 500 steps needed.:" + str(len(all_reactant_strings)))

            # We check if we have found a number of distinct molecules required
            if len(set(all_reactant_strings)) >= num_of_diverse_molecule_sets_to_be_found:
                break

        all_zs = np.concatenate(all_zs)

        assert all_zs.shape[0] == len(all_val_preds)
        assert len(all_reactant_strings) == len(all_val_preds)
        return all_zs, all_val_preds, all_reactant_strings

    def get_direction(self, last_z):
        if self.random_dir_flag:
            direction = torch.randn_like(last_z)
        else:
            last_z.requires_grad = True
            self.nn_predictor.zero_grad()
            if last_z.grad is not None:
                last_z.grad.detach_()
                last_z.grad.zero_()
            val_predicted = self.nn_predictor(last_z)
            assert val_predicted.shape[0] == 1, "should only be working on one example as a time"
            val_predicted = val_predicted.sum()
            val_predicted.backward()
            direction = last_z.grad
        normalized_dir = F.normalize(direction)
        return normalized_dir


def main(params: Params):
    rng = np.random.RandomState(1001)
    torch.manual_seed(rng.choice(10000000))

    # == Lets get the model! ==
    molchef_wae, latent_dim, stop_symbol_idx = load_in_mchef(params.weights_to_use, cuda_details=params.cuda_details,
                                 path_molecule_details=params.path_mol_details)

    # == Let's get the mapping from id to SMILES ==
    seq_to_smi_list = mt.MapSeqsToReactants()

    # == Get dataset  ==
    trsfm = symbol_sequence_data.TrsfmSeqStrToArray(symbol_sequence_data.StopSymbolDetails(True, stop_symbol_idx),
                                                    shuffle_seq_flag=True,
                                                    rng=rng)
    reaction_bags_dataset = symbol_sequence_data.SymbolSequenceDataset(params.path_react_bags_train, trsfm)

    # == Lets get a set of initial z  ==
    # start from train examples.
    print("starting from training examples.")
    zs_to_start_from_train_data = []
    indices_to_use = list(range(10)) + rng.permutation(len(reaction_bags_dataset))[:params.num_molecules_to_optimize - 10].tolist()
    # ^ use first ten as well as random ones as easy to look at first ten

    # We'll embed each molecule into the latent space one by one:
    # could batch this: but then would have to deal with ordering by length for packing padded sequence and for the
    # number of molecules that we run on it seems fast enough.
    for i in tqdm.tqdm(indices_to_use, desc="creating initial starting locations"):
        sequence_batch_first = reaction_bags_dataset[i][0]
        sequence_batch_first = torch.from_numpy(sequence_batch_first).view(1, -1)
        lengths = torch.tensor([sequence_batch_first.shape[1]])
        packed_seq = rnn.pack_padded_sequence(sequence_batch_first, lengths, batch_first=True)
        packed_seq = packed_seq.to(params.cuda_details.device_str)
        z_sample = molchef_wae._run_through_to_z(packed_seq)
        zs_to_start_from_train_data.append(z_sample)

    # == Now we shall run the optimization for each of these z samples  ==
    results = collections.defaultdict(list)

    searches = [('random_search',
                 LocalSearchRunner(True, molchef_wae.prop_predictor_, molchef_wae, seq_to_smi_list, params)),
                ('prop_opt',
                 LocalSearchRunner(False, molchef_wae.prop_predictor_, molchef_wae, seq_to_smi_list, params))]

    for search_name, searcher in searches:
        print(f"Doing {search_name}")

        init_points = zs_to_start_from_train_data
        for initial_z in tqdm.tqdm(init_points):
            results[search_name].append(searcher.optimize_z(initial_z, params.num_distinct_molecule_steps, params.epsilon))

    # == Now we can write out the files to store the reactants found on the trace. ==
    with open('local_search_results.pick', 'wb') as fo:
        pickle.dump(results, fo)

    # we shall also write out tokenized reactant bags that we need predicting.
    all_reactant_bags = set()
    for results_for_search_type in results.values():
        for individual_run_results in results_for_search_type:
            reactant_strs = individual_run_results[2]
            all_reactant_bags.update(reactant_strs)  # as defined reactants should already be in canonical form.

    tokenized_sampled_reactants = [mt.tokenization(smi_str) for smi_str in all_reactant_bags if len(smi_str)]
    with open("opt.tokenized-reactant.txt", 'w') as fo:
        fo.writelines('\n'.join(tokenized_sampled_reactants))


if __name__ == "__main__":
    main(Params())


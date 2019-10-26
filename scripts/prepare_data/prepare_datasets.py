"""

prepare_datasets.py

A script for preparing the data from the processed USPTO dataset given by Jin et al., 2017
Creates:

_for the both dataset:_
A reactants_to_reactant_id.json: A JSON dictionary where the keys are canonical SMILES strings and the values are integer ids
called reactant_ids, which we shall use for indexing the molecules elsewhere.
B reactants_feats.pick: A pickle of a dictionary mapping reactant_id to features which can be fed into a GNN.

_for the training dataset specifically:_
C train_react_bags.txt: The reactant bags which we see in the training data. Each line consists of comma separated
 reactant_ids indicating each reactant in a particular reaction.
D train_products.txt: canonical SMILES representation of the products with the associated reaction in train_react_bags.txt.

_for each of the testing datasets_:
E <testing-dataset-name>_react_bags.txt: same as the train version but for reactions in the testing dataset which consist
of reactions in our vocabulary.
F <testing-dataset-name>_products.txt: canonical SMILES representation of the products with the associated reaction in
 <testing-dataset-name>_react_bags.txt.
G <testing-dataset-name>_unreachable_reactants.txt: canonical SMILES representation of the reactants which contain at least
one reactant not in our vocabulary for a reaction in the test dataset and lead to the product in H.
H <testing-dataset-name>_unreachable_products.txt: canonical SMILES representation of the products which contain at least
one reactant not in our vocabulary for a reaction in the test dataset.


"""

import typing
import pickle
import collections
import itertools
import json
import datetime
from os import path
import subprocess

import tqdm
from dataclasses import dataclass
import multiset

from molecule_chef.data import uspto_data
from molecule_chef.chem_ops import rdkit_general_ops
from molecule_chef.chem_ops import rdkit_featurization_ops
from molecule_chef import mchef_config

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)


class ReactantBundler(object):
    """
    Class that stores reactant occurrences (and associated reactants/products) in reactions and can be used
    to find reactions/reactants that occur multiple times.
    """
    def __init__(self):
        self.reactant_sets: typing.List[multiset.FrozenMultiset] = []
        # stores the reactants that go together. Useful for training.

        self.reactant_to_reactant_sets_mapping: typing.Dict[str, typing.List[int]] = collections.defaultdict(list)
        # Maps from a reactant to all the reactant sets that it belongs to

        self.product_sets: typing.List[multiset.FrozenMultiset] = []
        # ^ used to store the equivalent products

    def add_reactant(self, reactant_set: multiset.FrozenMultiset, product_set: multiset.FrozenMultiset):
        for individual_reactant_str in reactant_set:
            # We remove atom mapping from reactant, and get the CANONICAL SMILES.
            self.reactant_to_reactant_sets_mapping[individual_reactant_str].append(len(self.reactant_sets))
        self.reactant_sets.append(reactant_set)
        self.product_sets.append(product_set)

    def get_most_popular_reactant_sets_and_equiv_products(self, occur_at_least=2):
        reactant_counts = self.reactant_counts
        reactants_meeting_criteria_set = set([k for k, v in reactant_counts.items() if v >= occur_at_least])
        reactants_meeting_criteria_sorted = sorted(list(reactants_meeting_criteria_set))

        reactant_product_bags = [(r_bag, p_bag) for r_bag, p_bag in zip(self.reactant_sets, self.product_sets)
                         if r_bag.issubset(reactants_meeting_criteria_set)]
        reactant_bags, product_bags = zip(*reactant_product_bags)
        reactant_bags = list(reactant_bags)
        product_bags = list(product_bags)

        # We not deal with trimming down the vocabulary. This is needed because we
        # could have reactants that do not actually occur in any of the reactant bags as although they occur
        # multiple times they do not occur with other reactants that occur multiple times. Here we shall exclude them.
        reactants_in_reactions = set(itertools.chain(*reactant_bags))
        reactants_meeting_criteria_and_occurring = []
        reactants_meeting_criteria_and_not_occurring = []
        for reactant_ in reactants_meeting_criteria_sorted:
            if reactant_ in reactants_in_reactions:
                reactants_meeting_criteria_and_occurring.append(reactant_)
            else:
                reactants_meeting_criteria_and_not_occurring.append(reactant_)

        return (reactant_bags, product_bags,
                reactants_meeting_criteria_and_occurring, reactants_meeting_criteria_and_not_occurring)

    @property
    def reactant_counts(self):
        """
        do not count if the same reactant turns up in the reaction multiple times.
        """
        return {k: len(set(v)) for k, v in self.reactant_to_reactant_sets_mapping.items()}


def create_training_dataset_files_and_reactant_vocab(uspto_train_dataset, num_times_reactant_should_occur: int):
    reactant_bundler = ReactantBundler()
    for reaction_smi_frozen_set, product_smi_frozen_set in tqdm.tqdm(uspto_train_dataset, desc="Adding reactions to bundler"):
        reactant_bundler.add_reactant(reaction_smi_frozen_set, product_smi_frozen_set)

    (reactant_bags, product_bags,
     reactant_vocab, _) = reactant_bundler.get_most_popular_reactant_sets_and_equiv_products(
        num_times_reactant_should_occur)

    print(f"Creating reactant smi to reactant_id map.")
    reactants_to_reactant_id_dict = dict(zip(reactant_vocab, range(len(reactant_vocab))))
    create_shared_dataset_files(reactants_to_reactant_id_dict)

    print("create training files.")
    # Create file C
    lines = []
    for r_bag in reactant_bags:
        line_str = ','.join([str(reactants_to_reactant_id_dict[react]) for react in r_bag])
        lines.append(line_str)
    with open(path.join(mchef_config.get_processed_data_dir(), 'train_react_bags.txt'), 'w') as fo:
        fo.write('\n'.join(lines))

    # Create file D
    product_lines = ['.'.join(sorted(list(p_bag))) for p_bag in product_bags]
    with open(path.join(mchef_config.get_processed_data_dir(), 'train_products.txt'), 'w') as fo:
        fo.write('\n'.join(product_lines))

    return reactants_to_reactant_id_dict


def create_testing_dataset_files(name_to_prepend, dataset, reactants_to_reactant_id_dict):
    print(f"Going through dataset {name_to_prepend}")

    reactants_interested_in_set = set(reactants_to_reactant_id_dict.keys())

    reactant_bags = []
    corresponding_products = []
    unreachable_reactants = []
    unreachable_products = []

    num_reachable = 0
    num_unreachable = 0
    for reaction_smi_frozen_set, product_smi_frozen_set in tqdm.tqdm(dataset, desc=f"Going through {name_to_prepend}"):
        if reaction_smi_frozen_set.issubset(reactants_interested_in_set):
            reactant_bags.append(','.join([str(reactants_to_reactant_id_dict[react]) for react in reaction_smi_frozen_set]))
            corresponding_products.append('.'.join(sorted(list(product_smi_frozen_set))))
            num_reachable += 1
        else:
            unreachable_reactants.append('.'.join(sorted(list(reaction_smi_frozen_set))))
            unreachable_products.append('.'.join(sorted(list(product_smi_frozen_set))))
            num_unreachable += 1

    print(f"For dataset {name_to_prepend} have found {num_reachable} and {num_unreachable}")

    # Create file E
    with open(path.join(mchef_config.get_processed_data_dir(), f'{name_to_prepend}_react_bags.txt'), 'w') as fo:
        fo.write('\n'.join(reactant_bags))

    # Create file F
    with open(path.join(mchef_config.get_processed_data_dir(), f"{name_to_prepend}_products.txt"), 'w') as fo:
        fo.write('\n'.join(corresponding_products))

    # Create file G
    with open(path.join(mchef_config.get_processed_data_dir(), f"{name_to_prepend}_unreachable_reactants.txt"), 'w') as fo:
        fo.write('\n'.join(unreachable_reactants))

    # Create file H
    with open(path.join(mchef_config.get_processed_data_dir(), f"{name_to_prepend}_unreachable_products.txt"), 'w') as fo:
        fo.write('\n'.join(unreachable_products))


def create_shared_dataset_files(reactants_to_reactant_id_dict):
    print("creating shared files")
    # Create file A
    with open(path.join(mchef_config.get_processed_data_dir(), 'reactants_to_reactant_id.json'), 'w') as fo:
        json.dump(reactants_to_reactant_id_dict, fo)

    # Create file B
    print(f"Creating reactant smi to reactant_id map.")
    reactant_feats = {}
    for smiles, id in tqdm.tqdm(reactants_to_reactant_id_dict.items()):
        mol = rdkit_general_ops.get_molecule(smiles, kekulize=True)
        mol, am_to_indx_map = rdkit_general_ops.add_atom_mapping(mol)
        reactant_feats[id] = rdkit_featurization_ops.mol_to_atom_feats_and_adjacency_list(mol, am_to_indx_map)
    with open(path.join(mchef_config.get_processed_data_dir(), 'reactants_feats.pick'), 'wb') as fo:
        pickle.dump(reactant_feats, fo)


@dataclass
class Params:
    num_at_least: int = 15

    testing_datasets: tuple = ("valid", "test")


def main(params: Params):

    uspto_train_dataset = uspto_data.UsptoDataset(dataset_name="train", transforms=uspto_data.trsfm_to_reactant_product_multisets)
    testing_datsets = {k: uspto_data.UsptoDataset(dataset_name=k,
                                                  transforms=uspto_data.trsfm_to_reactant_product_multisets) for k in
                       params.testing_datasets}

    reactants_to_reactant_id_dict = create_training_dataset_files_and_reactant_vocab(uspto_train_dataset,
                                                                                    params.num_at_least)

    for name, dataset in testing_datsets.items():
        create_testing_dataset_files(name, dataset, reactants_to_reactant_id_dict)

    subprocess.run((f"cd {mchef_config.get_processed_data_dir()}; shasum -a 256 * >"
                    f" {mchef_config.get_processed_data_dir()}/{datetime.datetime.now().isoformat()}_data_checklist.sha256"),
                   shell=True)


if __name__ == '__main__':
    main(Params())

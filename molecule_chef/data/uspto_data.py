

from os import path
import typing
import itertools
import functools

from torch.utils import data

from .. import mchef_config
from ..chem_ops import rdkit_general_ops


class UsptoDataset(data.Dataset):
    def __init__(self, dataset_name="train", transforms=None):
        uspto_path = path.join(mchef_config.get_uspto_data_dir(), f'{dataset_name}.txt')
        with open(uspto_path, 'r') as fo:
            data = fo.readlines()
        self.reaction_lines = data
        self.transforms = transforms

    def __getitem__(self, idx: int):
        smiles: str = self.reaction_lines[idx]
        rest, bond_changes = smiles.split()
        (reactants, products) = rest.split('>>')

        return_val: typing.Tuple[str] = (reactants, products, bond_changes)
        if self.transforms is not None:
            return_val = self.transforms(*return_val)
        return return_val

    def __len__(self):
        return len(self.reaction_lines)


def actionset_from_uspto_line(change_str):
    """
    Gets the sec of atom-map numbers which the reaction string has indicated has changed.
    ie for change str:
     11-14;11-13
    it will return {11, 13, 14}
    """
    change_list = change_str.split(';')
    atoms = set(itertools.chain(*[map(int, c.split('-')) for c in change_list]))
    return atoms


def split_reagents_out_from_reactants_and_products(reactant_all_str: str, product_all_str: str,
                                                   action_set: set) -> typing.Tuple[str, str, str]:
    """
    :param reactant_all_str: SMILES string of all reactants -- individual reactants seperated by dots.
    :param product_all_str: SMILES string of all products -- individual reactants seperated by dots.
    :param action_set: list of atoms involved in reaction
    """
    reactants_str = reactant_all_str.split('.')
    products_str = product_all_str.split('.')

    product_smiles_set = set(products_str)
    products_to_keep = set(products_str)
    product_atom_map_nums = functools.reduce(lambda x, y: x | y, (rdkit_general_ops.get_atom_map_nums(prod) for prod in products_str))
    actions_atom_map_nums = action_set

    reactants = []
    reagents = []
    for candidate_reactant in reactants_str:
        atom_map_nums = rdkit_general_ops.get_atom_map_nums(candidate_reactant)

        # a) any atoms in products
        in_product = list(product_atom_map_nums & atom_map_nums)
        # b) any atoms in reaction center
        in_center = list(set(actions_atom_map_nums & atom_map_nums))

        if (len(in_product) == 0) and (len(in_center) == 0):  # this is a reagent
            reagents.append(candidate_reactant)
        else:
            if candidate_reactant in product_smiles_set:
                # identical in reactants and product so does not change.
                reagents.append(candidate_reactant)
                products_to_keep -= {candidate_reactant}  # remove it from the products too.
            else:
                reactants.append(candidate_reactant)

    product_all_str = '.'.join(products_to_keep)
    return '.'.join(reactants), '.'.join(reagents), product_all_str


def trsfm_to_reactant_product_multisets(reactants, products, bond_changes):
    """
    Removes the reagents from reactants and products and return reactants and products as
    frozen multisets of the molecules in their canonical SMILES form.
    """
    action_set = actionset_from_uspto_line(bond_changes)

    reactants, reagents, products = split_reagents_out_from_reactants_and_products(reactants,
                                                                                        products, action_set)

    canconical_reactants_set = rdkit_general_ops.form_canonical_smi_frozenmultiset(reactants)
    canonical_product_set = rdkit_general_ops.form_canonical_smi_frozenmultiset(products)

    return canconical_reactants_set, canonical_product_set


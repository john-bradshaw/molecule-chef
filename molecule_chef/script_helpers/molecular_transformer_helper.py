
import typing
import re

from .. import mchef_config

# from: https://github.com/pschwllr/MolecularTransformer
THE_REGEX = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
REGEX = re.compile(THE_REGEX)


def tokenization(smiles_string):
    return ' '.join(REGEX.findall(smiles_string))


class MapSeqsToReactants:
    def __init__(self, json_reactants_to_id_path=None):
        self.reactant_id_to_smi_dict = {v:k for k,v
                                        in mchef_config.get_reactant_smi_to_reactant_id_dict(json_reactants_to_id_path).items()}
        self.stop_sym_idx = mchef_config.get_num_graphs(json_reactants_to_id_path)
        self.pad_idx = mchef_config.PAD_VALUE

    def __call__(self, array_of_symbol_indices):
        filtered_list = [idx for idx in array_of_symbol_indices if idx not in {self.stop_sym_idx, self.pad_idx}]
        return [self.reactant_id_to_smi_dict[idx] for idx in filtered_list]

def read_tokenized_file(file_name):
    with open(file_name, 'r') as fo:
        data = [''.join(x.strip().split()) for x in fo.readlines()]
    return data

def get_reaction_mapping(tokenized_reactants, tokenized_products, nbest_for_tokenized):
    reacts = read_tokenized_file(tokenized_reactants)
    prods = read_tokenized_file(tokenized_products)
    prods = [p for i, p in enumerate(prods) if i % nbest_for_tokenized == 0]
    assert len(reacts) == len(prods)
    reactants_to_products_dict = dict(zip(reacts, prods))
    return reactants_to_products_dict

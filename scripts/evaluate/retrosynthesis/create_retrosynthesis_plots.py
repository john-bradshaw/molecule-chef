"""Create the retrosynthesis plots

Usage:
  create_retrosynthesis_plots.py <tokenized_reactants_path> <tokenized_products_path> [--nbest=<nbest>]

Options:
  --nbest=<nbest>  The number of predictions the transformer has made for each reactant set [default: 1].

"""
from os import path
import warnings

from docopt import docopt
import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from molecule_chef import mchef_config
from molecule_chef.script_helpers import molecular_transformer_helper as mt
from molecule_chef.script_helpers import eval_code
from molecule_chef.chem_ops import rdkit_general_ops


class Params:
    def __init__(self):

        processed_data_dir = mchef_config.get_processed_data_dir()
        self.reactant_smi_to_id = mchef_config.get_reactant_smi_to_reactant_id_dict()

        self.path_reachable_products_ground_truth = path.join(processed_data_dir, "test_products.txt")
        self.path_react_bags_test = path.join(processed_data_dir, "test_react_bags.txt")
        self.path_reachable_reactants_restrosynthezed = "./op/test_reachable_retrosynthesized_reactants.txt"

        self.path_unreachable_products_ground_truth = path.join(processed_data_dir, "test_unreachable_products.txt")
        self.path_unreachable_reactants = path.join(processed_data_dir, "test_unreachable_reactants.txt")
        self.path_unreachable_reactants_restrosynthezed = "./op/test_unreachable_retrosynthesized_reactants.txt"

        arguments = docopt(__doc__)
        self.tokenized_reactants = arguments['<tokenized_reactants_path>']
        self.tokenized_products = arguments['<tokenized_products_path>']
        self.nbest_for_tokenized = int(arguments['--nbest'])


def _react_bags_to_smi_list(react_bags_path, smi_to_id_dict):
    with open(react_bags_path, 'r') as fo:
        react_bags = [list(map(int, x.strip().split(','))) for x in fo.readlines()]
    id_to_smi = {v:k for k,v in smi_to_id_dict.items()}
    return ['.'.join(sorted([id_to_smi[react_id] for react_id in seq])) for seq in react_bags]


def _read_in_smiles_file(file_path):
    with open(file_path, 'r') as fo:
        smiles = [x.strip() for x in fo.readlines()]
    return smiles


def _zip_together_cycle(grnd_truth_products, grnd_truth_reactants, suggested_reactants, suggested_products):
    out_list = []
    for grnd_truth_prod, grnd_truth_react, sugg_react, sugg_prod in zip(grnd_truth_products, grnd_truth_reactants,
                                                                        suggested_reactants, suggested_products):
        out_list.append(dict(ground_truth_product=grnd_truth_prod, ground_truth_reactant=grnd_truth_react,
                             suggested_reactant=sugg_react, suggested_product=sugg_prod))
    return out_list


def produce_the_kde_plot(cycles, color, save_name):
    ground_truth_and_suggested = [(eval_code.get_best_qed_from_smiles_bag(elem['ground_truth_product']),
                                   eval_code.get_best_qed_from_smiles_bag(elem['suggested_product']))
                                         for elem in cycles]
    len_out = len(ground_truth_and_suggested)
    ground_truth_and_suggested = [elem for elem in ground_truth_and_suggested if elem[1] != -np.inf]
    len_filter = len(ground_truth_and_suggested)
    num_discarding = len_out - len_filter
    if num_discarding:
        warnings.warn(f"Discarding {num_discarding} our of {len_out} as no successful reconstruction")
    ground_truth_and_suggested = np.array(ground_truth_and_suggested)
    ground_truth_product_qed = ground_truth_and_suggested[:, 0]
    suggested_product_qed = ground_truth_and_suggested[:, 1]

    g = sns.jointplot(x=ground_truth_product_qed, y=suggested_product_qed, kind="kde", color=color,
                      )
    g.set_axis_labels("product's QED", "reconstructed product's QED", fontsize=16)
    rsquare = lambda a, b: stats.pearsonr(ground_truth_product_qed, suggested_product_qed)[0] ** 2
    g = g.annotate(rsquare, template="{stat}: {val:.2f}",
                   stat="$R^2$", loc="upper left", fontsize=12)
    print(f"Rsquare: {stats.pearsonr(ground_truth_product_qed, suggested_product_qed)[0] ** 2}")
    print(f"scipystats: {stats.linregress(ground_truth_product_qed, suggested_product_qed)}")
    plt.tight_layout()
    plt.savefig(f"{save_name}.pdf")


def plot_unreachable(params: Params, return_result_or_product):
    print("Doing unreachable")
    grnd_truth_products = _read_in_smiles_file(params.path_unreachable_products_ground_truth)
    grnd_truth_reactants = _read_in_smiles_file(params.path_unreachable_reactants)
    suggested_reactants = _read_in_smiles_file(params.path_unreachable_reactants_restrosynthezed)
    suggested_products = [return_result_or_product(reactants) for reactants in suggested_reactants]
    bundle_unreachable = _zip_together_cycle(grnd_truth_products, grnd_truth_reactants, suggested_reactants, suggested_products)
    produce_the_kde_plot(bundle_unreachable, '#e19dde', 'unreachable_qed')


def plot_reachable(params: Params, return_result_or_product):
    print("Doing reachable")
    grnd_truth_products = _read_in_smiles_file(params.path_reachable_products_ground_truth)
    grnd_truth_reactants = _react_bags_to_smi_list(params.path_react_bags_test, params.reactant_smi_to_id)
    suggested_reactants = _read_in_smiles_file(params.path_reachable_reactants_restrosynthezed)
    suggested_products = [return_result_or_product(reactants) for reactants in suggested_reactants]
    bundle_reachable = _zip_together_cycle(grnd_truth_products, grnd_truth_reactants, suggested_reactants, suggested_products)

    # We also read in the training set, so to exclude those from this set.
    processed_data_dir = mchef_config.get_processed_data_dir()
    train_reactants = _react_bags_to_smi_list(path.join(processed_data_dir, 'train_react_bags.txt'), params.reactant_smi_to_id)
    train_products = _read_in_smiles_file(path.join(processed_data_dir, 'train_products.txt'))
    assert len(train_reactants) == len(train_products)
    
    train_reactants_products = set([(rdkit_general_ops.form_canonical_smi_frozenmultiset(react), rdkit_general_ops.form_canonical_smi_frozenmultiset(prod))
                                     for react, prod in tqdm.tqdm(zip(train_reactants, train_products), total=len(train_reactants),
                                                                  desc="putting train set into a set")
                                     ])
    
    def should_filter(elem):
        reactants_set = rdkit_general_ops.form_canonical_smi_frozenmultiset(elem['ground_truth_reactant'])
        products_set = rdkit_general_ops.form_canonical_smi_frozenmultiset(elem['ground_truth_product'])
        return (reactants_set, products_set) in train_reactants_products

    bundle_reachable = [elem for elem in tqdm.tqdm(bundle_reachable) if not(should_filter(elem))]

    produce_the_kde_plot(bundle_reachable, '#56dcd6', 'reachable_qed')
    print(bundle_reachable[:5])
    print("\n\n")


def main(params: Params):
    reaction_dict = mt.get_reaction_mapping(params.tokenized_reactants,
                                            params.tokenized_products,
                                            params.nbest_for_tokenized)
    def return_result_or_product(reactant):
        return '' if len(reactant) == 0 else reaction_dict[reactant]

    # Do the plot for the reachable products
    plot_reachable(params, return_result_or_product)

    # Do the plot for unreachable
    plot_unreachable(params, return_result_or_product)






if __name__ == '__main__':
    main(Params())




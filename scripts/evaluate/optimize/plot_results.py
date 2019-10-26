"""Plot Results

Usage:
  plot_results.py <tokenized_reactants_path> <tokenized_products_path> [--nbest=<nbest>]

Options:
  --nbest=<nbest>  The number of predictions the transformer has made for each reactant set [default: 1].


"""
import pickle
import typing

import tqdm
import tabulate
from docopt import docopt
import numpy as np

from matplotlib import pyplot as plt

from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

from molecule_chef.script_helpers import eval_code
from molecule_chef.script_helpers import molecular_transformer_helper as mt


def _get_first_x_molecules(reactant_bags: typing.List[str], value_for_x: int):
    out_reactants = []
    seen_reactants = set()
    for reactant_bg in reactant_bags:
        if reactant_bg in seen_reactants or reactant_bg == '':
            continue
        out_reactants.append(reactant_bg)
        seen_reactants.add(reactant_bg)
        if len(out_reactants) >= value_for_x:
            break
    return out_reactants


class Params:
    def __init__(self):
        self.num_distinct_molecules = 10
        self.batch_size = 2000
        self.results_file = 'local_search_results.pick'

        arguments = docopt(__doc__)
        self.tokenized_reactants = arguments['<tokenized_reactants_path>']
        self.tokenized_products = arguments['<tokenized_products_path>']
        self.nbest_for_tokenized = int(arguments['--nbest'])


def main(params: Params):
    reaction_dict = mt.get_reaction_mapping(params.tokenized_reactants, params.tokenized_products, params.nbest_for_tokenized)

    with open(params.results_file, 'rb') as fo:
        local_search_results = pickle.load(fo)

    best_qeds_found = {}
    first_qeds = None
    for search_type_name, results_for_search_type in local_search_results.items():
        reactant_strs = [run_res[2] for run_res in results_for_search_type]
        first_x_reactants = [_get_first_x_molecules(elem, params.num_distinct_molecules)
                     for elem in tqdm.tqdm(reactant_strs, desc=f"Computing results for {search_type_name}")]
        first_x_products = [[reaction_dict[reactant_bg] for reactant_bg in elem] for elem in first_x_reactants]
        qeds = [[eval_code.get_best_qed_from_smiles_bag(product_bg) for product_bg in elem] for elem in first_x_products]

        # For the random search we print out a few examples of walks
        if search_type_name == 'random_search':
            for i in range(10):
                print(f"Looking at the first x molecules in search {search_type_name}, example {i}:")
                rows = [first_x_reactants[i], first_x_products[i]]
                print(tabulate.tabulate(rows))
                print('\n')

        # Find the QEDs for the starting points (or check that they are the same with this run and the last one.
        first_qeds_this_run = [elem[0] for elem in qeds]
        if first_qeds is None:
            first_qeds = first_qeds_this_run
        else:
            assert first_qeds == first_qeds_this_run

        # Add the best QED found over each individual search for plotting later:
        best_qeds = [max(elem) for elem in qeds]
        best_qeds_found[search_type_name] = best_qeds
        print('\n\n\n\n\n')

    print("Now plotting ... ")
    import seaborn as sns
    BW = "silverman"
    LW = 3
    sns.kdeplot(np.array(first_qeds), bw=BW, label="Starting locations", color='#4f8f00', linestyle='-', lw=LW)
    sns.kdeplot(np.array(best_qeds_found['random_search']), bw=BW, label="Best found with random walk", color='#f7756c',
                linestyle='--', lw=LW)
    sns.kdeplot(np.array(best_qeds_found['prop_opt']), bw=BW, label="Best found with local optimization",
                color='#09bfc5', linestyle=':', lw=LW)
    plt.legend()
    plt.xlabel('QED score')
    plt.savefig('kde_results.pdf', bbox_inches='tight')
    print("Done!")


if __name__ == '__main__':
    main(Params())

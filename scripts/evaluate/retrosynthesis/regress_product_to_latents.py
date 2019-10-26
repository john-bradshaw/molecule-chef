"""Learn regressor from products to latents.

Usage:
  regress_product_to_latents.py <input_weights>


"""

from os import path
import datetime

import tqdm
import numpy as np

from docopt import docopt

import torch
from torch.nn.utils import rnn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils import data
from torch import nn
from torch import optim

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import  Loss

from graph_neural_networks.core import utils as gnn_utils

from molecule_chef import mchef_config
from molecule_chef.model import get_mchef
from molecule_chef.data import symbol_sequence_data
from molecule_chef.data import merged_dataset
from molecule_chef.data import atom_features_dataset
from molecule_chef.chem_ops import rdkit_featurization_ops
from molecule_chef.chem_ops import rdkit_general_ops
from molecule_chef.aux_models import graph_regressors

CHKPT_FOLDER = 'chkpts'

class Params:
    def __init__(self):
        # Training details
        self.batch_size = 50
        self.num_epochs = 30
        self.log_interval = 5
        self.cuda_details = gnn_utils.CudaDetails(use_cuda=torch.cuda.is_available())

        # Molecule details
        self.gnn_hidden_size: int = 101  # our molecule features have this dimensionality.
        self.edge_names = ['single', 'double', 'triple']
        self.gnn_time_steps = 4
        self.gnn_embedding_dim = 50

        #  Data paths
        processed_data_dir = mchef_config.get_processed_data_dir()

        self.path_mol_details = path.join(processed_data_dir, 'reactants_feats.pick')

        self.path_react_bags_train = path.join(processed_data_dir, 'train_react_bags.txt')
        self.path_react_bags_val = path.join(processed_data_dir, 'valid_react_bags.txt')

        self.path_products_train = path.join(processed_data_dir, 'train_products.txt')
        self.path_products_val = path.join(processed_data_dir, 'valid_products.txt')

        # Command line arguments.
        arguments = docopt(__doc__)
        self.weights_to_use = arguments['<input_weights>']


class LatentsDatasetCreator:
    """
    Class to get the zs (encoded to by Molecule Chef) associated with a set of reactant bags listed in a txt file.
    """
    def __init__(self, params: Params):
        # == Lets get the model! ==
        indices_to_graphs = atom_features_dataset.PickledGraphDataset(params.path_mol_details, params.cuda_details)

        chkpt = torch.load(params.weights_to_use, map_location=params.cuda_details.device_str)
        # ^ Load the checkpoint as we can get some model details from there.

        latent_dim = chkpt['wae_state_dict']['latent_prior._params'].shape[1] // 2
        print(f"Inferred a latent dimensionality of {latent_dim}")
        assert chkpt['stop_symbol_idx'] == mchef_config.get_num_graphs()

        mol_chef_params = get_mchef.MChefParams(params.cuda_details, indices_to_graphs, len(indices_to_graphs),
                                                chkpt['stop_symbol_idx'], latent_dim)
        molchef_wae = get_mchef.get_mol_chef(mol_chef_params)
        molchef_wae = params.cuda_details.return_cudafied(molchef_wae)
        molchef_wae.load_state_dict(chkpt['wae_state_dict'])

        self.ae = molchef_wae
        self.cuda_details = params.cuda_details
        self.rng = np.random.RandomState(1001)
        self.latent_dim = latent_dim
        self.stop_symbol_indx = chkpt['stop_symbol_idx']

    def __call__(self, dataset_path):
        """
        :param dataset_path: This text file should specify a selection of reactant bags.
        :return: the z embeddings given by the Molecule Chef associated with the given reactant bags.
        """
        # == Read in the data and set up a Dataloader ==
        trsfm = symbol_sequence_data.TrsfmSeqStrToArray(symbol_sequence_data.StopSymbolDetails(True, self.stop_symbol_indx),
                                                        shuffle_seq_flag=True,
                                                        rng=self.rng)
        trsfm_and_tuple_indx = lambda x: trsfm(x)[0]
        reaction_bags_dataset = symbol_sequence_data.SymbolSequenceDataset(dataset_path, trsfm_and_tuple_indx)
        dataloader = DataLoader(reaction_bags_dataset, batch_size=500, shuffle=False,
                                              collate_fn=symbol_sequence_data.reorder_and_pad_then_collate)

        # == Now go through this data in batches and calculate the z embeddings. ==
        # A subtle point is that out collate function above reorders the items of the batch -- so that it is compatible
        # with the packed/padded sequences of PyTorch (where longest seqs have to come first). So we need to flip back
        # this order to give zs in the collect order.
        results = []
        with tqdm.tqdm(dataloader, total=len(dataloader)) as t:
            for padded_seq_batch_first, lengths, order in t:
                packed_seq = rnn.pack_padded_sequence(padded_seq_batch_first, lengths, batch_first=True)
                packed_seq = self.cuda_details.return_cudafied(packed_seq)

                zs = self.ae._run_through_to_z(packed_seq)
                zs_np = zs.detach().cpu().numpy()
                order_np = order.cpu().numpy()

                reverse_order = np.argsort(order_np)
                zs_np = zs_np[reverse_order]
                results.append(zs_np)
        results = np.concatenate(results)
        results_tuples = [(row,) for row in results]
        return results_tuples


class GraphDatasetCreator:
    def __call__(self, path_to_smiles_file):
        graphs = []

        with open(path_to_smiles_file, 'r') as fo:
            lines = fo.readlines()

        for mol_smi in tqdm.tqdm(lines):
            mol = rdkit_general_ops.get_molecule(mol_smi, kekulize=True)
            mol, am_to_indx_map = rdkit_general_ops.add_atom_mapping(mol)
            mol_as_adj_list = rdkit_featurization_ops.mol_to_atom_feats_and_adjacency_list(mol, am_to_indx_map)
            graph = atom_features_dataset.trfm_mol_as_adj_list_to_graph_as_adj_list_trsfm(mol_as_adj_list)
            graphs.append((graph,))

        return graphs


def _graph_and_latents_collate_func(args):
    graphs, latent_space = zip(*args)
    all_graphs = graphs[0].concatenate(graphs)
    latent_space = torch.from_numpy(np.stack(latent_space))
    return all_graphs, latent_space


class _L2Loss(nn.Module):
    def forward(self, input, target):
        return F.mse_loss(input, target, reduction='sum') / input.shape[0]  # only want the mean over the batch dimension.


def main(params: Params):
    # == Data ==
    print("\nCreating the embeddings -- ie our y variable")
    latent_dataset_creator = LatentsDatasetCreator(params)
    train_latents_dataset = latent_dataset_creator(params.path_react_bags_train)
    val_latents_dataset = latent_dataset_creator(params.path_react_bags_val)

    print("\nCreating the graphs -- ie our x variable")
    graph_dataset_creator = GraphDatasetCreator()
    train_graphs_dataset = graph_dataset_creator(params.path_products_train)
    val_graphs_dataset = graph_dataset_creator(params.path_products_val)

    train_dataset = merged_dataset.MergedDataset(train_graphs_dataset, train_latents_dataset)
    val_dataset = merged_dataset.MergedDataset(val_graphs_dataset, val_latents_dataset)

    train_dataloader = data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,
                                       collate_fn=_graph_and_latents_collate_func)
    val_dataloader = data.DataLoader(val_dataset, batch_size=500, shuffle=False,
                                     collate_fn=_graph_and_latents_collate_func)

    # == Model & Optimizer ==
    latent_dim = latent_dataset_creator.latent_dim
    print(f"Latent dim is {latent_dim}")
    model = graph_regressors.GNNThenMLP(latent_dim, params.gnn_hidden_size, params.edge_names, params.gnn_embedding_dim,
                                        params.cuda_details, params.gnn_time_steps)
    model = params.cuda_details.return_cudafied(model)
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 25], gamma=0.2)
    loss = _L2Loss()

    # == Create Training hooks for Pytorch-Ignite ==
    def _prepare_batch(batch, device, non_blocking):
        graphs, output = batch
        graphs = graphs.to_torch(params.cuda_details)
        output = params.cuda_details.return_cudafied(output)
        return graphs, output
    trainer = create_supervised_trainer(model, optimizer, loss, prepare_batch=_prepare_batch)
    evaluator = create_supervised_evaluator(model, metrics={'loss': Loss(loss)},  prepare_batch=_prepare_batch)

    desc = "ITERATION - loss: {:.5f}"
    pbar = tqdm.tqdm(initial=0, leave=False, total=len(train_dataloader), desc=desc.format(0))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_dataloader) + 1

        if iter % params.log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(params.log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_dataloader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['loss']
        tqdm.tqdm.write(
            "Training Results - Epoch: {}  Avg loss: {:.5f}"
                .format(engine.state.epoch, avg_loss)
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_dataloader)
        metrics = evaluator.state.metrics
        avg_loss = metrics['loss']
        tqdm.tqdm.write(
            "Validation Results - Epoch: {}  Avg loss: {:.5f}"
                .format(engine.state.epoch, avg_loss)
        )
        lr_scheduler.step()

        # save a checkpoint
        torch.save(dict(engine_epochs=engine.state.epoch, model_state_dict=model.state_dict()), path.join(CHKPT_FOLDER,
                                                  f"latents_to-{datetime.datetime.now().isoformat()}.pth.pick"))

        # reset progress bar
        pbar.n = pbar.last_print_n = 0

    # == Train! ==
    print("\nTraining!")
    trainer.run(train_dataloader, max_epochs=params.num_epochs)
    pbar.close()
    torch.save(dict(engine_epochs=params.num_epochs,
                    model_state_dict=model.state_dict()), path.join(CHKPT_FOLDER,
                     f"final-weights.pth.pick"))


if __name__ == '__main__':
    main(Params())

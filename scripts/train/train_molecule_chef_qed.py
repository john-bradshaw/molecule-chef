
import time
import os
from os import path
import datetime
import shutil

import numpy as np
from rdkit.Chem import QED

from tabulate import tabulate
import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils import data
from torch.nn import functional as F
from torch.nn.utils import rnn
from torch import optim

from graph_neural_networks.core import utils as gnn_utils
from autoencoders import logging_tools

from molecule_chef import mchef_config
from molecule_chef.data import symbol_sequence_data
from molecule_chef.data import atom_features_dataset
from molecule_chef.data import merged_dataset
from molecule_chef.chem_ops import rdkit_general_ops
from molecule_chef.model import get_mchef
from molecule_chef.script_helpers.eval_code import evaluate_reconstruction_accuracy
from molecule_chef.script_helpers import tensorboard_helper as tb_


TB_LOGS_FILE = 'tb_logs'
CHKPT_FOLDER = 'chkpts'


class Params(object):
    def __init__(self):
        self.run_name = str(os.getenv("MCHEF_NAME"))
        print(f"Run name is {self.run_name}")

        processed_data_dir = mchef_config.get_processed_data_dir()

        self.path_mol_details = path.join(processed_data_dir, 'reactants_feats.pick')

        self.path_react_bags_train = path.join(processed_data_dir, 'train_react_bags.txt')
        self.path_react_bags_val = path.join(processed_data_dir, 'valid_react_bags.txt')

        self.path_products_train = path.join(processed_data_dir, 'train_products.txt')
        self.path_products_val = path.join(processed_data_dir, 'valid_products.txt')

        self.num_epochs = 100
        self.batch_size = 25
        self.learning_rate = 0.001

        self.lr_reduction_interval = 40
        self.lr_reduction_factor = 0.1

        self.cuda_details = gnn_utils.CudaDetails(use_cuda=torch.cuda.is_available())

        self.lambda_value = 10.  # see WAE paper, section 4
        self.property_pred_factor = 50.
        self.latent_dim = 25


def get_train_and_val_product_property_datasets(params: Params):

    product_paths = [
        params.path_products_train,
        params.path_products_val
    ]

    def transform_text_to_qed(text_line):
        molecules = [rdkit_general_ops.get_molecule(mol_str, kekulize=False) for mol_str in text_line.split('.')]
        qed_scores = [QED.qed(mol) for mol in molecules]
        # May have many products so take max (given this is what we are optimising for in the optimisation part).
        # Expect this to be less of an issue in practice as USPTO mostly details
        # single product reactions. It may be interesting to look at using the Molecular Transformer prediction on
        # these reactions rather than this ground truth and other ways of combining multiple products eg mean.
        return np.max(qed_scores)

    dataset_out = []
    print("Creating property datasets.")
    for path_ in tqdm.tqdm(product_paths):
        with open(path_, 'r') as fo:
            lines = [x.strip() for x in fo.readlines()]

        data_all = [transform_text_to_qed(l_) for l_ in tqdm.tqdm(lines, desc=f"Processing ...{path_[-20:]}", leave=False)]

        all_array = torch.tensor(data_all)
        dataset_out.append(data.TensorDataset(all_array))
    return tuple(dataset_out)


@torch.no_grad()
def validation(val_dataloader, ae, tb_writer_val, cuda_details, property_pred_factor, lambda_value):

    # We are going to record a series of measurements that we will average after we have gone through the whole data:
    total_loss_meter = gnn_utils.AverageMeter()
    ae_obj_with_teacher_forcing_meter = gnn_utils.AverageMeter()
    acc_meter = gnn_utils.AverageMeter()
    elem_acc_meter = gnn_utils.AverageMeter()
    prediction_mse_meter = gnn_utils.AverageMeter()

    # Now we will iterate through the minibatches
    with tqdm.tqdm(val_dataloader, total=len(val_dataloader)) as t:
        for i, (padded_seq_batch_first, lengths, order, properties) in enumerate(t):
            # Set up the data
            packed_seq = rnn.pack_padded_sequence(padded_seq_batch_first, lengths, batch_first=True)
            packed_seq = cuda_details.return_cudafied(packed_seq)
            properties = cuda_details.return_cudafied(properties)

            # Evaluate reconstruction accuracy
            reconstruction, _ = ae.reconstruct_no_grad(packed_seq)
            tb_write_to_pass = tb_writer_val if i == 0 else None
            acc_bags, acc_elems = evaluate_reconstruction_accuracy(padded_seq_batch_first,
                                                                   reconstruction, tb_write_to_pass)

            # Compute the loss
            ae_obj = ae.forward(packed_seq, lambda_=lambda_value).mean()

            prediction_of_property = ae.prop_predictor_(ae._last_z_sample_on_obj)
            prop_loss = F.mse_loss(input=prediction_of_property.squeeze(), target=properties.squeeze())

            total_loss = prop_loss * property_pred_factor + -ae_obj

            # Update the meters that record the various statistics.
            batch_size = lengths.shape[0]
            total_loss_meter.update(total_loss.item(), n=batch_size)
            ae_obj_with_teacher_forcing_meter.update(ae_obj.item(), n=batch_size)
            acc_meter.update(acc_bags, n=batch_size)
            elem_acc_meter.update(acc_elems, n=batch_size)
            prediction_mse_meter.update(prop_loss.item(), n=batch_size)

            # Update the stats in the progress bar
            t.set_postfix(
                total_loss=f'{total_loss_meter.avg:.4E}',
                ae_teacher_forcing_current_avg=f'{ae_obj_with_teacher_forcing_meter.avg:.4E}',
                acc_bags=f'{acc_meter.avg:.4E}',
                acc_elems=f'{elem_acc_meter.avg:.4E}',
                pred_mse=f'{prediction_mse_meter.avg:.4E}'
            )

    print("===============================================")
    print(f"Validation finished over a total of {ae_obj_with_teacher_forcing_meter.count} items.")
    print("The final scores are:")
    print(tabulate([
        [f'Total Loss (property_pred_factor ={property_pred_factor})', total_loss_meter.avg],
        ['AE obj (teacher forcing) (lam=10)', ae_obj_with_teacher_forcing_meter.avg],
        ['Reconstruct Acc (bag level)', acc_meter.avg],
        ['Reconstruct Acc (elem level)', elem_acc_meter.avg],
        ['Prediction MSE', prediction_mse_meter.avg],
    ],
        tablefmt="simple", floatfmt=".4f"))
    print("===============================================")
    if tb_writer_val is not None:
        tb_writer_val.add_scalar("total_loss", total_loss_meter.avg)
        tb_writer_val.add_scalar("AE_obj(larger_better)", ae_obj_with_teacher_forcing_meter.avg)
        tb_writer_val.add_scalar("recon_acc(bag_level)", acc_meter.avg)
        tb_writer_val.add_scalar("recon_acc(elem_level)", elem_acc_meter.avg)
        tb_writer_val.add_scalar("property_mse", prediction_mse_meter.avg)
    return ae_obj_with_teacher_forcing_meter.avg


def train(train_dataloader, ae, optimizer, optimizer_step, cuda_details: gnn_utils.CudaDetails, tb_logger,
          lambda_value, property_pred_factor):

    # Set up some average meters so we can average readings across batches
    loss_meter = gnn_utils.AverageMeter()
    time_meter = gnn_utils.AverageMeter()
    time_on_calc = gnn_utils.AverageMeter()
    prediction_mse_meter = gnn_utils.AverageMeter()

    # Iterate through the minibatches
    pre_time = time.time()
    with tqdm.tqdm(train_dataloader, total=len(train_dataloader)) as t:
        for i, (padded_seq, lengths, order, properties) in enumerate(t):
            # Set up the data
            packed_seq = rnn.pack_padded_sequence(padded_seq, lengths, batch_first=True)
            packed_seq = cuda_details.return_cudafied(packed_seq)
            properties = cuda_details.return_cudafied(properties)
            pre_calc_time = time.time()

            if i % 100 == 0:
                # Every 100 steps we store the histograms of our sampled z's to ensure not getting posterior collapse
                ae.encoder.shallow_dist._tb_logger = tb_logger  # turn it on for this step

            # Compute the loss
            ae_obj = ae(packed_seq, lambda_=lambda_value).mean()
            prediction_of_property = ae.prop_predictor_(ae._last_z_sample_on_obj)
            prop_loss = F.mse_loss(input=prediction_of_property.squeeze(), target=properties.squeeze())

            loss = -ae_obj
            loss += property_pred_factor * prop_loss

            # Update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer_step()

            if i % 100 == 0:
                ae.encoder.shallow_dist._tb_logger = None  # turn it back off

            # Update the meters that record the various statistics.
            loss_meter.update(loss.item(), n=lengths.shape[0])
            time_meter.update(time.time() - pre_time)
            time_on_calc.update(time.time() - pre_calc_time)
            prediction_mse_meter.update(prop_loss.item(), n=lengths.shape[0])
            pre_time = time.time()
            if tb_logger is not None:
                tb_logger.add_scalar('property_mse', prop_loss.item())

            # Update the stats in the progress bar
            t.set_postfix(avg_epoch_loss=f'{loss_meter.avg:.4E}',
                          total_time=f'{time_meter.avg:.3E}', calc_time=f'{time_on_calc.avg:.3E}',
                          prop_mse=f'{prediction_mse_meter.avg:.3E}'
                          )


def collate_datasets_func(batch):
    """
    This is a custom collating function for use with datasets which  have been merged from symbol sequence data and
    a property dataset.
    """
    args_from_ds_for_seqs, args_from_ds_for_property = zip(*batch)
    args_from_ds_for_seqs = list(args_from_ds_for_seqs)
    args_from_ds_for_property = list(args_from_ds_for_property)

    ds1_collated = symbol_sequence_data.reorder_and_pad_then_collate(args_from_ds_for_seqs)
    # ^ note that this re-ordered so that items are in descending sizes..
    # Hence we need to reorder datset 2 in the same way!

    ordering = ds1_collated[2]
    ds2_collated = data.dataloader.default_collate(args_from_ds_for_property)
    ds2_collated = ds2_collated[ordering]
    return (*ds1_collated, ds2_collated)


def save_checkpoint(state, is_best, filename=None):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.pick')


def main(params: Params):
    # Set the random seeds.
    rng = np.random.RandomState(5156416)
    torch.manual_seed(rng.choice(1000000))

    # Set up data
    # == The property data
    train_prop_dataset, val_prop_dataset = get_train_and_val_product_property_datasets(params)
    print("Created property datasets!")

    # == The sequence data
    stop_symbol_idx = mchef_config.get_num_graphs() # comes after al the graphs
    trsfm = symbol_sequence_data.TrsfmSeqStrToArray(symbol_sequence_data.StopSymbolDetails(True, stop_symbol_idx),
                                                   shuffle_seq_flag=True,
                                                   rng=rng)

    reaction_bags_dataset = symbol_sequence_data.SymbolSequenceDataset(params.path_react_bags_train, trsfm)
    reaction_train_dataset = merged_dataset.MergedDataset(reaction_bags_dataset, train_prop_dataset)
    train_dataloader = DataLoader(reaction_train_dataset, batch_size=params.batch_size, shuffle=True,
                                          collate_fn=collate_datasets_func)

    reaction_bags_dataset_val = symbol_sequence_data.SymbolSequenceDataset(params.path_react_bags_val, trsfm)
    reaction_val_dataset = merged_dataset.MergedDataset(reaction_bags_dataset_val, val_prop_dataset)
    val_dataloader = DataLoader(reaction_val_dataset, batch_size=500, shuffle=False,
                                          collate_fn=collate_datasets_func)

    # == The graph data
    indices_to_graphs = atom_features_dataset.PickledGraphDataset(params.path_mol_details, params.cuda_details)
    assert stop_symbol_idx == len(indices_to_graphs), "stop symbol index should be after graphs"

    # Set up Model
    mol_chef_params = get_mchef.MChefParams(params.cuda_details, indices_to_graphs, len(indices_to_graphs),
                                            stop_symbol_idx, params.latent_dim)
    mc_wae = get_mchef.get_mol_chef(mol_chef_params)
    mc_wae = params.cuda_details.return_cudafied(mc_wae)

    # set up trainer
    optimizer = optim.Adam(mc_wae.parameters(), lr=params.learning_rate)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=params.lr_reduction_factor)

    # Set up some loggers
    tb_writer_train = tb_.get_tb_writer(f"{TB_LOGS_FILE}/{params.run_name}_train")
    tb_writer_val = tb_.get_tb_writer(f"{TB_LOGS_FILE}/{params.run_name}_val")
    def add_details_to_train(dict_to_add):
        for name, value in dict_to_add.items():
            tb_writer_train.add_scalar(name, value)
    train_log_helper = logging_tools.LogHelper([add_details_to_train])
    tb_writer_train.global_step = 0

    # Set up steps and setup funcs.
    def optimizer_step():
        optimizer.step()
        tb_writer_train.global_step += 1

    def setup_for_train():
        mc_wae._logger_manager = train_log_helper
        mc_wae.train()  # put in train mode

    def setup_for_val():
        tb_writer_val.global_step = tb_writer_train.global_step
        mc_wae._tb_logger = None  # turn off the more concise logging
        mc_wae.eval()

    # Run an initial validation
    setup_for_val()
    best_ae_obj_sofar = validation(val_dataloader, mc_wae, tb_writer_val, params.cuda_details,
                                   params.property_pred_factor, params.lambda_value)

    # Train!
    for epoch_num in range(params.num_epochs):
        print(f"We are starting epoch {epoch_num}")
        tb_writer_train.add_scalar("epoch_num", epoch_num)
        setup_for_train()
        train(train_dataloader, mc_wae, optimizer, optimizer_step, params.cuda_details, tb_writer_train,
              params.lambda_value,  params.property_pred_factor)

        print("Switching to eval.")
        setup_for_val()
        ae_obj = validation(val_dataloader, mc_wae, tb_writer_val, params.cuda_details,
                            params.property_pred_factor, params.lambda_value)

        if ae_obj >= best_ae_obj_sofar:
            print("** Best LL found so far! :-) **")
            best_ae_obj_sofar = ae_obj
            best_flag = True
        else:
            best_flag = False

        save_checkpoint(
            dict(epochs_completed=epoch_num + 1, wae_state_dict=mc_wae.state_dict(),
                 optimizer=optimizer.state_dict(), learning_rate_scheduler=lr_scheduler.state_dict(),
                 ll_from_val=ae_obj, wae_lambda_value=params.property_pred_factor,
                 stop_symbol_idx=stop_symbol_idx
                 ),
            is_best=best_flag, filename=path.join(CHKPT_FOLDER,
                                                  f"{params.run_name}-{datetime.datetime.now().isoformat()}.pth.pick"))

        # See https://github.com/pytorch/pytorch/pull/7889, in PyTorch 1.1 you have to call scheduler after:
        if (epoch_num % params.lr_reduction_interval == 0 and epoch_num / params.lr_reduction_interval > 0.9):
            print("Running the learning rate scheduler. Optimizer is:")
            lr_scheduler.step()
            print(optimizer)
        print(f"==========================================================================================")


if __name__ == '__main__':
    main(Params())


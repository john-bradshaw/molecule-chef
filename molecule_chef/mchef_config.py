
from os import path
import json

import numpy as np
import torch


def get_uspto_data_dir():
    return path.join(path.dirname(__file__), '../data')


def get_processed_data_dir():
    return path.join(path.dirname(__file__), '../processed_data')


def get_num_graphs(json_reactants_to_id_path=None):
    data = get_reactant_smi_to_reactant_id_dict(json_reactants_to_id_path)
    num_graphs = len(data)
    assert max(data.values()) < num_graphs, "ids larger than number of graphs being used."
    return num_graphs


def get_reactant_smi_to_reactant_id_dict(json_reactants_to_id_path=None):
    if json_reactants_to_id_path is None:
        json_reactants_to_id_path = path.join(get_processed_data_dir(), 'reactants_to_reactant_id.json')
    with open(json_reactants_to_id_path, 'r') as fo:
        data = json.load(fo)
    return data



FLOAT_TYPE = np.float32
PT_FLOAT_TYPE = torch.float32
PT_INT_TYPE = torch.int64
INT_TYPE = np.int64

PAD_VALUE = -10000
SOS_TOKEN = 0.  # value to use for when feeding x_0 into decoder


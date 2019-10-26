
import typing
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils import data

from .. import mchef_config


class SymbolSequenceDataset(data.Dataset):
    """
    Reads sequences from a text file where each line is an example.
    Each example consists of integers separated by commas. Each integer corresponds to a unique symbol.

    """
    def __init__(self, path_to_read, transforms=None):
        """
        :param path_to_read: str to where a pickle should be read
        """
        with open(path_to_read, 'r') as fo:
            data = fo.readlines()
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        if self.transforms is not None:
            data_item = self.transforms(data_item)
        return data_item

@dataclass
class StopSymbolDetails:
    add_stop_symbol_on_end_flag: bool = False
    stop_symbol_value: typing.Optional[int] = None  # set if add_stop_symbol_on_end_flag is True


class TrsfmSeqStrToArray:
    def __init__(self, stop_symbol_data: StopSymbolDetails,
                               shuffle_seq_flag=False,
                               rng=None):
        if rng is None:
            rng = np.random.RandomState(500)
        self.stop_symbol_data = stop_symbol_data
        self.shuffle_seq_flag = shuffle_seq_flag
        self.rng = rng

    def __call__(self, seq_str: str) -> typing.Tuple[np.ndarray]:
        """
        eg "24,125,567" -> (np.array([24,125,567]),)
        or
        "24,125,567" -> (np.array([24,125,567, <stop symbol>]),) if stop_symbol_data set appropriately
        """

        seq_elems = seq_str.split(',')
        seq_integers = list(map(int, seq_elems))
        if self.shuffle_seq_flag:
            self.rng.shuffle(seq_integers)
        if self.stop_symbol_data.add_stop_symbol_on_end_flag:
            seq_integers.append(self.stop_symbol_data.stop_symbol_value)
        return (np.array(seq_integers, mchef_config.INT_TYPE),)


def reorder_and_pad_then_collate(batch):
    """
    Readjusts the batch so that before calling default collate:
     (1) all the items are padded to the length of the longest before,
     (2) items are reordered from longest sequence to smallest value (useful for when packing)
     (3) original size and reordering that has been done is added do the data.

    :param batch:  iterable for collating from dataloader. Each data item should have sequence length in dimension 1.
    and the output should be padded_seqs, lengths, order
    """
    batch_sizes = np.array([b.shape[0] for b in batch], dtype=mchef_config.INT_TYPE)
    max_padded_size = np.max(batch_sizes)

    order = np.argsort(-batch_sizes)  # so largest first
    sorted_padded_batch = [
        (np.pad(batch[i], (0, max_padded_size - batch[i].shape[0]), mode='constant', constant_values=mchef_config.PAD_VALUE),
         batch[i].shape[0], i)
                            for i in order]

    sorted_batch_collated = data.dataloader.default_collate(sorted_padded_batch)
    return sorted_batch_collated

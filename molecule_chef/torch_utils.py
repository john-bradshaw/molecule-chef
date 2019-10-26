
import torch
from torch.nn.utils import rnn


def remove_last_from_packed_seq(symbol_in: rnn.PackedSequence) -> rnn.PackedSequence:
    padded, lengths = rnn.pad_packed_sequence(symbol_in)
    symbol_out = rnn.pack_padded_sequence(padded, lengths - 1)
    return symbol_out


def prepend_tensor_to_start_of_packed_seq(packed_seq: rnn.PackedSequence, value_to_add):
    """
    This function shifts the whole sequence down and adds value_to_add to the start.
    """

    data, batch_sizes, *others = packed_seq

    # We're gonna be a bit cheeky and construct a Packed Sequence manually at the bottom of this function -- which the
    # docs tell us not to do but have seen others do it, eg
    # https://github.com/pytorch/pytorch/issues/8921#issuecomment-400552029
    # Originally we coded this in PyTorch 1.0 and PackedSequence was a thinner wrapper on a NamedTuple
    # to continue to check that we are still using enforce_sorted=True Packed Sequences
    if len(others):
        assert others[0] is None
        assert others[1] is None

    num_in_first_batch =batch_sizes[0]
    front = torch.zeros_like(data[:num_in_first_batch])
    front[...] = value_to_add
    new_packed_seq_data = torch.cat([front, data], dim=0)
    new_length_at_beginning = batch_sizes[:1].clone()
    new_packed_seq = rnn.PackedSequence(new_packed_seq_data, torch.cat([new_length_at_beginning, packed_seq.batch_sizes]))
    return new_packed_seq


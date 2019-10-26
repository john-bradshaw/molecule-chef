
import torch
import numpy as np
from torch.nn.utils import rnn

from molecule_chef import torch_utils


def test_remove_last_from_packed_seq():
    padded_seq = torch.tensor([[1,2,3],[4,3,0], [12,18, 0]])
    orig_lengths = torch.tensor([3,2,2])

    packed_seq = rnn.pack_padded_sequence(padded_seq, orig_lengths, batch_first=True)

    computed = torch_utils.remove_last_from_packed_seq(packed_seq)
    computed_padded_seq, lengths = rnn.pad_packed_sequence(computed, batch_first=True)

    expected_computed_padded_seq = np.array([[1,2],[4,0], [12, 0]])
    expected_lengths = orig_lengths - 1

    np.testing.assert_array_equal(expected_computed_padded_seq, computed_padded_seq.numpy())
    np.testing.assert_array_equal(lengths, expected_lengths.numpy())



def test_prepend_tensor_to_start_of_packed_seq():
    padded_seq = torch.tensor([
        [[1,2], [3,10], [4,7]],
        [[8, 9], [10, 6], [11, 18]],
        [[5,12], [17, 15], [0, 0]]
    ])
    orig_lengths = torch.tensor([3,3,2])

    packed_seq = rnn.pack_padded_sequence(padded_seq, orig_lengths, batch_first=True)

    computed = torch_utils.prepend_tensor_to_start_of_packed_seq(packed_seq, 3)
    computed_padded_seq, lengths = rnn.pad_packed_sequence(computed, batch_first=True)


    expected_computed_padded_seq = np.array(
        [
            [[3, 3], [1, 2], [3, 10], [4, 7]],
            [[3, 3], [8, 9], [10, 6], [11, 18]],
            [[3, 3], [5, 12], [17, 15], [0, 0]]
        ])

    expected_lengths = np.array([4,4,3])

    np.testing.assert_array_equal(expected_computed_padded_seq, computed_padded_seq.numpy())
    np.testing.assert_array_equal(expected_lengths, lengths.numpy())

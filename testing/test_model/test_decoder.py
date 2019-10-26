
import pytest
import torch
from torch.nn.utils import rnn
from torch import nn

import numpy as np

from molecule_chef import mchef_config
from molecule_chef.model import embedders
from molecule_chef.model import decoder


@pytest.fixture
def dummy_embedder():
    num_graphs = 50
    embedding_dim = 25
    rng = np.random.RandomState(521)
    embeddings = rng.randn(num_graphs, embedding_dim).astype(mchef_config.FLOAT_TYPE)
    embeddings_t = torch.tensor(embeddings, dtype=mchef_config.PT_FLOAT_TYPE)

    dummy_graph_embedder = lambda indcs_masquerading_as_graphs: embeddings_t[indcs_masquerading_as_graphs, :]
    dummy_graph_embedder.embedding_dim = embedding_dim
    dummy_idx_to_graph_lookup = lambda indices: indices

    embedder_ = embedders.SymbolGNNEmbedder(dummy_graph_embedder, dummy_idx_to_graph_lookup, num_graphs, num_graphs)
    return embedder_


def test_that_nll_does_not_look_ahead(dummy_embedder):
    """
    Will feed in two different observations to the decoder which only differ in their last sequence values and we shall
    expect to see no change in the output probabilities as should not depend on this part.
    """
    # Setup some differing observations
    obs1 = torch.tensor([[0, 10, 49, 50],
                         [22, 9, 25, 50],
                         [26, 8, 50, mchef_config.PAD_VALUE]],
                        dtype=mchef_config.PT_INT_TYPE
                        )
    lengths = torch.tensor([4,4,3])
    obs1_packed = rnn.pack_padded_sequence(obs1, lengths, batch_first=True)


    obs2 = torch.tensor([[0, 10, 49, 14],
                         [22, 9, 25, 7],
                         [26, 8, 3, mchef_config.PAD_VALUE]],
                        dtype=mchef_config.PT_INT_TYPE
                        )
    lengths = torch.tensor([4,4,3])
    obs2_packed = rnn.pack_padded_sequence(obs2, lengths, batch_first=True)

    # Set up the decoder class:
    z_dim = 7
    gru_hsize = 10
    batch_size = 3
    rng = np.random.RandomState(1212)
    initial_z = torch.tensor(rng.randn(batch_size, z_dim),
                        dtype=mchef_config.PT_FLOAT_TYPE)
    mlp_z_to_h = nn.Linear(z_dim, gru_hsize)
    mlp_out_to_h2 = nn.Linear(gru_hsize, 4)
    decoder_params = decoder.DecoderParams(dummy_embedder.embedding_dim, gru_hsize, 2, 0., mlp_out_to_h2, mlp_z_to_h, 5)

    decoder_ = decoder.Decoder(dummy_embedder, decoder_params)



    # Replace the decoder top with this dummy class which allows us to access the logits it put into via the update op.
    class DummyDecoderTop(nn.Module):
        def __init__(self):
            self.update_val = None

        def update(self, value):
            self.update_val = value

        def nlog_like_of_obs(self, obs):
            return -1 * torch.ones_like(obs)

    decoder_.decoder_top = DummyDecoderTop()
    decoder_.update(initial_z)

    # Run through on first obs:
    decoder_.nlog_like_of_obs(obs1_packed)
    logits1 = decoder_.decoder_top.update_val.detach().cpu().numpy()
    decoder_.decoder_top.update_val = None

    # Run through on second obs:
    decoder_.nlog_like_of_obs(obs2_packed)
    logits2 = decoder_.decoder_top.update_val.detach().cpu().numpy()

    # Check they are the same
    np.testing.assert_array_almost_equal(logits1, logits2)









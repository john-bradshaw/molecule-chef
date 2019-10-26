
import torch
import numpy as np
from torch.nn.utils import rnn

from molecule_chef.model import embedders
from molecule_chef import mchef_config


def test_forward_on_packed_sequence_no_stop():

    num_graphs = 50
    embedding_dim = 100
    rng = np.random.RandomState(521)
    embeddings = rng.randn(num_graphs, embedding_dim)
    embeddings_t = torch.tensor(embeddings)

    dummy_graph_embedder = lambda indcs_masquerading_as_graphs: embeddings_t[indcs_masquerading_as_graphs, :]
    dummy_graph_embedder.embedding_dim = embedding_dim
    dummy_idx_to_graph_lookup = lambda indices: indices

    embedder_ = embedders.SymbolGNNEmbedder(dummy_graph_embedder, dummy_idx_to_graph_lookup, num_graphs, num_graphs)

    seq_batch_first = torch.tensor([
        [3,1,4,29],
        [3,1,4,mchef_config.PAD_VALUE],
        [14, 12, 0, mchef_config.PAD_VALUE],
        [1, 2, mchef_config.PAD_VALUE, mchef_config.PAD_VALUE],
        [49, mchef_config.PAD_VALUE, mchef_config.PAD_VALUE, mchef_config.PAD_VALUE],

    ])
    lengths = torch.tensor([4,3,3,2,1])
    seq_packed = rnn.pack_padded_sequence(seq_batch_first, lengths, batch_first=True)

    new_packed = embedder_.forward_on_packed_sequence(seq_packed, stops_pre_filtered_flag=True)

    padding = np.ones_like(embeddings[0]) * mchef_config.PAD_VALUE
    top_embedding = embedder_.stop_embedding.detach().cpu().numpy()
    expected_out = np.array([
        [embeddings[3], embeddings[1], embeddings[4], embeddings[29]],
        [embeddings[3], embeddings[1], embeddings[4], padding],
        [embeddings[14], embeddings[12], embeddings[0], padding],
        [embeddings[1], embeddings[2], padding, padding],
        [embeddings[49], padding, padding, padding],
    ])

    actual_padded = rnn.pad_packed_sequence(new_packed, batch_first=True,
                                            padding_value=mchef_config.PAD_VALUE)[0].detach().cpu().numpy()

    np.testing.assert_array_equal(actual_padded, expected_out)


def test_forward_on_packed_sequence_no_forward():
    num_graphs = 50
    embedding_dim = 100
    rng = np.random.RandomState(521)
    embeddings = rng.randn(num_graphs, embedding_dim).astype(mchef_config.FLOAT_TYPE)
    embeddings_t = torch.tensor(embeddings, dtype=mchef_config.PT_FLOAT_TYPE)

    dummy_graph_embedder = lambda indcs_masquerading_as_graphs: embeddings_t[indcs_masquerading_as_graphs, :]
    dummy_graph_embedder.embedding_dim = embedding_dim
    dummy_idx_to_graph_lookup = lambda indices: indices

    embedder_ = embedders.SymbolGNNEmbedder(dummy_graph_embedder, dummy_idx_to_graph_lookup, num_graphs, num_graphs)

    seq_batch_first = torch.tensor([
        [3, 1, 4, num_graphs],
        [3, 1, 4, num_graphs],
        [14, 12, 0, num_graphs],
        [1, 2, num_graphs, mchef_config.PAD_VALUE],
        [49, num_graphs, mchef_config.PAD_VALUE, mchef_config.PAD_VALUE],

    ])
    lengths = torch.tensor([4, 4, 4, 3, 2])
    seq_packed = rnn.pack_padded_sequence(seq_batch_first, lengths, batch_first=True)

    new_packed = embedder_.forward_on_packed_sequence(seq_packed, stops_pre_filtered_flag=False)

    padding = np.ones_like(embeddings[0]) * mchef_config.PAD_VALUE
    stop_embedding = embedder_.stop_embedding.detach().cpu().numpy()
    expected_out = np.array([
        [embeddings[3], embeddings[1], embeddings[4], stop_embedding],
        [embeddings[3], embeddings[1], embeddings[4], stop_embedding],
        [embeddings[14], embeddings[12], embeddings[0], stop_embedding],
        [embeddings[1], embeddings[2], stop_embedding, padding],
        [embeddings[49], stop_embedding, padding, padding],
    ])

    actual_padded = rnn.pad_packed_sequence(new_packed, batch_first=True,
                                            padding_value=mchef_config.PAD_VALUE)[0].detach().cpu().numpy()

    np.testing.assert_array_equal(actual_padded, expected_out)
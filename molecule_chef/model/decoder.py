
import typing

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import rnn
from torch.distributions import gumbel

from autoencoders import similarity_funcs
from autoencoders.dist_parameterisers import base_parameterised_distribution


from . import embedders
from .. import mchef_config
from .. import torch_utils


@dataclass
class DecoderParams:
    gru_insize: int
    gru_hsize: int
    num_layers: int
    gru_dropout: float

    mlp_out: nn.Module  # h -> h'
    mlp_parameterise_hidden: nn.Module  # latent space dim -> rnn hidden dim

    max_steps:int


class Decoder(base_parameterised_distribution.BaseParameterisedDistribution):
    """
    An RNN which sees the observations from earlier in time. z conditions the initial hidden layer.

    During nll calcs it currently sees the correct answers -- ie teacher forcing.
    """
    def __init__(self, embedder: embedders.SymbolGNNEmbedder, params: DecoderParams):
        super().__init__()
        self.embedder = embedder
        self.decoder_top = EmbeddingSimilarityDecoderTop(self.embedder)

        self.params = params
        self.gru = nn.GRU(input_size=params.gru_insize, hidden_size=params.gru_hsize,
                          num_layers=params.num_layers, dropout=params.gru_dropout)
        self.mlp_out = params.mlp_out
        self.mlp_parameterise_hidden = params.mlp_parameterise_hidden

        assert self.embedder.embedding_dim == self.params.gru_insize
        self._z_sample = None  # placeholder
        self._initial_hidden_after_update = None

    def _compute_initial_h(self, z_sample):
        h_samples = self.mlp_parameterise_hidden(z_sample)
        initial_hidden = h_samples.unsqueeze(0).repeat(self.params.num_layers, 1, 1)
        return initial_hidden  # [num_layers, b, h]

    def update(self, input_):
        self._z_sample = input_
        self._initial_hidden_after_update = self._compute_initial_h(self._z_sample)
        return self

    def mode(self):
        def sample_func(x):
            self.decoder_top.update(x)
            return self.decoder_top.mode()
        return self._run_forward_via_sampling(sample_func)

    def sample_no_grad(self, num_samples: int = 1):
        samples = []
        def sample_func(x):
            self.decoder_top.update(x)
            return self.decoder_top.sample_no_grad(1)[0]
        with torch.no_grad():
            for _ in range(num_samples):
                samples.append(self._run_forward_via_sampling(sample_func))
        return samples

    def nlog_like_of_obs(self, obs: rnn.PackedSequence):
        """
        Here we calculate the negative log likelihood of the sequence. For each  we feed in the previous observation
        ie if you use this function during training then doing teacher forcing.
        """
        # Set up the ground truth inputs from previous time-steps to be fed into the bottom of the RNN
        symbol_seq_packed_minus_last = torch_utils.remove_last_from_packed_seq(obs)
        embeddings = self.embedder.forward_on_packed_sequence(symbol_seq_packed_minus_last, stops_pre_filtered_flag=True)
        inputs = torch_utils.prepend_tensor_to_start_of_packed_seq(embeddings, mchef_config.SOS_TOKEN)

        # Feed the emebeddings through the network
        initial_hidden = self._initial_hidden_after_update
        outputs, _ = self.gru(inputs, initial_hidden)
        outputs_mapped = self.mlp_out(outputs.data)
        self.decoder_top.update(outputs_mapped)

        # Now work out the nll for each element of each sequence and then sum over the whole sequence length.
        nll_per_obs = self.decoder_top.nlog_like_of_obs(obs.data)
        nll_packed = rnn.PackedSequence(nll_per_obs, *obs[1:])
        nll_padded, _ = rnn.pad_packed_sequence(nll_packed, batch_first=True, padding_value=0.0)

        nll_per_seq = nll_padded.sum(dim=tuple(range(1, len(nll_padded.shape))))

        return nll_per_seq

    def convolve_with_function(self, obs: torch.Tensor,
                               function: similarity_funcs.BaseSimilarityFunctions) -> torch.Tensor:
        # With the WAE you are minimising an optimal transport cost, usually using the squared Euclidean distance
        # to measure distance between points. Using this metric on real output means that the first term of the WAE loss
        # matches the usual log likelihood term you would get in a VAE and it is only the second KL diveregence term that
        # is  different. Our task is more like classification where we are using a post softmax vector to represent the
        # probability of picking different reactants. So we shall still keep the first term of the WAE the same as it
        # would be in a VAE (even if this does not correspond to an optimal transport loss in our case) as we think it
        # is a sensible way to penalise reconstruction.
        return self.nlog_like_of_obs(obs)

    def _run_forward_via_sampling(self, sampling_func):
        """
        This runs the decoder forward conditioning on the samples produced so far by the sampling function.
        It returns a padded sequence BUT THIS IS NOT ORDERED BY SEQUENCE SIZE
        """
        # Create the padded sequence that we will fill in
        hidden = self._initial_hidden_after_update
        batch_size = hidden.shape[1]
        device_str = str(hidden.device)
        padded_seq_not_sorted = mchef_config.PAD_VALUE * torch.ones(self.params.max_steps, batch_size,
                                                                    dtype=mchef_config.PT_INT_TYPE,
                                            device=device_str)

        # Set up the initial input and a continue mask
        x = mchef_config.SOS_TOKEN * torch.ones(1, batch_size, self.params.gru_insize, device=device_str,
                                dtype=mchef_config.PT_FLOAT_TYPE)
        cont_indcs_into_original = torch.arange(batch_size, dtype=mchef_config.PT_INT_TYPE, device=device_str)
        lengths = torch.zeros(batch_size,  dtype=mchef_config.PT_INT_TYPE, device=device_str)

        # Go through and iterate through time steps, sampling a sequence.
        for i in range(self.params.max_steps):
            # Run through one step and sample the values for each batch member:
            op, hidden = self.gru(x, hidden)  # op [1, b', h]
            op_mapped = self.mlp_out(op.squeeze(0))  # [b', h']
            y = sampling_func(op_mapped)  # [b']

            # Fill in the sequnce values and work out  which points are going to be continued
            padded_seq_not_sorted[i, cont_indcs_into_original] = y
            lengths[cont_indcs_into_original] += 1
            cont_mask_into_current = y != self.embedder.stop_symbol_indx
            cont_indcs_into_original = cont_indcs_into_original[cont_mask_into_current]

            # if all the sequences in the batch have finished then we can break early!
            if not cont_indcs_into_original.shape[0] > 0:
                break

            # update the inputs for next time
            y = y[cont_mask_into_current]
            hidden = hidden[:, cont_mask_into_current, :]
            x = self.embedder(y, stops_pre_filtered_flag=True)
            x = x.unsqueeze(0)  # need to add a time sequence dimension.

        # cut off the unecessary padding:
        padded_seq_not_sorted = padded_seq_not_sorted[:i+1]
        return padded_seq_not_sorted, lengths


class EmbeddingSimilarityDecoderTop(base_parameterised_distribution.BaseParameterisedDistribution):
    def __init__(self, embedder: embedders.SymbolGNNEmbedder):
        super().__init__()
        self._params = None  # placeholder
        self.embedder = embedder

        # The following properties are currently fixed and define how you judge similarity between embeddings:
        def dot_product_similarity(proposed_embeddings, all_embeddings):
            """
            :param proposed_embeddings: [b, h]
            :param all_embeddings: eg [V, h]
            """
            return proposed_embeddings @ all_embeddings.transpose(0,1)
        self.embedding_similarity_func_batch_on_batch = dot_product_similarity

    def update(self, x: torch.Tensor):
        self._params = x

    def mode(self) -> torch.Tensor:
        logits = self._create_logits_on_all_vocab()
        mode = torch.argmax(logits, dim=1)
        return mode

    @torch.no_grad()
    def sample_no_grad(self, num_samples: int = 1) -> typing.List[torch.Tensor]:
        """
        Samples this distribution with no gradients flowing back.
        """
        samples = []
        gumbel_dist = gumbel.Gumbel(0, 1)
        for _ in range(num_samples):
            logits = self._create_logits_on_all_vocab()
            logits_plus_gumbel_noise = logits + \
                                       gumbel_dist.sample(sample_shape=logits.shape).to(str(logits.device))
            mode = torch.argmax(logits_plus_gumbel_noise, dim=1)
            samples.append(mode)
        return samples

    def nlog_like_of_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # For the case where we dont sample we form the logits matrix by
        # computing it for all possible candidate reactants.
        logits = self._create_logits_on_all_vocab()
        loss = F.cross_entropy(logits, target=obs, reduction='none')
        return loss

    def _create_logits_on_all_vocab(self):
        all_embeddings = self.embedder.all_embeddings  # V, h
        logits = self.embedding_similarity_func_batch_on_batch(self._params, all_embeddings)
        return logits


def get_decoder(gnn_embedder: embedders.SymbolGNNEmbedder, params: DecoderParams):
    return Decoder(gnn_embedder, params)

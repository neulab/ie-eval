import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class MLP(nn.Module):

    """A multi-layer perceptron with Dropout after each linear layer."""

    _ACTIVATION = {
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "none": lambda x: x,
    }

    def __init__(
        self,
        hidden_dims: List[int],
        activations: List[str] = "none",
        dropouts: Optional[List[float]] = None,
        biases: Optional[List[bool]] = None,
    ):
        super().__init__()

        assert len(hidden_dims) >= 0
        self._input_dim = hidden_dims[0]
        self._output_dim = hidden_dims[-1]

        if biases is None:
            biases = [True] * (len(hidden_dims) - 1)
        linears = [
            nn.Linear(d, hidden_dims[i + 1], bias=b)
            for i, (d, b) in enumerate(zip(hidden_dims[:-1], biases))
        ]
        self.linears = nn.ModuleList(linears)
        self.activations = [self._ACTIVATION[act] for act in activations]

        if isinstance(dropouts, int):
            drop_layer = nn.Dropout(dropouts)
            self.dropouts = nn.ModuleList(
                [copy.copy(drop_layer) for _ in enumerate(hidden_dims[:-1])]
            )
        elif isinstance(dropouts, list):
            self.dropouts = nn.ModuleList([nn.Dropout(d) for d in dropouts])
        else:
            self.dropouts = [None for _ in enumerate(hidden_dims[:-1])]

    def forward(self, input):
        hidden = input
        for l, act, drop in zip(self.linears, self.activations, self.dropouts):
            hidden = l.forward(hidden)
            hidden = act(hidden)
            if drop is not None:
                hidden = drop.forward(hidden)
        return hidden


class MultiEmbedding(nn.Module):

    """a thin wrapper embedding module which takes multiple indices tensors and
    return concatenated embedings.
    """

    def __init__(
        self,
        vocab_sizes: List[int],
        embed_dims: List[int],
        padding_idxs: Optional[List[int]] = None,
    ):
        super().__init__()

        assert len(vocab_sizes) == len(embed_dims)

        self._vocab_sizes = vocab_sizes
        self._embed_dims = embed_dims
        self._padding_idxs = padding_idxs
        if padding_idxs is None:
            self._padding_idxs = [1 for _ in vocab_sizes]

        self.embs = []
        for vsize, edim, p in zip(vocab_sizes, embed_dims, padding_idxs):
            self.embs.append(nn.Embedding(vsize, edim, padding_idx=p))
        self.embs = nn.ModuleList(self.embs)

        # self.weight_init()

    def forward(self, inputs):
        assert len(inputs) == len(self._vocab_sizes)
        return torch.cat([emb(ip) for ip, emb in zip(inputs, self.embs)], dim=2)

    def pad_init(self):
        for i, emb in enumerate(self.embs):
            emb.weight.data[self._padding_idxs[i], :].zero_()


class BLSTMModel(nn.Module):

    """ Bidirectional LSTM followed by a max-pooling and a MLP."""

    def __init__(
        self,
        vocab_sizes,
        embed_dims,
        hidden_dim,
        fc_hidden_dim,
        label_size,
        padding_idxs,
        dropout,
    ):
        super().__init__()

        if isinstance(vocab_sizes, list):
            assert len(vocab_sizes) == len(embed_dims)

        self._vocab_sizes = vocab_sizes
        self._embed_dims = embed_dims
        self._padding_idxs = padding_idxs
        self._hidden_dim = hidden_dim
        self._fc_hidden_dim = fc_hidden_dim
        self._label_size = label_size

        self.embed = MultiEmbedding(vocab_sizes, embed_dims, padding_idxs)
        self.lstm = nn.LSTM(
            sum(self._embed_dims),
            self._hidden_dim,
            bidirectional=True,
            batch_first=True,
        )
        self.MLP = MLP(
            [self._hidden_dim * 2, self._fc_hidden_dim, self._label_size],
            activations=["relu", "none"],
            dropouts=[dropout, 0],
        )

    def forward(self, inputs: List[torch.Tensor], sort_lengths, sort_idx, org_idx):

        # B x max_len x sum(embed_dims)
        embs = self.embed(inputs)
        embs = embs[sort_idx]

        packed_embs = pack(embs, lengths=sort_lengths, batch_first=True)
        outs, hiddens = self.lstm(packed_embs)
        unpacked_outs, _ = unpack(outs, batch_first=True)

        aggregated, _ = torch.max(unpacked_outs[org_idx], dim=1)
        logit = self.MLP(aggregated)
        return F.softmax(logit, dim=1)

    def get_loss(self, batch, predict=False, return_prob=False):
        inputs = [
            batch.sentences,
            batch.features["entdists"],
            batch.features["numdists"],
        ]
        prob = self.forward(inputs, batch.sort_lengths, batch.sort_idx, batch.org_idx)
        log_likelihood = 0
        for i, l in enumerate(batch.labels):
            log_likelihood = log_likelihood + torch.log(torch.mean(prob[i, l]))

        # size average
        log_likelihood = log_likelihood / prob.size(0)

        if return_prob:
            return -log_likelihood, prob

        if predict:
            return -log_likelihood, torch.max(prob, 1)[1]

        return -log_likelihood


class ConvModel(nn.Module):

    """CNN with 3 filter widths, followed by a MLP."""

    def __init__(
        self,
        vocab_sizes,
        embed_dims,
        hidden_dim,
        fc_hidden_dim,
        label_size,
        padding_idxs,
        num_filters,
        dropout,
        filter_widths=[2, 3, 5],
    ):
        super().__init__()
        self._vocab_sizes = vocab_sizes
        self._embed_dims = embed_dims
        self._padding_idxs = padding_idxs
        self._hidden_dim = hidden_dim
        self._fc_hidden_dim = fc_hidden_dim
        self._label_size = label_size
        self._filter_widths = filter_widths
        self._num_filters = num_filters

        self.embed = MultiEmbedding(vocab_sizes, embed_dims, padding_idxs)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(1, self._num_filters, (k, sum(self._embed_dims)), stride=1)
                for k in filter_widths
            ]
        )
        self.MLP = MLP(
            [
                self._num_filters * len(self._filter_widths),
                self._fc_hidden_dim,
                self._label_size,
            ],
            activations=["relu", "none"],
            dropouts=[dropout, 0],
        )

        self.drop = nn.Dropout(dropout)

        self.weight_init()

    @staticmethod
    def conv_and_pool(x, conv):
        x = F.relu(conv(x)).squeeze(-1)
        x = F.max_pool1d(x, x.size(2)).squeeze(-1)
        return x

    def forward(self, inputs):
        emb = self.embed(inputs).unsqueeze(1)

        conved = torch.cat(
            [self.conv_and_pool(emb, conv) for conv in self.convs], dim=1
        )
        conved = self.drop(conved)
        logit = self.MLP(conved)
        return F.log_softmax(logit, dim=1)

    def get_loss(self, batch, predict=False, return_prob=False):
        inputs = [
            batch.sentences,
            batch.features["entdists"],
            batch.features["numdists"],
        ]
        prob = self.forward(inputs)
        log_likelihood = 0
        for i, l in enumerate(batch.labels):
            log_likelihood = log_likelihood + torch.logsumexp(prob[i, l], dim=0)

        # size average
        log_likelihood = log_likelihood / prob.size(0)

        if return_prob:
            return -log_likelihood, prob

        if predict:
            return -log_likelihood, torch.max(prob, 1)[1]

        return -log_likelihood

    def weight_init(self):
        for conv in self.convs.parameters():
            conv.data.uniform_(-0.1, 0.1)

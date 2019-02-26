# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from ...utils.nn import get_rnn_hidden_state
from .. import FF
from .rnn import RNN


class Generator(RNN):
    """A decoder which implements Show-attend-and-tell decoder."""
    def __init__(self, input_size, hidden_size, ctx_size_dict, ctx_name, n_vocab,
                 rnn_type, tied_emb=False, dec_init='zero', dropout=0,
                 emb_maxnorm=None, emb_gradscale=False,
                 bos_type='emb'):

        super().__init__(rnn_type, dec_init)

        # Other arguments
        self.n_vocab = n_vocab
        self.dropout = dropout
        self.tied_emb = tied_emb
        self.ctx_name = ctx_name
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.emb_maxnorm = emb_maxnorm
        self.emb_gradscale = emb_gradscale
        self.ctx_size_dict = ctx_size_dict
        self.bos_type = bos_type


        # Create target embeddings
        self.emb = nn.Embedding(self.n_vocab, self.input_size,
                                padding_idx=0, max_norm=self.emb_maxnorm,
                                scale_grad_by_freq=self.emb_gradscale)

        # Dropout
        if self.dropout > 0:
            self.do = nn.Dropout(p=self.dropout)

        # Create decoder from [y_t, z_t] to dec_dim
        self.dec0 = self.RNN(self.input_size, self.hidden_size)
        self.dec1 = self.RNN(self.hidden_size, self.hidden_size)

        #attention
        self.att = FF(self.ctx_size_dict[self.ctx_name], self.hidden_size, activ="tanh")

        # Final softmax (lang)
        self.hid2out = FF(self.hidden_size, self.input_size,
                         bias_zero=True, activ='tanh')
        self.out2prob = FF(self.input_size, self.n_vocab)


        # Tie input embedding matrix and output embedding matrix
        if self.tied_emb:
            self.out2prob.weight = self.emb.weight

    def f_next(self, ctx_dict, y, h):
        """Applies one timestep of recurrence."""
        # Get hidden states from the first decoder (purely cond. on LM)
        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)

        ct = self.att(ctx_dict[self.ctx_name][0])

        h1_ct = torch.mul(h1,ct)

        # Run second decoder (h1 is compatible now as it was returned by GRU)
        h2_c2 = self.dec1(h1_ct.squeeze(0), h1_c1)
        h2 = get_rnn_hidden_state(h2_c2)

        # This is a bottleneck to avoid going from H to V directly
        logit = self.hid2out(h2)

        # Apply dropout if any
        if self.dropout > 0:
            logit = self.do(logit)

        # Transform logit to T*B*V (V: vocab_size)
        # Compute log_softmax over token dim
        prob = F.softmax(self.out2prob(logit), dim=-1)


        # Return log probs and new hidden states
        return prob, self._rnn_pack_states(h2_c2)

    def forward(self, ctx_dict, y):

        probs = torch.zeros(
            y.shape[0] - 1, y.shape[1], self.n_vocab, device=y.device)

        # Convert token indices to embeddings -> T*B*E
        y_emb = self.emb(y)

        # Get initial hidden state
        h = self.f_init(ctx_dict)

        # -1: So that we skip the timestep where input is <eos>
        for t in range(y_emb.shape[0] - 1):
            prob, h = self.f_next(ctx_dict, y_emb[t], h)
            probs[t] = prob.data


        return probs

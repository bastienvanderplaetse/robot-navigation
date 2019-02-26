# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from ...utils.nn import get_rnn_hidden_state
from .. import FF
from .rnn import RNN


class Discriminator(RNN):
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

        # Create decoder from [y_t, z_t] to dec_dim
        self.dec0 = self.RNN(self.input_size, self.hidden_size)
        self.dec1 = self.RNN(self.hidden_size, self.hidden_size)

        #attention
        self.att = FF(self.ctx_size_dict[self.ctx_name], self.hidden_size, activ="tanh")

        # Final softmax (classif)
        self.out2prob_classif = FF(self.hidden_size, 1, activ="sigmoid")


    def f_next(self, ctx_dict, y, h):
        """Applies one timestep of recurrence."""

        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)

        h1_ct = torch.mul(h1,self.att(ctx_dict[self.ctx_name][0]))

        h2_c2 = self.dec1(h1_ct.squeeze(0), h1_c1)

        return self._rnn_pack_states(h2_c2)

    def forward(self, ctx_dict, y, one_hot=True):

        # Convert token indices to embeddings -> T*B*E
        if one_hot:
            y_emb = self.emb(y)
        else:
            y_emb = torch.matmul(y,self.emb.weight)

        # Get initial hidden state
        h = self.f_init(ctx_dict)

        # -1: So that we skip the timestep where input is <eos>
        for t in range(y_emb.shape[0] - 1):
            h = self.f_next(ctx_dict, y_emb[t], h)

        valid = self.out2prob_classif(h)

        #accuracy
        # max_vals, max_indices = torch.max(log_c, 1)
        # train_acc = (max_indices == label).sum().data.cpu().numpy() / max_indices.size()[0]
        # print(train_acc)

        return valid

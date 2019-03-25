# -*- coding: utf-8 -*-
import torch
from torch import nn


from ...utils.nn import get_rnn_hidden_state
from .. import FF
from .rnn import RNN


class Discriminator(RNN):
    """A decoder which implements Show-attend-and-tell decoder."""
    def __init__(self, input_size, hidden_size, ctx_size_dict, ctx_name, n_vocab,
                 rnn_type, emb, dec0, tied_emb=False, dec_init='zero', dropout=0,
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

        self._init_func = getattr(self, '_rnn_init_{}'.format(dec_init))


        # Create target embeddings
        self.emb = emb

        # Create decoder from [y_t, z_t] to dec_dim
        # self.dec0 = dec0
        self.dec0 = self.RNN(self.input_size, self.hidden_size)
        self.dec1 = self.RNN(self.hidden_size, self.hidden_size)

        #attention
        self.att = FF(self.ctx_size_dict[self.ctx_name], self.hidden_size, activ="tanh")

        # Final softmax (classif)
        self.out2prob_classif = FF(self.hidden_size, 1, activ="sigmoid")
        # self.out2prob_classif = FF(self.hidden_size, 1)

        # Dropout
        if self.dropout > 0:
            self.do = nn.Dropout(p=self.dropout)

    def f_next(self, ctx_dict, y, h):
        """Applies one timestep of recurrence."""
        # Apply dropout if any
        # if self.dropout > 0:
        #     y = self.do(y)


        h1_c1 = self.dec0(y, h)

        h1 = get_rnn_hidden_state(h1_c1)

        ct = self.att(ctx_dict[self.ctx_name][0]).squeeze(0)
        #
        h1_ct = torch.mul(h1,ct)
        #
        o = self.dec1(h1_ct, h1_c1)


        return o, o

    def forward(self, ctx_dict, y, one_hot=True):

        # Convert token indices to embeddings -> T*B*E
        if one_hot:
            y_emb = self.emb(y)
        else:
            y_emb = torch.matmul(y,self.emb.weight)

        # Get initial hidden state
        h = self.f_init(ctx_dict)



        # -1: So that we skip the timestep where input is <eos>
        for t in range(y_emb.shape[0]):
            o,h = self.f_next(ctx_dict, y_emb[t], h)
        # if self.dropout > 0:
        #     h = self.do(h)
        valid = self.out2prob_classif(o)

        #accuracy
        # max_vals, max_indices = torch.max(log_c, 1)
        # train_acc = (max_indices == label).sum().data.cpu().numpy() / max_indices.size()[0]
        # print(train_acc)

        return valid

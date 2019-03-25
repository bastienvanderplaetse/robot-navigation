# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from ...utils.nn import get_rnn_hidden_state
from .. import FF
from .rnn import RNN
from ...utils.data import onehot_batch_data
from . import GRUCell

class Generator(RNN):
    """A decoder which implements Show-attend-and-tell decoder."""
    def __init__(self, input_size, hidden_size, ctx_size_dict, ctx_name, n_vocab,
                 rnn_type,rnn_type2, emb, dec0, tied_emb=False, dec_init='zero', dropout=0,
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
        self.z = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))

        self._init_func = getattr(self, '_rnn_init_{}'.format(dec_init))

        # Create target embeddings
        self.emb = emb
        # Dropout
        if self.dropout > 0:
            self.do = nn.Dropout(p=self.dropout)

        # self.dec0 = self.RNN(self.input_size, self.n_vocab, self.hidden_size)
        self.dec0 = self.RNN(self.input_size, self.hidden_size)
        # self.dec0 = dec0

        second = getattr(GRUCell, '{}'.format(rnn_type2))

        self.dec1 = second(self.hidden_size, self.hidden_size)

        #attention
        self.att = FF(self.ctx_size_dict[self.ctx_name], self.hidden_size, activ="tanh")

        # Final softmax (lang)
        self.hid2out = FF(self.hidden_size, self.input_size,
                         bias_zero=True, activ='tanh')
        self.out2prob = FF(self.input_size, self.n_vocab)

        # Tie input embedding matrix and output embedding matrix
        if self.tied_emb:
            self.out2prob.weight = self.emb.weight

    def f_next(self, ctx_dict, y, prob, h):
        """Applies one timestep of recurrence."""
        #
        if self.dropout > 0:
            y = self.do(y)

        # y_ct = torch.cat([ct,y],dim=1)
        # Get hidden states from the first decoder (purely cond. on LM)
        # h1_c1 = self.dec0(y, prob, h)
        h1_c1 = self.dec0(y, h)

        h1 = get_rnn_hidden_state(h1_c1)

        ct = self.att(ctx_dict[self.ctx_name][0]).squeeze(0)
        h1_ct = torch.mul(h1,ct)

        # Run second decoder (h1 is compatible now as it was returned by GRU)
        h2_c2 = self.dec1(h1_ct, h1_c1)
        h2 = get_rnn_hidden_state(h2_c2)

        # This is a bottleneck to avoid going from H to V directly
        logit = self.hid2out(h2)

        # Apply dropout if any
        # if self.dropout > 0:
        #     logit = self.do(logit)


        prob = F.softmax(self.out2prob(logit), dim=-1)


        # Return log probs and new hidden states
        return prob, h1_c1


    def f_probs(self, batch_size, n_vocab, device):
        return torch.zeros(
            batch_size, n_vocab, device=device)

    def forward(self, ctx_dict, y):


        omask = (y != 0).long()
        if(omask == 0).nonzero().numel():
            import sys
            sys.exit("non aligned input warning")


        # sentence = y[1:-1]
        # bos = y[:1]
        # eos = y[-1:]
        #
        # # sentence = y[-2:-1]
        # # bos = y[:-2]
        # # eos = y[-1:]
        # sentence = y
        # z = self.z.rsample(torch.Size([sentence.shape[], sentence.shape[1],self.n_vocab])).squeeze(-1).to(y.device)
        # y = torch.matmul(z, self.emb.weight)
        #
        # token = y[:1]
        # y = y[:-1]
        # z = self.z.rsample(torch.Size([1, token.shape[1], self.input_size])).squeeze(-1).to(y.device)
        # y = self.emb(y)
        # y = torch.cat((z,y), dim=0)



        # y = torch.cat((self.emb(bos), subsentence_noise, self.emb(eos)), dim=0)

        # Convert token indices to embeddings -> T*B*E
        #
        # z = self.z.rsample(torch.Size([y.shape[0], y.shape[1],self.n_vocab])).squeeze(-1).to(y.device)
        # y = torch.matmul(z, self.emb.weight)

        y = self.emb(y)

        # Get initial hidden state
        h = self.f_init(ctx_dict)

        # import random
        # if (random.randint(0,50)==5):
        #     print(self.emb.weight)

        probs = torch.zeros(
            y.shape[0], y.shape[1], self.n_vocab, device=y.device)


        prob = self.f_probs(y.shape[1], self.n_vocab, y.device)

        for t in range(y.shape[0]):
            prob, h = self.f_next(ctx_dict, y[t], prob, h)
            probs[t] = prob

        return probs

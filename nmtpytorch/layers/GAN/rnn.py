import torch
from torch import nn
from .. import FF
from collections import defaultdict

class RNN(nn.Module):
    def __init__(self, rnn_type, dec_init):

        super().__init__()

        self.rnn_type = rnn_type.upper()
        self.dec_init = dec_init


        # Safety checks
        assert self.rnn_type in ('GRU', 'LSTM'), \
            "rnn_type '{}' not known".format(rnn_type)
        assert dec_init in ('zero', 'mean_ctx'), \
            "dec_init '{}' not known".format(dec_init)

        self.RNN = getattr(nn, '{}Cell'.format(self.rnn_type))
        # LSTMs have also the cell state
        self.n_states = 1 if self.rnn_type == 'GRU' else 2

        # Set custom handlers for GRU/LSTM
        if self.rnn_type == 'GRU':
            self._rnn_unpack_states = lambda x: x
            self._rnn_pack_states = lambda x: x
        elif self.rnn_type == 'LSTM':
            self._rnn_unpack_states = self._lstm_unpack_states
            self._rnn_pack_states = self._lstm_pack_states

        # Set decoder initializer
        self._init_func = getattr(self, '_rnn_init_{}'.format(dec_init))

        # Decoder initializer FF (for mean_ctx)
        if self.dec_init == 'mean_ctx': #for gan, dec_init is 0
            self.ff_dec_init = FF(
                self.ctx_size_dict[self.ctx_name],
                self.hidden_size * self.n_states, activ='tanh')

    def _lstm_pack_states(self, h):
        return torch.cat(h, dim=-1)

    def _lstm_unpack_states(self, h):
        # Split h_t and c_t into two tensors and return a tuple
        return torch.split(h, self.hidden_size, dim=-1)

    def _rnn_init_zero(self, ctx, ctx_mask):
        return torch.zeros(
            ctx.shape[1], self.hidden_size * self.n_states, device=ctx.device)

    def _rnn_init_mean_ctx(self, ctx, ctx_mask):
        mean_ctx = ctx.mean(dim=0)
        if self.dropout > 0:
            mean_ctx = self.do(mean_ctx)
        return self.ff_dec_init(mean_ctx)

    def f_init(self, ctx_dict):
        """Returns the initial h_0, c_0 for the decoder."""
        self.history = defaultdict(list)
        return self._init_func(*ctx_dict[self.ctx_name])

    def get_emb(self, idxs, tstep):
        """Returns time-step based embeddings."""
        if tstep == 0:
            if self.bos_type == 'emb':
                # Learned <bos> embedding
                return self.emb(idxs)
            elif self.bos_type == 'zero':
                # Constant-zero <bos> embedding
                return torch.zeros(
                    idxs.shape[0], self.input_size, device=idxs.device)
            else:
                # Feature-based <bos> computed in f_init()
                return self.bos
        # For other timesteps, look up the embedding layer
        return self.emb(idxs)
import torch
from torch import nn
from .. import FF
from collections import defaultdict
from . import GRUCell
from . import LSTMCell

class RNN(nn.Module):
    def __init__(self, rnn_type, dec_init):

        super().__init__()

        self.dec_init = dec_init

        # self.RNN = getattr(GRUCell, '{}'.format(rnn_type))
        self.RNN = getattr(GRUCell, '{}'.format(rnn_type))
        self.n_states = 1


    def _rnn_init_zero(self, ctx, ctx_mask):
        h = torch.zeros(
            ctx.shape[1], self.hidden_size, device=ctx.device)
        if self.n_states == 2:
            return (h,h)
        return h

    def _rnn_init_mean_ctx(self, ctx, ctx_mask):
        mean_ctx = ctx.mean(dim=0)
        if self.dropout > 0:
            mean_ctx = self.do(mean_ctx)
        return self.ff_dec_init(mean_ctx)

    def _rnn_init_random(self, ctx, ctx_mask):
        """Returns the initial h_0, c_0 for the decoder."""
        return self.z.rsample(torch.Size([ctx.shape[1], self.hidden_size])).squeeze(-1).to(ctx.device)

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
            elif self.bos_type == 'random':
                # return self.emb(self.z.rsample(torch.Size([idxs.shape[0]])).squeeze(-1).to(idxs.device))
                return self.z.rsample(torch.Size([idxs.shape[0], self.input_size])).squeeze(-1).to(idxs.device)
            else:
                # Feature-based <bos> computed in f_init()
                return self.bos
        # For other timesteps, look up the embedding layer
        return self.emb(idxs)
# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn

from ..utils.misc import get_n_params
from ..vocabulary import Vocabulary
from ..utils.topology import Topology
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..datasets import MultimodalDataset
from ..metrics import Metric

logger = logging.getLogger('nmtpytorch')


class NMT(nn.Module):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'emb_dim': 128,             # Source and target embedding sizes
            'emb_maxnorm': None,        # Normalize embeddings l2 norm to 1
            'emb_gradscale': False,     # Scale embedding gradients w.r.t. batch frequency
            'enc_dim': 256,             # Encoder hidden size
            'enc_type': 'gru',          # Encoder type (gru|lstm)
            'enc_lnorm': False,         # Add layer-normalization to encoder output
            'n_encoders': 1,            # Number of stacked encoders
            'dec_dim': 256,             # Decoder hidden size
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_variant': 'cond',      # (cond|simplegru|vector)
            'dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx/feats)
            'dec_init_size': None,      # feature vector dimensionality for
            'dec_init_activ': 'tanh',   # Decoder initialization activation func
                                        # dec_init == 'feats'
            'att_type': 'mlp',          # Attention type (mlp|dot)
            'att_temp': 1.,             # Attention temperature
            'att_activ': 'tanh',        # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': False,      # Enables bias in attention mechanism
            'att_bottleneck': 'ctx',    # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,  # Transform annotations before attention
            'dropout_emb': 0,           # Simple dropout to source embeddings
            'dropout_ctx': 0,           # Simple dropout to source encodings
            'dropout_out': 0,           # Simple dropout to decoder output
            'dropout_enc': 0,           # Intra-encoder dropout if n_encoders > 1
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': 80,              # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       # Curriculum: ascending/descending/None
            'sampler_type': 'bucket',   # bucket or approximate
            'sched_sampling': 0,        # Scheduled sampling ratio
            'bos_type': 'emb',          # 'emb': default learned emb
            'bos_activ': None,          #
            'bos_dim': None,            #
        }

    def __init__(self, opts):
        super().__init__()

        # opts -> config file sections {.model, .data, .vocabulary, .train}
        self.opts = opts

        # Vocabulary objects
        self.vocabs = {}

        # Each auxiliary loss should be stored inside this dictionary
        # in order to be taken into account by the mainloop for multi-tasking
        self.aux_loss = {}

        # Setup options
        self.opts.model = self.set_model_options(opts.model)

        # Parse topology & languages
        self.topology = Topology(self.opts.model['direction'])

        # Load vocabularies here
        for name, fname in self.opts.vocabulary.items():
            self.vocabs[name] = Vocabulary(fname, name=name)

        # Inherently non multi-lingual aware
        slangs = self.topology.get_src_langs()
        tlangs = self.topology.get_trg_langs()
        if slangs:
            self.sl = slangs[0]
            self.src_vocab = self.vocabs[self.sl]
            self.n_src_vocab = len(self.src_vocab)
        if tlangs:
            self.tl = tlangs[0]
            self.trg_vocab = self.vocabs[self.tl]
            self.n_trg_vocab = len(self.trg_vocab)
            # Need to be set for early-stop evaluation
            # NOTE: This should come from config or elsewhere
            self.val_refs = self.opts.data['val_set'][self.tl]

        # Textual context size is always equal to enc_dim * 2 since
        # it is the concatenation of forward and backward hidden states
        if 'enc_dim' in self.opts.model:
            self.ctx_sizes = {str(self.sl): self.opts.model['enc_dim'] * 2}

        # Check vocabulary sizes for 3way tying
        if self.opts.model['tied_emb'] not in [False, '2way', '3way']:
            raise RuntimeError(
                "'{}' not recognized for tied_emb.".format(self.opts.model['tied_emb']))
        if self.opts.model['tied_emb'] == '3way':
            assert self.n_src_vocab == self.n_trg_vocab, \
                "The vocabulary sizes do not match for 3way tied embeddings."

    def __repr__(self):
        s = super().__repr__() + '\n'
        for vocab in self.vocabs.values():
            s += "{}\n".format(vocab)
        s += "{}\n".format(get_n_params(self))
        return s

    def set_model_options(self, model_opts):
        self.set_defaults()
        for opt, value in model_opts.items():
            if opt in self.defaults:
                # Override defaults from config
                self.defaults[opt] = value
            else:
                logger.info('Warning: unused model option: {}'.format(opt))
        return self.defaults

    def reset_parameters(self):
        for name, param in self.named_parameters():
            # Skip 1-d biases and scalars
            if param.requires_grad and param.dim() > 1:
                nn.init.kaiming_normal_(param.data)
        # # Reset padding embedding to 0
        # with torch.no_grad():
        #     self.enc.emb.weight.data[0].fill_(0)

    def setup(self, is_train=True):
        """Sets up NN topology by creating the layers."""

        # Share encoder and decoder weights
        if self.opts.model['tied_emb'] == '3way':
            self.enc.emb.weight = self.dec.emb.weight

    def get_bos(self, batch_size):
        """Returns a representation for <bos> embeddings for decoding."""
        return torch.LongTensor(batch_size).fill_(self.trg_vocab['<bos>'])


    def forward(self, batch, **kwargs):
        pass

# -*- coding: utf-8 -*-
import logging

import torch
from ..layers import Generator
from ..layers import Discriminator
import numpy as np
from torch.autograd import Variable
from ..datasets import MultimodalDataset
import torch.autograd as autograd
from torch import nn
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..metrics import Metric

from .nmt import NMT

logger = logging.getLogger('nmtpytorch')


class GAN(NMT):
    r"""An Implementation of 'Show, attend and tell' image captioning paper.

    Paper: http://www.jmlr.org/proceedings/papers/v37/xuc15.pdf
    Reference implementation: https://github.com/kelvinxu/arctic-captions
    """
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            'emb_dim': 128,             # Source and target embedding sizes
            'emb_maxnorm': None,        # Normalize embeddings l2 norm to 1
            'emb_gradscale': False,     # Scale embedding gradients w.r.t. batch frequency
            'dec_dim': 256,             # Decoder hidden size
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_init': 'zero',         # How to initialize decoder (zero/mean_ctx)
            'dropout': 0,               # Simple dropout
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'direction':None,
            'feat_dim': 2048,
        }

    def __init__(self, opts):
        super().__init__(opts)


    def setup(self, is_train=True):

        # Number of channels defines the spatial vector dim for us
        self.ctx_sizes = {'feats': self.opts.model['feat_dim']}
        self.adversarial_loss = nn.BCELoss()

        # Create Decoder
        self.G = Generator(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name='feats',
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            dropout=self.opts.model['dropout'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
        )

        self.D = Discriminator(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name='feats',
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            dropout=self.opts.model['dropout'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
        )



    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data[split + '_set'],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model.get('max_len', None),
            warmup=(split != 'train'),
        )
        logger.info(dataset)
        return dataset

    def encode(self, batch, **kwargs):
        feats = (batch['feats'])
        return {'feats': (feats, None)}

    def forward(self, batch, optim_G, optim_D, train=True, **kwargs):
        ret = {}
        valid = Variable(torch.cuda.FloatTensor(batch.size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.cuda.FloatTensor(batch.size, 1).fill_(0.0), requires_grad=False)

        feature = self.encode(batch)
        # -----------------
        #  Train Generator
        # -----------------

        if train:
            optim_G.zero_grad()

        gen_s = self.G(feature, batch[self.tl])
        g_loss = self.adversarial_loss(self.D(feature, gen_s, one_hot=False), valid)

        if train:
            g_loss.backward()
            optim_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        if train:
            optim_D.zero_grad()

        real_loss = self.adversarial_loss(self.D(feature, batch[self.tl]), valid)
        fake_loss = self.adversarial_loss(self.D(feature, gen_s.detach(), one_hot=False), fake)

        d_loss = (real_loss + fake_loss) / 2

        if train:
            d_loss.backward()
            optim_D.step()


        ret['loss_G'] = g_loss
        ret['loss_D'] = d_loss
        return ret

    def test_performance(self, data_loader):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)
            out = self.forward(batch, None, None, train=False)
            g_loss = out['loss_G']
            d_loss = out['loss_D']
            loss.update(g_loss, d_loss)

        return [
            Metric('LOSS', g_loss+d_loss, higher_better=False),
        ]


    def get_generator(self):
        """Compatibil
        ity function for multi-tasking architectures."""
        return self.G

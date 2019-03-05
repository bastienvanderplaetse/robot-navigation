# -*- coding: utf-8 -*-
import logging

import torch
from ..layers import Generator_no
from ..layers import Discriminator_no
import numpy as np
from torch.autograd import Variable
from ..datasets import MultimodalDataset
import torch.autograd as autograd
from torch import nn
from ..utils.ml_metrics import Loss
from ..utils.device import DEVICE
from ..utils.misc import pbar
from ..utils.data import onehot_batch_data
from ..metrics import Metric

from .nmt import NMT
import math
logger = logging.getLogger('nmtpytorch')


class WGAN_NO(NMT):

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

        self.emb_G = nn.Embedding(self.n_trg_vocab, self.opts.model['emb_dim'],
                                padding_idx=0, max_norm=self.opts.model['emb_maxnorm'],
                                scale_grad_by_freq=self.opts.model['emb_gradscale'])
        self.emb_D = nn.Embedding(self.n_trg_vocab, self.opts.model['emb_dim'],
                                        padding_idx=0, max_norm=self.opts.model['emb_maxnorm'],
                                        scale_grad_by_freq=self.opts.model['emb_gradscale'])

        # Create Decoder
        self.G = Generator_no(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            emb=self.emb_G,
            ctx_size_dict=self.ctx_sizes,
            ctx_name='feats',
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            dropout=self.opts.model['dropout'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
        )

        self.D = Discriminator_no(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            emb=self.emb_D,
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

    def compute_gradient_penalty(self, D, feature, real_samples, fake_samples):
        Tensor = torch.cuda.FloatTensor
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((1, real_samples.size(1), 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(feature, interpolates, one_hot=False)
        fake = Variable(Tensor(real_samples.size(1), 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    def forward(self, batch, optim_G, optim_D, **kwargs):
        ret = {}
        iteration = kwargs["uctr"]
        epoch = kwargs["ectr"]

        y = batch[self.tl]
        sentence = y[1:-1]
        bos = y[:1]
        eos = y[-1:]

        #curriculum learning
        seq_len = sentence.size(0)
        min_len = min(seq_len, math.ceil(epoch/10))

        index = torch.LongTensor(1).random_(0, (seq_len-min_len)+1)
        sample_sentence = y[index:index+min_len]
        sample_sentence = sentence
        # d_sentence = torch.cat((bos, sample_sentence, eos), dim=0)

        feature = self.encode(batch)
        g_loss = 0
        d_loss = 0

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optim_D.zero_grad()


        gen_s = self.G(feature, sample_sentence)

        real = self.D(feature, sample_sentence)
        fake = self.D(feature, gen_s, one_hot=False)

        gradient_penalty = self.compute_gradient_penalty(self.D, feature, onehot_batch_data(sample_sentence, self.n_trg_vocab), gen_s)

        d_loss = -torch.mean(real) + torch.mean(fake) + 10 * gradient_penalty


        d_loss.backward()
        optim_D.step()

        optim_G .zero_grad()


        if iteration % 5 == 0:
            # -----------------
            #  Train Generator
            # -----------------

            gen_s = self.G(feature, sample_sentence)
            fake  = self.D(feature, gen_s, one_hot=False)
            g_loss = -torch.mean(fake)
            g_loss.backward()
            optim_G.step()


        ret['loss_G'] = g_loss
        ret['loss_D'] = d_loss
        return ret

    def get_val_loss(self, batch):
        ret = {}
        feature = self.encode(batch)

        gen_s = self.G(feature, batch[self.tl])
        fake =  self.D(feature, gen_s, one_hot=False)
        g_loss = -torch.mean(fake)

        fake = self.D(feature, gen_s, one_hot=False)
        d_loss = -torch.mean(gen_s) + torch.mean(fake)

        ret['loss_G'] = g_loss
        ret['loss_D'] = d_loss
        return ret

    def test_performance(self, data_loader):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)
            out = self.get_val_loss(batch)
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

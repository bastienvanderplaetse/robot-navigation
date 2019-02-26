from collections import defaultdict

import numpy as np
import torch

from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score as lrap

from .device import DEVICE


class Loss:
    """Accumulates and computes correctly training and validation losses."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._loss_D = 0
        self._loss_G = 0
        self.batch_loss_D = 0
        self.batch_loss_G = 0
        self.n_batch = 0

    def update(self, loss_G, loss_D):
        # Store last batch loss
        self.batch_loss_D = loss_D.item()
        self.batch_loss_G = loss_G.item()
        # Add it to cumulative loss
        self._loss_D += self.batch_loss_D
        self._loss_G += self.batch_loss_G
        self.n_batch +=1



    def get(self):
        return self._loss_G/self.n_batch, self._loss_D/self.n_batch


class CoverageError:
    def __init__(self):
        self._cov = 0
        self._n_items = 0

    def update(self, y_true, y_pred):
        self._cov += coverage_error(y_true, y_pred) * y_pred.shape[0]
        self._n_items += y_pred.shape[0]

    def get(self):
        return self._cov / self._n_items


class LRAPScore:
    def __init__(self):
        self._lrap = 0
        self._n_items = 0

    def update(self, y_true, y_pred):
        self._lrap += lrap(y_true, y_pred) * y_pred.shape[0]
        self._n_items += y_pred.shape[0]

    def get(self):
        return self._lrap / self._n_items


class MeanReciprocalRank:
    """Computes the mean reciprocal rank (MRR) metric for a batch along with
    per time-step MRR statistics that accumulate."""
    def __init__(self, n_classes):
        self.denom = torch.arange(1, 1 + n_classes, device=DEVICE, dtype=torch.float)
        self._mrr_per_timestep = defaultdict(float)
        self._per_timestep_counts = defaultdict(int)

    def update(self, y_true, y_pred):
        # y_pred: tstep x bsize x n_classes
        # y_true: tstep x bsize

        # Get a clone to mask out zero padded elements
        y_true_nz = y_true.clone()
        y_true_nz[y_true_nz.eq(0)] = -1
        y_true_nz.unsqueeze_(-1)

        # Sort negative log-probabilities from most-likely to less-likely
        sorted_logp, sorted_idxs = torch.sort(y_pred, dim=-1, descending=True)

        # matches: tstep x bsize x vocab of binary indicators to mark matches
        matches = (sorted_idxs == y_true_nz).float()

        samples_per_timestep = (y_true > 1).sum(1).tolist()

        # Compute MRR per timestep
        for tstep, n_samples in enumerate(samples_per_timestep):
            self._mrr_per_timestep[tstep + 1] += (
                matches[tstep].sum(0) / self.denom).sum()
            self._per_timestep_counts[tstep + 1] += n_samples

    def normalized_mrr(self):
        x, y = self.per_timestep_mrr()
        return 100. * (x.sum() / y.sum())

    def per_timestep_mrr(self):
        timesteps = list(range(1, 1 + len(self._per_timestep_counts)))
        counts = np.array([self._per_timestep_counts[t] for t in timesteps])
        scores = np.array([self._mrr_per_timestep[t] for t in timesteps])
        return scores, counts

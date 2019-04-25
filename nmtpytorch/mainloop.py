# -*- coding: utf-8 -*-
import time
import logging
import os
import json
import sys

import torch

from .evaluator import Evaluator
from .optimizer import Optimizer
from .monitor import Monitor
from .utils.misc import get_module_groups
from .utils.misc import load_pt_file, fix_seed
from .utils.ml_metrics import Loss
from .utils.data import make_dataloader
from .utils.tensorboard import TensorBoard
from .search import beam_search

from os import listdir
from os.path import exists, isfile, join

logger = logging.getLogger('nmtpytorch')


class MainLoop:
    def __init__(self, model, train_opts, dev_mgr):
        torch.cuda.init()
        # Get all training options into this mainloop
        self.__dict__.update(train_opts)

        self.print = logger.info
        self.model = model
        self.dev_mgr = dev_mgr
        self.epoch_valid = (self.eval_freq == 0)
        self.oom_count = 0
        self.loss_meter = Loss()
        self._found_optim_state = None

        # Log scores
        self.log_score = train_opts['log_score']
        self.log_dir_name = "scoresevaluation"
        if not os.path.exists(self.log_dir_name) or not os.path.isdir(self.log_dir_name):
            os.mkdir(self.log_dir_name)
        self.criteria = train_opts[train_opts['criteria']]
        filename = "{0}_{1}.json".format(train_opts['log_score_file'], self.criteria)
        print(filename)
        self.log_score_file = join(self.log_dir_name, filename)

        # Load training and validation data & create iterators
        self.print('Loading dataset(s)')
        self.train_iterator = make_dataloader(
            self.model.load_data('train', self.batch_size),
            self.pin_memory, self.num_workers)

        # Create monitor for validation, evaluation, checkpointing stuff
        self.monitor = Monitor(self.save_path +"/"+ self.subfolder, self.exp_id,
                               self.model, logger, self.patience,
                               self.eval_metrics,
                               save_best_metrics=self.save_best_metrics,
                               n_checkpoints=self.n_checkpoints)

        # If a validation set exists
        if 'val_set' in self.model.opts.data and self.eval_freq >= 0:
            if 'LOSS' in self.monitor.eval_metrics:
                self.vloss_iterator = make_dataloader(
                    self.model.load_data('val', self.batch_size, mode='eval'))

            if self.monitor.beam_metrics is not None:
                self.beam_iterator = make_dataloader(
                    self.model.load_data('val', self.eval_batch_size, mode='beam'))
                # Create hypothesis evaluator
                self.evaluator = Evaluator(
                    self.model.val_refs, self.monitor.beam_metrics,
                    filters=self.eval_filters)

        # Setup model
        self.model.setup()
        self.model.reset_parameters()

        ################################################
        # Initialize model weights with a pretrained one
        # This should come after model.setup()
        ################################################
        if train_opts['pretrained_file']:
            # Relax the strict condition for partial initialization
            data = load_pt_file(train_opts['pretrained_file'])
            weights = data['model']
            self._found_optim_state = data.get('optimizer', None)

            for name in get_module_groups(weights.keys()):
                self.print(
                    ' -> will initialize {}.* with pretrained weights.'.format(name))
            model.load_state_dict(weights, strict=False)

        ############################
        # Freeze layers if requested
        ############################
        if train_opts['freeze_layers']:
            frozen = []
            for layer in train_opts['freeze_layers'].split(','):
                for name, param in self.model.named_parameters():
                    if name.startswith(layer):
                        param.requires_grad = False
                        frozen.append(name)

            for name in get_module_groups(frozen):
                self.print(' -> froze parameter {}.*'.format(name))

        self.print(self.model)
        self.model = self.model.to(self.dev_mgr.dev)

        if self.dev_mgr.req_cpu or len(self.dev_mgr.cuda_dev_ids) == 1:
            self.net = self.model.cuda()
        else:
            self.net = torch.nn.DataParallel(
                self.model, device_ids=self.dev_mgr.cuda_dev_ids, dim=1)

        # Create optimizer instance
        self.optim_G = Optimizer(
            self.optimizer, self.model.G, lr=self.lr, momentum=self.momentum,
            nesterov=self.nesterov, weight_decay=self.l2_reg,
            gclip=self.gclip, lr_decay=self.lr_decay,
            lr_decay_factor=self.lr_decay_factor,
            lr_decay_mode=self.monitor.lr_decay_mode,
            lr_decay_min=self.lr_decay_min,
            lr_decay_patience=self.lr_decay_patience)

        self.optim_D = Optimizer(
            self.optimizer, self.model.D, lr=self.lr, momentum=self.momentum,
            nesterov=self.nesterov, weight_decay=self.l2_reg,
            gclip=self.gclip, lr_decay=self.lr_decay,
            lr_decay_factor=self.lr_decay_factor,
            lr_decay_mode=self.monitor.lr_decay_mode,
            lr_decay_min=self.lr_decay_min,
            lr_decay_patience=self.lr_decay_patience)

        self.print(self.optim_G)
        self.print(self.optim_D)

        if self._found_optim_state:
            # NOTE: This will overwrite weight_decay and lr parameters
            # from the checkpoint without obeying to new config file!
            self.optim_G.load_state_dict(self._found_optim_state)


        # Shift-by-1 and reseed to reproduce batch orders independently
        # from model initialization etc.
        fix_seed(self.seed + 1)

    def train_batch(self, batch):
        """Trains a batch."""
        nn_start = time.time()
        start = nn_start
        # self.print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        # self.print(batch)
        # self.print(batch['label'])
        # self.print(batch['en'])
        # self.print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        # Forward pass with training progress
        out = self.net(batch, self.optim_G, self.optim_D, uctr=self.monitor.uctr, ectr=self.monitor.ectr)
        end = time.time()
        # print("FORWARD : {0}".format(end-start))
        g_loss = out['loss_G']
        d_loss = out['loss_D']
        start = time.time()
        if self.monitor.uctr % 10 == 0:
            self.loss_meter.update(g_loss, d_loss)
            end = time.time()
            # print("UPDATE : {0}".format(end-start))


        return time.time() - nn_start

    def train_epoch(self):
        """Trains a full epoch."""
        self.print('Starting Epoch {}'.format(self.monitor.ectr))
        start = time.time()
        nn_sec = 0.0
        eval_sec = 0.0
        total_sec = time.time()
        self.loss_meter.reset()
        self.oom_count = 0

        for batch in self.train_iterator:
            print("BAAAATCH")
            print(batch['feats'])
            print("END BAAAATCH")
            batch.device(self.dev_mgr.dev)

            try:
                nn_sec += self.train_batch(batch)
            except RuntimeError as e:
                if self.handle_oom and 'out of memory' in e.args[0]:
                    torch.cuda.empty_cache()
                    self.oom_count += 1
                else:
                    raise e

            self.monitor.uctr += 1
            # if self.monitor.uctr % self.disp_freq == 0:
            #     # Send statistics
            #     msg = "Epoch {} - update {:10d} => loss G: {:>7.3f}, loss D: {:>7.3f}".format(
            #         self.monitor.ectr, self.monitor.uctr,
            #         self.loss_meter.batch_loss_G, self.loss_meter.batch_loss_D)
            #     for key, value in self.net.aux_loss.items():
            #         val = value.item()
            #         msg += ' [{}: {:.3f}]'.format(key, val)
            #     msg += ' (#OOM: {})'.format(self.oom_count)
            #     self.print(msg)

            # Do validation?
            if (not self.epoch_valid and
                    self.monitor.ectr >= self.eval_start and
                    self.eval_freq > 0 and
                    self.monitor.uctr % self.eval_freq == 0):
                eval_start = time.time()
                self.do_validation()
                eval_sec += time.time() - eval_start

            if (self.checkpoint_freq and self.n_checkpoints > 0 and
                    self.monitor.uctr % self.checkpoint_freq == 0):
                self.print('Saving checkpoint...')
                self.monitor.save_checkpoint()

            # Check stopping conditions
            if self.monitor.early_bad == self.monitor.patience:
                self.print("Early stopped.")
                return False

            if self.monitor.uctr == self.max_iterations:
                self.print("Max iterations {} reached.".format(
                    self.max_iterations))
                return False

        # All time spent for this epoch
        total_min = (time.time() - total_sec) / 60
        # All time spent during forward/backward/step
        nn_min = nn_sec / 60
        # All time spent during validation(s)
        eval_min = eval_sec / 60
        # Rest is iteration overhead + checkpoint saving
        overhead_min = total_min - nn_min - eval_min

        # Compute epoch loss
        epoch_loss_G, epoch_loss_D = self.loss_meter.get()
        self.monitor.train_loss_G.append(epoch_loss_G)
        self.monitor.train_loss_D.append(epoch_loss_D)

        self.print("--> Epoch {} finished with mean G loss {:.5f} and mean D loss {:.5f}".format(
            self.monitor.ectr, epoch_loss_G, epoch_loss_D))
        self.print("--> Overhead/Training/Evaluation: {:.2f}/{:.2f}/{:.2f} "
                   "mins (total: {:.2f} mins)   ({} samples/sec)".format(
                       overhead_min, nn_min, eval_min, total_min,
                       int(len(self.train_iterator.dataset) / nn_sec)))

        # Do validation?
        if self.epoch_valid and self.monitor.ectr >= self.eval_start:
            self.do_validation()

        # Check whether maximum epoch is reached
        if self.monitor.ectr == self.max_epochs:
            self.print("Max epochs {} reached.".format(self.max_epochs))
            return False

        end = time.time()
        print("End epoch Time : {0}".format(end-start))

        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

        self.monitor.ectr += 1
        return True

    def do_validation(self):
        """Do early-stopping validation."""
        results = []
        self.monitor.vctr += 1
        self.net.train(False)
        torch.set_grad_enabled(False)

        # Collect simple validation stats first
        self.print('Computing evaluation loss...')
        results.extend(self.net.test_performance(self.vloss_iterator))

        if self.monitor.beam_metrics:
            self.print('Performing beam search (beam_size:{})'.format(
                self.eval_beam))
            beam_time = time.time()
            # For multitask learning models, language-specific validation uses
            # by default the 0th Topology in val_tasks
            task = None
            if hasattr(self.net, 'val_tasks'):
                task = self.net.val_tasks[0].direction
            hyps = beam_search([self.net], self.beam_iterator,
                               task_id=task,
                               beam_size=self.eval_beam,
                               max_len=self.eval_max_len)
            print("HYPS---------------- {0}".format(hyps))
            # sys.exit(0)
            beam_time = time.time() - beam_time

            # Compute metrics and update results
            score_time = time.time()
            results.extend(self.evaluator.score(hyps))
            # results contains Loss and Bleu scores
            score_time = time.time() - score_time

            self.print("Beam Search: {:.2f} sec, Scoring: {:.2f} sec "
                       "({} sent/sec)".format(beam_time, score_time,
                                              int(len(hyps) / beam_time)))

        # Add new scores to history
        self.monitor.update_scores(results)

        # Do a scheduler LR step
        self.optim_G.lr_step(self.monitor.get_last_eval_score())
        self.optim_D.lr_step(self.monitor.get_last_eval_score())

        # Check early-stop criteria and save snapshots if any
        self.monitor.save_models()

        # Dump summary and switch back to training mode
        self.monitor.val_summary()
        self.net.train(True)
        torch.set_grad_enabled(True)

    def load_json_scores(self):
        if exists(self.log_score_file) and isfile(self.log_score_file):
            with open(self.log_score_file, 'r') as f:
                return json.load(f)
        else:
            return dict()

    def save_json_scores(self, scores):
        with open(self.log_score_file, 'w') as fp:
            json.dump(scores, fp)

    def score_evaluation(self, scores):
        d = self.load_json_scores()
        d[self.criteria] = [score.score for score in scores]
        self.save_json_scores(d)

    def __call__(self):
        """Runs training loop."""
        self.print('Training started on %s' % time.strftime('%d-%m-%Y %H:%M:%S'))
        self.net.train(True)
        torch.set_grad_enabled(True)

        # Evaluate once before even starting training
        if self.eval_zero:
            self.do_validation()

        while self.train_epoch():
            pass

        if self.monitor.vctr > 0:
            self.monitor.val_summary()
        else:
            # No validation done, save final model
            self.print('Saving final model.')
            self.monitor.save_model(suffix='final')

        print("==============================")
        print(self.monitor.val_scores['BLEU'])
        print("==============================")

        if self.log_score:
            self.score_evaluation(self.monitor.val_scores['BLEU'])

        self.print('Training finished on %s' % time.strftime('%d-%m-%Y %H:%M'))

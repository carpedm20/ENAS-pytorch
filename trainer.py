"""The module for training ENAS."""
import contextlib
import glob
import math
import os

import numpy as np
import scipy.signal
from tensorboard import TensorBoard
import torch
from torch import nn
import torch.nn.parallel
from torch.autograd import Variable

import models
import utils


logger = utils.get_logger()


def _apply_penalties(extra_out, args):
    """Based on `args`, optionally adds regularization penalty terms for
    activation regularization, temporal activation regularization and/or hidden
    state norm stabilization.

    Args:
        extra_out[*]:
            dropped: Post-dropout activations.
            hiddens: All hidden states for a batch of sequences.
            raw: Pre-dropout activations.

    Returns:
        The penalty term associated with all of the enabled regularizations.

    See:
        Regularizing and Optimizing LSTM Language Models (Merity et al., 2017)
        Regularizing RNNs by Stabilizing Activations (Krueger & Memsevic, 2016)
    """
    penalty = 0

    # Activation regularization.
    if args.activation_regularization:
        penalty += (args.activation_regularization_amount *
                    extra_out['dropped'].pow(2).mean())

    # Temporal activation regularization (slowness)
    if args.temporal_activation_regularization:
        raw = extra_out['raw']
        penalty += (args.temporal_activation_regularization_amount *
                    (raw[1:] - raw[:-1]).pow(2).mean())

    # Norm stabilizer regularization
    if args.norm_stabilizer_regularization:
        penalty += (args.norm_stabilizer_regularization_amount *
                    (extra_out['hiddens'].norm(dim=-1) -
                     args.norm_stabilizer_fixed_point).pow(2).mean())

    return penalty


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


def _get_no_grad_ctx_mgr():
    """Returns a the `torch.no_grad` context manager for PyTorch version >=
    0.4, or a no-op context manager otherwise.
    """
    if float(torch.__version__[0:3]) >= 0.4:
        return torch.no_grad()

    return contextlib.suppress()


def _check_abs_max_grad(abs_max_grad, model):
    """Checks `model` for a new largest gradient for this epoch, in order to
    track gradient explosions.
    """
    finite_grads = [p.grad.data
                    for p in model.parameters()
                    if p.grad is not None]

    new_max_grad = max([grad.max() for grad in finite_grads])
    new_min_grad = min([grad.min() for grad in finite_grads])

    new_abs_max_grad = max(new_max_grad, abs(new_min_grad))
    if new_abs_max_grad > abs_max_grad:
        logger.info(f'abs max grad {abs_max_grad}')
        return new_abs_max_grad

    return abs_max_grad


class Trainer(object):
    """A class to wrap training code."""
    def __init__(self, args, dataset):
        """Constructor for training algorithm.

        Args:
            args: From command line, picked up by `argparse`.
            dataset: Currently only `data.text.Corpus` is supported.

        Initializes:
            - Data: train, val and test.
            - Model: shared and controller.
            - Inference: optimizers for shared and controller parameters.
            - Criticism: cross-entropy loss for training the shared model.
        """
        self.args = args
        self.controller_step = 0
        self.cuda = args.cuda
        self.dataset = dataset
        self.epoch = 0
        self.shared_step = 0
        self.start_epoch = 0

        logger.info('regularizing:')
        for regularizer in [('activation regularization',
                             self.args.activation_regularization),
                            ('temporal activation regularization',
                             self.args.temporal_activation_regularization),
                            ('norm stabilizer regularization',
                             self.args.norm_stabilizer_regularization)]:
            if regularizer[1]:
                logger.info(f'{regularizer[0]}')

        self.train_data = utils.batchify(dataset.train,
                                         args.batch_size,
                                         self.cuda)
        # NOTE(brendan): The validation set data is batchified twice
        # separately: once for computing rewards during the Train Controller
        # phase (valid_data, batch size == 64), and once for evaluating ppl
        # over the entire validation set (eval_data, batch size == 1)
        self.valid_data = utils.batchify(dataset.valid,
                                         args.batch_size,
                                         self.cuda)
        self.eval_data = utils.batchify(dataset.valid,
                                        args.test_batch_size,
                                        self.cuda)
        self.test_data = utils.batchify(dataset.test,
                                        args.test_batch_size,
                                        self.cuda)

        self.max_length = self.args.shared_rnn_max_length

        if args.use_tensorboard:
            self.tb = TensorBoard(args.model_dir)
        else:
            self.tb = None
        self.build_model()

        if self.args.load_path:
            self.load_model()

        shared_optimizer = _get_optimizer(self.args.shared_optim)
        controller_optimizer = _get_optimizer(self.args.controller_optim)

        self.shared_optim = shared_optimizer(
            self.shared.parameters(),
            lr=self.shared_lr,
            weight_decay=self.args.shared_l2_reg)

        self.controller_optim = controller_optimizer(
            self.controller.parameters(),
            lr=self.args.controller_lr)

        self.ce = nn.CrossEntropyLoss()

    def build_model(self):
        """Creates and initializes the shared and controller models."""
        if self.args.network_type == 'rnn':
            self.shared = models.RNN(self.args, self.dataset)
        elif self.args.network_type == 'cnn':
            self.shared = models.CNN(self.args, self.dataset)
        else:
            raise NotImplementedError(f'Network type '
                                      f'`{self.args.network_type}` is not '
                                      f'defined')
        self.controller = models.Controller(self.args)

        if self.args.num_gpu == 1:
            self.shared.cuda()
            self.controller.cuda()
        elif self.args.num_gpu > 1:
            raise NotImplementedError('`num_gpu > 1` is in progress')

    def train(self, single=False):
        """Cycles through alternately training the shared parameters and the
        controller, as described in Section 2.2, Training ENAS and Deriving
        Architectures, of the paper.

        From the paper (for Penn Treebank):

        - In the first phase, shared parameters omega are trained for 400
          steps, each on a minibatch of 64 examples.

        - In the second phase, the controller's parameters are trained for 2000
          steps.
          
        Args:
            single (bool): If True it won't train the controller and use the
                           same dag instead of derive().
        """
        dag = utils.load_dag(self.args) if single else None
        
        if self.args.shared_initial_step > 0:
            self.train_shared(self.args.shared_initial_step)
            self.train_controller()

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # 1. Training the shared parameters omega of the child models
            self.train_shared(dag=dag)

            # 2. Training the controller parameters theta
            if not single:
                self.train_controller()

            if self.epoch % self.args.save_epoch == 0:
                with _get_no_grad_ctx_mgr():
                    best_dag = dag if dag else self.derive()
                    self.evaluate(self.eval_data,
                                  best_dag,
                                  'val_best',
                                  max_num=self.args.batch_size*100)
                self.save_model()

            if self.epoch >= self.args.shared_decay_after:
                utils.update_lr(self.shared_optim, self.shared_lr)

    def get_loss(self, inputs, targets, hidden, dags):
        """Computes the loss for the same batch for M models.

        This amounts to an estimate of the loss, which is turned into an
        estimate for the gradients of the shared model.
        """
        if not isinstance(dags, list):
            dags = [dags]

        loss = 0
        for dag in dags:
            output, hidden, extra_out = self.shared(inputs, dag, hidden=hidden)
            output_flat = output.view(-1, self.dataset.num_tokens)
            sample_loss = (self.ce(output_flat, targets) /
                           self.args.shared_num_sample)
            loss += sample_loss

        assert len(dags) == 1, 'there are multiple `hidden` for multple `dags`'
        return loss, hidden, extra_out

    def train_shared(self, max_step=None, dag=None):
        """Train the language model for 400 steps of minibatches of 64
        examples.

        Args:
            max_step: Used to run extra training steps as a warm-up.
            dag: If not None, is used instead of calling sample().

        BPTT is truncated at 35 timesteps.

        For each weight update, gradients are estimated by sampling M models
        from the fixed controller policy, and averaging their gradients
        computed on a batch of training data.
        """
        model = self.shared
        model.train()
        self.controller.eval()

        hidden = self.shared.init_hidden(self.args.batch_size)

        if max_step is None:
            max_step = self.args.shared_max_step
        else:
            max_step = min(self.args.shared_max_step, max_step)

        abs_max_grad = 0
        abs_max_hidden_norm = 0
        step = 0
        raw_total_loss = 0
        total_loss = 0
        train_idx = 0
        # TODO(brendan): Why - 1 - 1?
        while train_idx < self.train_data.size(0) - 1 - 1:
            if step > max_step:
                break

            dags = dag if dag else self.controller.sample(
                self.args.shared_num_sample)
            inputs, targets = self.get_batch(self.train_data,
                                             train_idx,
                                             self.max_length)

            loss, hidden, extra_out = self.get_loss(inputs,
                                                    targets,
                                                    hidden,
                                                    dags)
            hidden.detach_()
            raw_total_loss += loss.data

            loss += _apply_penalties(extra_out, self.args)

            # update
            self.shared_optim.zero_grad()
            loss.backward()

            h1tohT = extra_out['hiddens']
            new_abs_max_hidden_norm = utils.to_item(
                h1tohT.norm(dim=-1).data.max())
            if new_abs_max_hidden_norm > abs_max_hidden_norm:
                abs_max_hidden_norm = new_abs_max_hidden_norm
                logger.info(f'max hidden {abs_max_hidden_norm}')
            abs_max_grad = _check_abs_max_grad(abs_max_grad, model)
            torch.nn.utils.clip_grad_norm(model.parameters(),
                                          self.args.shared_grad_clip)
            self.shared_optim.step()

            total_loss += loss.data

            if ((step % self.args.log_step) == 0) and (step > 0):
                self._summarize_shared_train(total_loss, raw_total_loss)
                raw_total_loss = 0
                total_loss = 0

            step += 1
            self.shared_step += 1
            train_idx += self.max_length

    def get_reward(self, dag, entropies, hidden, valid_idx=0):
        """Computes the perplexity of a single sampled model on a minibatch of
        validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()

        inputs, targets = self.get_batch(self.valid_data,
                                         valid_idx,
                                         self.max_length,
                                         volatile=True)
        valid_loss, hidden, _ = self.get_loss(inputs, targets, hidden, dag)
        valid_loss = utils.to_item(valid_loss.data)

        valid_ppl = math.exp(valid_loss)

        # TODO: we don't know reward_c
        if self.args.ppl_square:
            # TODO: but we do know reward_c=80 in the previous paper
            R = self.args.reward_c / valid_ppl ** 2
        else:
            R = self.args.reward_c / valid_ppl

        if self.args.entropy_mode == 'reward':
            rewards = R + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = R * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards, hidden

    def train_controller(self):
        """Fixes the shared parameters and updates the controller parameters.

        The controller is updated with a score function gradient estimator
        (i.e., REINFORCE), with the reward being c/valid_ppl, where valid_ppl
        is computed on a minibatch of validation data.

        A moving average baseline is used.

        The controller is trained for 2000 steps per epoch (i.e.,
        first (Train Shared) phase -> second (Train Controller) phase).
        """
        model = self.controller
        model.train()
        # TODO(brendan): Why can't we call shared.eval() here? Leads to loss
        # being uniformly zero for the controller.
        # self.shared.eval()

        avg_reward_base = None
        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        hidden = self.shared.init_hidden(self.args.batch_size)
        total_loss = 0
        valid_idx = 0
        for step in range(self.args.controller_max_step):
            # sample models
            dags, log_probs, entropies = self.controller.sample(
                with_details=True)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            # NOTE(brendan): No gradients should be backpropagated to the
            # shared model during controller training, obviously.
            with _get_no_grad_ctx_mgr():
                rewards, hidden = self.get_reward(dags,
                                                  np_entropies,
                                                  hidden,
                                                  valid_idx)

            # discount
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            adv_history.extend(adv)

            # policy loss
            loss = -log_probs*utils.get_variable(adv,
                                                 self.cuda,
                                                 requires_grad=False)
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()

            total_loss += utils.to_item(loss.data)

            if ((step % self.args.log_step) == 0) and (step > 0):
                self._summarize_controller_train(total_loss,
                                                 adv_history,
                                                 entropy_history,
                                                 reward_history,
                                                 avg_reward_base,
                                                 dags)

                reward_history, adv_history, entropy_history = [], [], []
                total_loss = 0

            self.controller_step += 1

            prev_valid_idx = valid_idx
            valid_idx = ((valid_idx + self.max_length) %
                         (self.valid_data.size(0) - 1))
            # NOTE(brendan): Whenever we wrap around to the beginning of the
            # validation data, we reset the hidden states.
            if prev_valid_idx > valid_idx:
                hidden = self.shared.init_hidden(self.args.batch_size)

    def evaluate(self, source, dag, name, batch_size=1, max_num=None):
        """Evaluate on the validation set.

        NOTE(brendan): We should not be using the test set to develop the
        algorithm (basic machine learning good practices).
        """
        self.shared.eval()
        self.controller.eval()

        data = source[:max_num*self.max_length]

        total_loss = 0
        hidden = self.shared.init_hidden(batch_size)

        pbar = range(0, data.size(0) - 1, self.max_length)
        for count, idx in enumerate(pbar):
            inputs, targets = self.get_batch(data, idx, volatile=True)
            output, hidden, _ = self.shared(inputs,
                                            dag,
                                            hidden=hidden,
                                            is_train=False)
            output_flat = output.view(-1, self.dataset.num_tokens)
            total_loss += len(inputs) * self.ce(output_flat, targets).data
            hidden.detach_()
            ppl = math.exp(utils.to_item(total_loss) / (count + 1) / self.max_length)

        val_loss = utils.to_item(total_loss) / len(data)
        ppl = math.exp(val_loss)

        self.tb.scalar_summary(f'eval/{name}_loss', val_loss, self.epoch)
        self.tb.scalar_summary(f'eval/{name}_ppl', ppl, self.epoch)
        logger.info(f'eval | loss: {val_loss:8.2f} | ppl: {ppl:8.2f}')

    def derive(self, sample_num=None, valid_idx=0):
        """TODO(brendan): We are always deriving based on the very first batch
        of validation data? This seems wrong...
        """
        hidden = self.shared.init_hidden(self.args.batch_size)

        if sample_num is None:
            sample_num = self.args.derive_num_sample

        dags, _, entropies = self.controller.sample(sample_num,
                                                    with_details=True)

        max_R = 0
        best_dag = None
        for dag in dags:
            R, _ = self.get_reward(dag, entropies, hidden, valid_idx)
            if R.max() > max_R:
                max_R = R.max()
                best_dag = dag

        logger.info(f'derive | max_R: {max_R:8.6f}')
        fname = (f'{self.epoch:03d}-{self.controller_step:06d}-'
                 f'{max_R:6.4f}-best.png')
        path = os.path.join(self.args.model_dir, 'networks', fname)
        utils.draw_network(best_dag, path)
        self.tb.image_summary('derive/best', [path], self.epoch)

        return best_dag

    @property
    def shared_lr(self):
        degree = max(self.epoch - self.args.shared_decay_after + 1, 0)
        return self.args.shared_lr * (self.args.shared_decay ** degree)

    @property
    def controller_lr(self):
        return self.args.controller_lr

    def get_batch(self, source, idx, length=None, volatile=False):
        # code from
        # https://github.com/pytorch/examples/blob/master/word_language_model/main.py
        length = min(length if length else self.max_length,
                     len(source) - 1 - idx)
        data = Variable(source[idx:idx + length], volatile=volatile)
        target = Variable(source[idx + 1:idx + 1 + length].view(-1),
                          volatile=volatile)
        return data, target

    @property
    def shared_path(self):
        return f'{self.args.model_dir}/shared_epoch{self.epoch}_step{self.shared_step}.pth'

    @property
    def controller_path(self):
        return f'{self.args.model_dir}/controller_epoch{self.epoch}_step{self.controller_step}.pth'

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join(self.args.model_dir, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                    name.split(delimiter)[idx].replace(replace_word, ''))
                    for name in basenames if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def save_model(self):
        torch.save(self.shared.state_dict(), self.shared_path)
        logger.info(f'[*] SAVED: {self.shared_path}')

        torch.save(self.controller.state_dict(), self.controller_path)
        logger.info(f'[*] SAVED: {self.controller_path}')

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join(self.args.model_dir, f'*_epoch{epoch}_*.pth'))

            for path in paths:
                utils.remove_file(path)

    def load_model(self):
        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        if len(epochs) == 0:
            logger.info(f'[!] No checkpoint found in {self.args.model_dir}...')
            return

        self.epoch = self.start_epoch = max(epochs)
        self.shared_step = max(shared_steps)
        self.controller_step = max(controller_steps)

        if self.args.num_gpu == 0:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        self.shared.load_state_dict(
            torch.load(self.shared_path, map_location=map_location))
        logger.info(f'[*] LOADED: {self.shared_path}')

        self.controller.load_state_dict(
            torch.load(self.controller_path, map_location=map_location))
        logger.info(f'[*] LOADED: {self.controller_path}')

    def _summarize_controller_train(self,
                                    total_loss,
                                    adv_history,
                                    entropy_history,
                                    reward_history,
                                    avg_reward_base,
                                    dags):
        """Logs the controller's progress for this training epoch."""
        cur_loss = total_loss / self.args.log_step

        avg_adv = np.mean(adv_history)
        avg_entropy = np.mean(entropy_history)
        avg_reward = np.mean(reward_history)

        if avg_reward_base is None:
            avg_reward_base = avg_reward

        logger.info(
            f'| epoch {self.epoch:3d} | lr {self.controller_lr:.5f} '
            f'| R {avg_reward:.5f} | entropy {avg_entropy:.4f} '
            f'| loss {cur_loss:.5f}')

        # Tensorboard
        if self.tb is not None:
            self.tb.scalar_summary('controller/loss',
                                   cur_loss,
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward',
                                   avg_reward,
                                   self.controller_step)
            self.tb.scalar_summary('controller/reward-B_per_epoch',
                                   avg_reward - avg_reward_base,
                                   self.controller_step)
            self.tb.scalar_summary('controller/entropy',
                                   avg_entropy,
                                   self.controller_step)
            self.tb.scalar_summary('controller/adv',
                                   avg_adv,
                                   self.controller_step)

            paths = []
            for dag in dags:
                fname = (f'{self.epoch:03d}-{self.controller_step:06d}-'
                         f'{avg_reward:6.4f}.png')
                path = os.path.join(self.args.model_dir, 'networks', fname)
                utils.draw_network(dag, path)
                paths.append(path)

            self.tb.image_summary('controller/sample',
                                  paths,
                                  self.controller_step)

    def _summarize_shared_train(self, total_loss, raw_total_loss):
        """Logs a set of training steps."""
        cur_loss = utils.to_item(total_loss) / self.args.log_step
        # NOTE(brendan): The raw loss, without adding in the activation
        # regularization terms, should be used to compute ppl.
        cur_raw_loss = utils.to_item(raw_total_loss) / self.args.log_step
        ppl = math.exp(cur_raw_loss)

        logger.info(f'| epoch {self.epoch:3d} '
                    f'| lr {self.shared_lr:4.2f} '
                    f'| raw loss {cur_raw_loss:.2f} '
                    f'| loss {cur_loss:.2f} '
                    f'| ppl {ppl:8.2f}')

        # Tensorboard
        if self.tb is not None:
            self.tb.scalar_summary('shared/loss',
                                   cur_loss,
                                   self.shared_step)
            self.tb.scalar_summary('shared/perplexity',
                                   ppl,
                                   self.shared_step)

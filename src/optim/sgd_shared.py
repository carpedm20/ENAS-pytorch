import typing
import torch
from torch.optim import Optimizer, Adam, SGD
from torch.optim.optimizer import required

from optim.shared_optimizer_base import SharedOptimizerBase


class SGDShared(SGD, SharedOptimizerBase):
    """
    SGD with functions which allow Optimizer variables to be shifted to and from a gpu device
    """
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, gpu_device=None):
        SGD.__init__(self, params, lr=lr, momentum=momentum, dampening=dampening,
                         weight_decay=weight_decay, nesterov=nesterov)
        SharedOptimizerBase.__init__(self, gpu_device=gpu_device)
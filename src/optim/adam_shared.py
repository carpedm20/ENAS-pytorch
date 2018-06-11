import typing
import torch
from torch.optim import Optimizer, Adam
from optim.shared_optimizer_base import SharedOptimizerBase


class AdamShared(Adam, SharedOptimizerBase):
    """
    SharedAdamOptimizer
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, gpu_device=None):
        Adam.__init__(self, params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        SharedOptimizerBase.__init__(self, gpu_device=gpu_device)
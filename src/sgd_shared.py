import typing
import torch
from torch.optim import Optimizer, Adam, SGD
from torch.optim.optimizer import required


class SGDShared(SGD):
    """
    SharedAdamOptimizer
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, gpu=None):
        super().__init__(params, lr=lr, momentum=momentum, dampening=dampening,
                         weight_decay=weight_decay, nesterov=nesterov)
        self.gpu_device = gpu
        self.gpu_params = set()
        self.cpu_device = torch.device("cpu")

    def __to_device(self, device, params):
        param_groups = list(params)
        if len(param_groups) == 0:
            return
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            for p in param_group['params']:
                state = self.state[p]
                for key in state:
                    if not isinstance(state[key], int):
                        state[key] = state[key].to(device)
                # self.add_param_group(param_group)

    def to_gpu(self, params: typing.Iterable):
        params = set(params)
        if self.gpu_device is None:
            raise Exception("No GPU given")
        else:
            params_to_gpu = params - self.gpu_params
            params_to_cpu = self.gpu_params - params

            self.gpu_params = params
            self.__to_device(self.gpu_device, params_to_gpu)
            self.__to_device(self.cpu_device, params_to_cpu)

    def to_cpu(self):
        self.to_gpu(set())

    def full_reset_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad = None

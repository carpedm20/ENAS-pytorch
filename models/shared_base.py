import numpy as np
import torch


def size(p):
    return np.prod(p.size())

class SharedModel(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)

    @property
    def num_parameters(self):
        return sum([size(param) for param in self.parameters()])

    def get_f(self, name):
        raise NotImplementedError()

    def get_num_cell_parameters(self, dag):
        raise NotImplementedError()

    def reset_parameters(self):
        raise NotImplementedError()

import numpy as np
from torch import nn


def size(p):
    return np.prod(p.size())

class SharedModel(nn.Module):
    def __init__(self):
        super(SharedModel, self).__init__()

    @property
    def num_parameters(self):
        return sum([size(param) for param in self.parameters()])

    def get_f(self, name):
        raise NotImplemented()

    def get_num_cell_parameters(self, dag):
        raise NotImplemented()

    def reset_parameters(self):
        raise NotImplemented()

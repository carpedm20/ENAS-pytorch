import numpy as np
from collections import defaultdict, deque

import torch as t
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.shared_base import *
from utils import get_logger, get_variable, keydefaultdict

logger = get_logger()


def conv3x3(in_planes, out_planes, stride=1):
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    return conv_layer(layer, out_planes)

def conv5x5(in_planes, out_planes, stride=1):
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=1, bias=False)
    return conv_layer(layer, out_planes)

def avg3x3(in_planes, out_planes, stride=1):
    return nn.AvgPool2d(3, stride=stride)

def max3x3(in_planes, out_planes, stride=1):
    return nn.MaxPool2d(3, stride=stride)

def identity(in_planes, out_planes, stride=1):
    return nn.MaxPool2d(1, stride=1)

def conv_layer(conv, num_features):
    return nn.Sequential(
        nn.ReLU(), #nn.ReLU(inplace=True),
        conv,
        nn.BatchNorm2d(num_features=num_features),
    )

def conv(kernel, planes, reducing):
    if kernel == 3:
        _conv = conv3x3
    elif kernel == 5:
        _conv = conv5x5
    else:
        raise NotImplemented(f"Unkown kernel size: {kernel}")

    if reducing:
        stride = 2
    else:
        stride = 1

    return nn.Sequential(
            nn.ReLU(inplace=True),
            _conv(planes, planes, stride=stride),
            nn.BatchNorm2d(planes),
    )


class CNNCell(SharedModel):
    default_layer_types = [conv3x3, conv5x5]
    def __init__(self, args, input_channels, num_filters, reducing=False):
        super().__init__()

        self.args = args
        self.reset_parameters()

        self.inputs = [0] * args.num_blocks
        self.reducing = reducing
        self.num_filters = num_filters


        self.connections = defaultdict(lambda : defaultdict(dict))
        for idx in range(args.num_blocks):
            for jdx in range(idx+1, args.num_blocks+1):
                for _type in [conv3x3, conv5x5]:
                    if idx == 0:
                        if reducing:
                            stride = 2
                        else:
                            stride = 1
                        in_planes = input_channels
                    else:
                        stride = 1
                        in_planes = num_filters
                    out_planes = num_filters
                    # print((idx, jdx, _type), (in_planes, out_planes))
                    self.connections[idx][jdx][_type] = _type(in_planes=in_planes, out_planes=out_planes, stride=stride)
                    self.add_module(f'{idx}-{jdx}-{_type}', self.connections[idx][jdx][_type])


        # raise NotImplemented("In progress...")

    def forward(self, inputs, dag):
        self.inputs = [0] * (self.args.num_blocks + 1)
        # self.inputs = [t.zeros((inputs.shape[0], self.num_filters, inputs.shape[2], inputs.shape[3]), dtype=torch.float32)] * (self.args.num_blocks + 1)
        self.inputs[0] = inputs
        for source, target, type in dag:
            # print(source, target, type)
            # print( self.connections[source][target][type])
            # print(self.inputs[source].shape)

            self.inputs[target] += self.connections[source][target][type](self.inputs[source])
        return self.inputs[-1]

    def get_f(self, name):
        name = name.lower()


    def get_num_cell_parameters(self, dag):
        count = 0
        for source, target, type in dag:
            submodule = self.connections[source][target][type]
            model_parameters = filter(lambda p: p.requires_grad, submodule.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            count += params

        return count

    def reset_parameters(self):
        #TODO: Figure out if this should be implemented
        pass

    def to_cuda(self, device, dag):
        for source, target, type in dag:
            self.connections[source][target][type].to(device)


class CNN(SharedModel):
    def __init__(self, args, input_channels, height, width, output_classes,
                 architecture=[('normal', 768//4)]*6 + [('reducing', 768//2)] + [('normal', 768//2)]*6 + [('reducing', 768)] +  [('normal', 768)]*6):
        super().__init__()

        self.args = args

        self.height = height
        self.width = width
        self.output_classes = output_classes
        self.architecture = architecture

        self.output_height = self.height
        self.output_width = self.width

        self.cells = nn.Sequential()

        last_filters = input_channels
        for i, (type, num_filters) in enumerate(architecture):
            if type == 'normal':
                reducing = False
            elif type == 'reducing':
                reducing = True
            else:
                raise Exception("unexpected cell type")
            self.cells.add_module(f'{i}-{type}-{num_filters}', CNNCell(args, input_channels=last_filters, num_filters=num_filters, reducing=reducing))
            last_filters = num_filters

            if type == 'reducing':
                #TODO: do this calculation correctly
                self.output_height /= 2
                self.output_width /= 2


        if self.output_classes:
            self.conv_output_size = self.output_height * self.output_width * self.architecture[-1][-1]
            self.out_layer = nn.Linear(self.conv_output_size, self.output_classes)


    def forward(self, inputs, cell_dag, reducing_cell_dag):
        for cell in self.cells:
            if cell.reducing:
                dag = reducing_cell_dag
            else:
                dag = cell_dag
            # print(cell)
            inputs = cell(inputs, dag)

        x = inputs.view(-1, self.conv_output_size)
        x = self.out_layer(x)
        return x

    def get_f(self, name):
        name = name.lower()

    def get_num_cell_parameters(self, dag):
        count = 0
        for cell in self.cells:
            count += cell.get_num_parameters(dag)
        return count

    def reset_parameters(self):
        for cell in self.cells:
            cell.reset_parameters()

    def to_cuda(self, device, cell_dag, reducing_cell_dag):
        for cell in self.cells:
            if cell.reducing:
                cell.to_cuda(device, reducing_cell_dag)
            else:
                cell.to_cuda(device, cell_dag)

        self.out_layer.to(device)
import os
import pickle

import torch as t
import torch.nn.functional as F
from scipy.special import expit, logit
from torch import nn

from models.shared_base import *
from utils import get_logger

logger = get_logger()

def simple_depthwise_1x1(in_planes, out_planes, stride=1):
    if out_planes % in_planes != 0:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                  padding=0, groups=in_planes, bias=True)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                         padding=0, bias=True)

def fix_layers(in_planes, out_planes, stride, _funct):
    layer = nn.Sequential(_funct(in_planes, in_planes, stride), identity(in_planes, out_planes, stride=1))
    if hasattr(layer[0], 'input_relu') and layer[0].input_relu:
        layer.input_relu = layer.input_relu
    return layer

def conv_1x7_7x1(in_planes, out_planes, stride):
    layer = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=(1, 7), stride=(1, stride), padding=(0, 3), bias=True),
                nn.Conv2d(out_planes, out_planes, kernel_size=(7, 1), stride=(stride, 1), padding=(3, 0), bias=False)
    )
    return conv_layer(layer, out_planes)


def conv_1x3_3x1(in_planes, out_planes, stride):
    layer = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), bias=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), bias=False)
    )
    return conv_layer(layer, out_planes)


def dilated_3x3(in_planes, out_planes, stride):
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=2, dilation=2, bias=False)
    return conv_layer(layer, out_planes)

def depthwise_1x1(in_planes, out_planes, stride):
    if out_planes % in_planes != 0:
        return conv1x1(in_planes, out_planes, stride)
        # return fix_layers(in_planes, out_planes, stride, avg3x3)
    else:
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                          padding=0, groups=in_planes, bias=False)
        return conv_layer(layer, out_planes)

def depthwise_3x3(in_planes, out_planes, stride):
    if out_planes % in_planes != 0:
        # return conv3x3(in_planes, out_planes, stride)
        return fix_layers(in_planes, out_planes, stride, conv3x3)
    else:
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                          padding=1, groups=in_planes, bias=False)
        return conv_layer(layer, out_planes)

def depthwise_5x5(in_planes, out_planes, stride):
    if out_planes % in_planes != 0:
        # return conv5x5(in_planes, out_planes, stride)
        return fix_layers(in_planes, out_planes, stride, conv5x5)
    else:
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                          padding=2, groups=in_planes, bias=False)
        return conv_layer(layer, out_planes)

def depthwise_7x7(in_planes, out_planes, stride):
    if out_planes % in_planes != 0:
        # return conv5x5(in_planes, out_planes, stride)
        return fix_layers(in_planes, out_planes, stride, conv7x7)
    else:
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                          padding=3, groups=in_planes, bias=False)
        return conv_layer(layer, out_planes)

def conv1x1(in_planes, out_planes, stride=1):
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)
    return conv_layer(layer, out_planes)

def conv3x3(in_planes, out_planes, stride=1):
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    return conv_layer(layer, out_planes)

def conv5x5(in_planes, out_planes, stride=1):
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)
    return conv_layer(layer, out_planes)

def conv7x7(in_planes, out_planes, stride=1):
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)
    return conv_layer(layer, out_planes)

def avg3x3(in_planes, out_planes, stride=1):
    if in_planes != out_planes:
        return fix_layers(in_planes, out_planes, stride, avg3x3)
        # return nn.Sequential(depthwise_1x1(in_planes, out_planes, 1), avg3x3(out_planes, out_planes, stride=1))
    else:
        return nn.AvgPool2d(3, stride=stride, padding=1)

def max3x3(in_planes, out_planes, stride=1):
    if in_planes != out_planes:
        return fix_layers(in_planes, out_planes, stride, max3x3)
        # return depthwise_3x3(in_planes, out_planes, stride)
    else:
        return nn.MaxPool2d(3, stride=stride, padding=1)

def max5x5(in_planes, out_planes, stride=1):
    if in_planes != out_planes:
        return fix_layers(in_planes, out_planes, stride, max5x5)
        # return depthwise_5x5(in_planes, out_planes, stride)
    else:
        return nn.MaxPool2d(5, stride=stride, padding=2)

def max7x7(in_planes, out_planes, stride=1):
    if in_planes != out_planes:
        return fix_layers(in_planes, out_planes, stride, max7x7)
        # return depthwise_5x5(in_planes, out_planes, stride)
    else:
        return nn.MaxPool2d(5, stride=stride, padding=2)

class IdentityModule(nn.AvgPool2d):
    def __init__(self, stride, stack=1):
        super().__init__(kernel_size=1, stride=stride)
        self.stack = stack
        self.stride = stride

    def forward(self, input):
        if self.stack == 1:
            if self.stride == 1:
                return input
            else:
                return super().forward(input)
        else:
            return t.cat([super().forward(input)] * self.stack, dim=1)


def identity(in_planes, out_planes, stride=1):
    if out_planes % in_planes == 0:
        return IdentityModule(stride=stride, stack = out_planes//in_planes)
        # return simple_depthwise_1x1(in_planes, out_planes, stride)
    else:
        raise Exception("unsupported oporation")
        # return IdentityModule(stride=1)

def conv_layer(conv, num_features):
    layer = nn.Sequential(
        # nn.ReLU(),
        # nn.ReLU(inplace=False),
        conv,
        nn.BatchNorm2d(num_features=num_features, track_running_stats=False),
    )
    layer.input_relu= True
    return layer

# def conv(kernel, planes, reducing):
#     if kernel == 3:
#         _conv = conv3x3
#     elif kernel == 5:
#         _conv = conv5x5
#     else:
#         raise NotImplemented(f"Unkown kernel size: {kernel}")
#
#     if reducing:
#         stride = 2
#     else:
#         stride = 1
#
#     return nn.Sequential(
#             nn.ReLU(inplace=True),
#             _conv(planes, planes, stride=stride),
#             nn.BatchNorm2d(planes, track_running_stats=False),
#     )

def sigmoid_derivitive(x):
    return expit(x)*(1.0-expit(x))

class CNNCell(SharedModel):
    class InputInfo:
        def __init__(self, input_channels, input_width):
            self.input_channels = input_channels
            self.input_width = input_width
    # default_layer_types = [identity, conv_1x3_3x1, conv_1x7_7x1, dilated_3x3, avg3x3, max3x3, max5x5, max7x7, conv1x1, conv3x3, depthwise_3x3, depthwise_5x5, depthwise_7x7]

    default_layer_types = [identity, conv_1x3_3x1, conv_1x7_7x1, avg3x3, max3x3, conv1x1, conv3x3, depthwise_3x3, depthwise_5x5, depthwise_7x7]
    def __init__(self, args, input_1_info: InputInfo, input_2_info: InputInfo, output_channels, output_width, reducing, dag_vars):
        super().__init__()

        self.args = args
        self.reset_parameters()

        self.input_1_info = input_1_info
        self.input_2_info = input_2_info
        self.input_infos = (self.input_1_info, self.input_2_info)
        # self.inputs = [0] * (2 + args.num_blocks)
        # self.inputs = [0] * args.num_blocks
        num_outputs = 2 + args.num_blocks
        self.output_channels = output_channels
        self.output_width = output_width
        self.reducing = reducing

        self.dag_vars = dag_vars

        self.connections = dict()
        # self.connections = defaultdict(lambda : defaultdict(dict))
        for idx in range(num_outputs - 1):
            for jdx in range(max(idx+1, 2), num_outputs):
                for _type in CNNCell.default_layer_types:
                    if idx == 0 or idx == 1:
                        input_info = self.input_infos[idx]
                        if input_info.input_width != output_width:
                            assert(input_info.input_width/2 == output_width)
                            stride = 2
                        else:
                            stride = 1
                        in_planes = input_info.input_channels

                    else:
                        stride = 1
                        in_planes = output_channels

                    out_planes = output_channels
                    # print((idx, jdx, _type), (in_planes, out_planes))
                    self.connections[(idx, jdx, _type.__name__)] = _type(in_planes=in_planes, out_planes=out_planes, stride=stride)
                    # self.connections[idx][jdx][_type] = _type(in_planes=in_planes, out_planes=out_planes, stride=stride)
                    self.add_module(f'{idx}-{jdx}-{_type.__name__}', self.connections[(idx,jdx,_type.__name__)])


        # raise NotImplemented("In progress...")

    def forward(self, inputs1, inputs2, dag):
        inputs = [None] * (2 + self.args.num_blocks)
        outputs = [0] * (2 + self.args.num_blocks)
        num_inputs = [0] * (2 + self.args.num_blocks)
        inputs[0], inputs[1] = inputs1, inputs2
        inputs_relu = [None] * (2 + self.args.num_blocks)
        # self.inputs = [t.zeros((inputs.shape[0], self.num_filters, inputs.shape[2], inputs.shape[3]), dtype=torch.float32)] * (self.args.num_blocks + 1)
        # self.inputs[0] = inputs
        for source, target, _type in dag:
            # print(source, target, type)
            # print( self.connections[source][target][type])
            # print(self.inputs[source].shape)
            # print(type)
            key = (source, target, _type)
            conn = self.connections[key]

            if inputs[source] is None:
                outputs[source] /= num_inputs[source]
                inputs[source] = outputs[source]
            layer_input = inputs[source]
            if hasattr(conn, 'input_relu') and conn.input_relu:
                if inputs_relu[source] is None:
                    inputs_relu[source] = t.nn.functional.relu(layer_input)
                layer_input = inputs_relu[source]

            val = conn(layer_input) * self.dag_vars[key]
            # print(val.shape)
            outputs[target] += val
            num_inputs[target] += self.dag_vars[key]

        outputs[-1] /= num_inputs[-1]
        output = outputs[-1]
        return output

    def get_f(self, name):
        name = name.lower()


    def get_num_cell_parameters(self, dag):
        count = 0
        for source, target, type_name in dag:
            submodule = self.connections[(source, target, type_name)]
            model_parameters = filter(lambda p: p.requires_grad, submodule.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            count += params

        return count

    def reset_parameters(self):
        #TODO: Figure out if this should be implemented
        pass

    def to_device(self, device, dag):
        for source, target, type_name in dag:
            self.connections[(source, target, type_name)].to(device)

    def get_parameters(self, dag):
        params = []
        for key in dag:
            params.extend(self.connections[key].parameters())
        return params



class CNN(SharedModel):
    def __init__(self, args, input_channels, height, width, output_classes, gpu=None,
                 architecture=[('normal', 768//8)]*6 + [('reducing', 768//4)] + [('normal', 768//4)]*6 + [('reducing', 768//2)] +  [('normal', 768//2)]*6):
        super().__init__()

        self.args = args

        self.height = height
        self.width = width
        self.output_classes = output_classes
        self.architecture = architecture

        self.output_height = self.height
        self.output_width = self.width

        self.cells = nn.Sequential()
        self.gpu = gpu
        self.gpu_dag = set()
        self.gpu_reducing_dag = set()
        self.cpu_device = torch.device("cpu")

        self.dag_variables_dict = {}
        self.reducing_dag_variables_dict = {}

        last_input_info = CNNCell.InputInfo(input_channels=input_channels, input_width=width)
        current_input_info = CNNCell.InputInfo(input_channels=input_channels, input_width=width)

        #count connections
        temp_cell = CNNCell(args, input_1_info=last_input_info, input_2_info=current_input_info,
                output_channels=architecture[0][1], output_width=self.output_width, reducing=False, dag_vars=None)

        self.all_connections = list(temp_cell.connections.keys())

        self.dag_variables = t.ones(len(self.all_connections), requires_grad=True, device=self.gpu)
        self.reducing_dag_variables = t.ones(len(self.all_connections), requires_grad=True, device=self.gpu)

        for i, key in enumerate(self.all_connections):
            self.dag_variables_dict[key] = self.dag_variables[i]
            self.reducing_dag_variables_dict[key] = self.reducing_dag_variables[i]

        for i, (type, num_filters) in enumerate(architecture):
            if type == 'reducing':
                #TODO: do this calculation correctly
                self.output_height /= 2
                self.output_width /= 2
                reducing = True
            else:
                reducing = False
                assert(type == 'normal')

            dag_vars = self.dag_variables_dict if reducing == False else self.reducing_dag_variables_dict
            self.cells.add_module(f'{i}-{type}-{num_filters}', CNNCell(args, input_1_info=last_input_info, input_2_info=current_input_info,
                                            output_channels=num_filters, output_width=self.output_width, reducing=reducing, dag_vars=dag_vars))

            last_input_info, current_input_info = current_input_info, CNNCell.InputInfo(input_channels=num_filters, input_width=self.output_width)

        if self.output_classes:
            self.conv_output_size = self.output_height * self.output_width * self.architecture[-1][-1]
            self.out_layer = nn.Linear(self.conv_output_size, self.output_classes)
            # self.out_layer.weight.data.zero_()
            # self.out_layer.bias.data.zero_()
            # torch.nn.init.constant_(self.out_layer.weight, 0)
            torch.nn.init.constant_(self.out_layer.bias, 0)

        self.all_connections = list(self.cells[0].connections.keys())
        parent_counts = [0] * (2 + args.num_blocks)

        for idx, jdx, _type in self.all_connections:
            parent_counts[jdx] += 1

        print(parent_counts)

        probs = np.array(list(2 / parent_counts[jdx] for idx, jdx, _type in self.all_connections))
        self.dags_logits = (logit(probs), logit(probs))

        self.target_ave_prob = np.mean(probs)

    def forward(self, output, cell_dags):
        cell_dag, reducing_cell_dag = cell_dags
        last_input, current_input = output, output

        for cell in self.cells:
            if cell.reducing:
                dag = reducing_cell_dag
            else:
                dag = cell_dag
            # print(cell)
            output = cell(last_input, current_input, dag)
            last_input, current_input = current_input, output

        x = output.view(-1, self.conv_output_size)
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

    def update_dag_logits(self, gradient_dicts, weight_decay, max_grad=0.1):
        dag_grad_dict, reduction_dag_grad_dict = gradient_dicts[0], gradient_dicts[1]

        dag_probs = tuple(expit(logit) for logit in self.dags_logits)

        current_average_dag_probs = tuple(np.mean(prob) for prob in dag_probs)

        for i, key in enumerate(self.all_connections):
            for grad_dict, current_average_dag_prob, dag_logits in zip(gradient_dicts, current_average_dag_probs, self.dags_logits):
                if key in dag_grad_dict:
                    grad = grad_dict[key] - weight_decay * (current_average_dag_prob - self.target_ave_prob)  # *expit(dag_logits[i])
                    deriv = sigmoid_derivitive(dag_logits[i])
                    logit_grad = grad * deriv
                    dag_logits[i] += np.clip(logit_grad, -max_grad, max_grad)

    def get_dags_probs(self):
        return tuple(expit(logits) for logits in self.dags_logits)

    def to_device(self, device, cell_dag, reducing_cell_dag):
        for cell in self.cells:
            if cell.reducing:
                cell.to_device(device, reducing_cell_dag)
            else:
                cell.to_device(device, cell_dag)

        self.out_layer.to(device)

    def to_cpu(self):
        self.to_gpu(([], []))

    def to_gpu(self, cell_dags):
        cell_dag, reducing_cell_dag = cell_dags
        cell_dag = set(cell_dag)
        reducing_cell_dag = set(reducing_cell_dag)
        if self.gpu is None:
            raise Exception("No GPU given")
        else:
            cell_dag_to_gpu = cell_dag - self.gpu_dag
            cell_dag_to_cpu = self.gpu_dag - cell_dag

            reducing_cell_dag_to_gpu = reducing_cell_dag - self.gpu_reducing_dag
            reducing_cell_dag_to_cpu = self.gpu_reducing_dag - reducing_cell_dag

            self.gpu_reducing_dag = reducing_cell_dag
            self.gpu_dag = cell_dag

            self.to_device(self.cpu_device, cell_dag_to_cpu, reducing_cell_dag_to_cpu)
            self.to_device(self.gpu, cell_dag_to_gpu, reducing_cell_dag_to_gpu)

    def get_parameters(self, dags):
        dag, reducing_dag = dags
        params = []
        for cell in self.cells:
            if cell.reducing:
                d = reducing_dag
            else:
                d = dag
            params.extend(cell.get_parameters(d))
        return params


    def load(self, load_path):
        self.load_state_dict(torch.load(os.path.join(load_path, "cnn_model")))
        with open(os.path.join(load_path, "dags_logits.pickle"), 'rb') as f:
            self.dags_logits = pickle.load(f)

    def save(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, "cnn_model"))
        with open(os.path.join(save_path, "dags_logits.pickle"), 'wb') as f:
            pickle.dump(self.dags_logits, f)


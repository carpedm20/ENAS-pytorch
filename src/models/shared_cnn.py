import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np

from models.cnn_layers import CNN_LAYER_CREATION_FUNCTIONS, initialize_layers_weights
from scipy.special import expit, logit
from torch import nn

def sigmoid_derivitive(x):
    return expit(x)*(1.0-expit(x))

class CNNCell(torch.nn.Module):
    class InputInfo:
        def __init__(self, input_channels, input_width):
            self.input_channels = input_channels
            self.input_width = input_width

    def __init__(self, input_infos: list[InputInfo], output_channels, output_width, reducing, dag_vars, num_blocks):
        super().__init__()

        self.reset_parameters()

        self.input_infos = input_infos
        self.num_inputs = len(self.input_infos)
        self.num_blocks = num_blocks
        num_outputs = self.num_inputs + num_blocks
        self.output_channels = output_channels
        self.output_width = output_width
        self.reducing = reducing

        self.dag_vars = dag_vars

        self.connections = dict()
        for idx in range(num_outputs - 1):
            for jdx in range(max(idx+1, self.num_inputs), num_outputs):
                for _type in CNN_LAYER_CREATION_FUNCTIONS:
                    if idx < self.num_inputs:
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
                    self.connections[(idx, jdx, _type.__name__)] = _type(in_planes=in_planes, out_planes=out_planes, stride=stride)
                    initialize_layers_weights(self.connections[(idx, jdx, _type.__name__)])
                    self.add_module(f'{idx}-{jdx}-{_type.__name__}', self.connections[(idx,jdx,_type.__name__)])

    def forward(self, dag, *inputs):
        assert(len(inputs) == self.num_inputs)
        inputs = inputs + self.num_inputs * [None]
        outputs = [0] * (self.num_inputs + self.num_blocks)
        num_inputs = [0] * (self.num_inputs + self.num_blocks)
        inputs_relu = [None] * (self.num_inputs + self.num_blocks)


        for source, target, _type in dag:
            key = (source, target, _type)
            conn = self.connections[key]

            if inputs[source] is None:
                outputs[source] /= num_inputs[source]
                inputs[source] = outputs[source]
            layer_input = inputs[source]
            if hasattr(conn, 'input_relu') and conn.input_relu:
                if inputs_relu[source] is None:
                    inputs_relu[source] = torch.nn.functional.relu(layer_input)
                layer_input = inputs_relu[source]

            val = conn(layer_input) * self.dag_vars[key]
            outputs[target] += val
            num_inputs[target] += self.dag_vars[key]

        outputs[-1] /= num_inputs[-1]
        output = outputs[-1]
        return output

    def get_num_cell_parameters(self, dag):
        count = 0
        for source, target, type_name in dag:
            submodule = self.connections[(source, target, type_name)]
            model_parameters = filter(lambda p: p.requires_grad, submodule.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            count += params

        return count

    def to_device(self, device, dag):
        for source, target, type_name in dag:
            self.connections[(source, target, type_name)].to(device)

    def get_parameters(self, dag):
        params = []
        for key in dag:
            params.extend(self.connections[key].parameters())
        return params



class CNN(torch.nn.Module):
    def __init__(self, args, input_channels, height, width, output_classes, gpu,
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
        self.cpu_device = torch.device("cpu")

        self.dag_variables_dict = {}
        self.reducing_dag_variables_dict = {}

        last_input_info = CNNCell.InputInfo(input_channels=input_channels, input_width=width)
        current_input_info = CNNCell.InputInfo(input_channels=input_channels, input_width=width)

        #count connections
        temp_cell = CNNCell(input_infos=[last_input_info, current_input_info], output_channels=architecture[0][1],
                            output_width=self.output_width, reducing=False, dag_vars=None)

        self.all_connections = list(temp_cell.connections.keys())

        self.dag_variables = torch.ones(len(self.all_connections), requires_grad=True, device=self.gpu)
        self.reducing_dag_variables = torch.ones(len(self.all_connections), requires_grad=True, device=self.gpu)

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
            self.cells.add_module(f'{i}-{type}-{num_filters}', CNNCell(input_infos=[last_input_info, current_input_info],
                                            output_channels=num_filters, output_width=self.output_width, reducing=reducing, dag_vars=dag_vars, num_blocks=self.num_blocks))

            last_input_info, current_input_info = current_input_info, CNNCell.InputInfo(input_channels=num_filters, input_width=self.output_width)

        if self.output_classes:
            self.conv_output_size = self.output_height * self.output_width * self.architecture[-1][-1]
            self.out_layer = nn.Linear(self.conv_output_size, self.output_classes)
            torch.nn.init.kaiming_normal_(self.out_layer.weight, mode='fan_out', nonlinearity='relu')
            torch.nn.init.constant_(self.out_layer.bias, 0)
            self.out_layer.to(self.gpu)

        parent_counts = [0] * (2 + args.num_blocks)

        for idx, jdx, _type in self.all_connections:
            parent_counts[jdx] += 1

        print(parent_counts)

        probs = np.array(list(2 / parent_counts[jdx] for idx, jdx, _type in self.all_connections))
        self.dags_logits = (logit(probs), logit(probs))

        self.target_ave_prob = np.mean(probs)
        self.cell_dags = ([], [])

    def forward(self, input):
        """
        :param cell_dags: (normal_cell_dag, reduction_cell_dag)
        :param input: Input to Neural Network
        """
        cell_dag, reducing_cell_dag = self.cell_dags
        last_input, current_input = input, input

        for cell in self.cells:
            if cell.reducing:
                dag = reducing_cell_dag
            else:
                dag = cell_dag
            output = cell(dag, last_input, current_input)
            last_input, current_input = current_input, output

        x = output.view(-1, self.conv_output_size)
        x = self.out_layer(x)
        return x

    def update_dag_logits(self, gradient_dicts, weight_decay, max_grad=0.1):
        dag_probs = tuple(expit(logit) for logit in self.dags_logits)
        current_average_dag_probs = tuple(np.mean(prob) for prob in dag_probs)

        for i, key in enumerate(self.all_connections):
            for grad_dict, current_average_dag_prob, dag_logits in zip(gradient_dicts, current_average_dag_probs, self.dags_logits):
                if key in grad_dict:
                    grad = grad_dict[key] - weight_decay * (current_average_dag_prob - self.target_ave_prob)  # *expit(dag_logits[i])
                    deriv = sigmoid_derivitive(dag_logits[i])
                    logit_grad = grad * deriv
                    dag_logits[i] += np.clip(logit_grad, -max_grad, max_grad)

    def get_dags_probs(self):
        return tuple(expit(logits) for logits in self.dags_logits)

    def __to_device(self, device, cell_dags):
        cell_dag, reducing_cell_dag = cell_dags
        for cell in self.cells:
            if cell.reducing:
                cell.to_device(device, reducing_cell_dag)
            else:
                cell.to_device(device, cell_dag)

    def set_dags(self, new_cell_dags = ([], [])):
        """
        :param new_cell_dags: (normal_cell_dag, reduction_cell_dag)
        """
        new_cell_dags = tuple(list(sorted(cell_dag)) for cell_dag in new_cell_dags)

        set_cell_dags = [set(cell_dag) for cell_dag in new_cell_dags]
        last_set_cell_dags = [set(cell_dag) for cell_dag in self.cell_dags]

        cell_dags_to_cpu = [last_set_cell_dag - set_cell_dag
                            for last_set_cell_dag, set_cell_dag in zip(last_set_cell_dags, set_cell_dags)]
        cell_dags_to_gpu = [set_cell_dag - last_set_cell_dag
                            for last_set_cell_dag, set_cell_dag in zip(last_set_cell_dags, set_cell_dags)]

        self.__to_device(self.cpu_device, cell_dags_to_cpu)
        self.__to_device(self.gpu, cell_dags_to_gpu)
        self.cell_dags = new_cell_dags

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


import os
from collections import defaultdict, namedtuple

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from utils import draw_network, get_variable, keydefaultdict


Node = namedtuple('Node', ['id', 'name'])

class Controller(nn.Module):
    # Based on https://github.com/pytorch/examples/blob/master/word_language_model/model.py

    def __init__(self, args):
        super(Controller, self).__init__()
        self.args = args

        if self.args.network_type == 'rnn':
            self.num_tokens = [len(args.shared_rnn_activations)]
            for idx in range(self.args.num_blocks):
                    self.num_tokens += [idx + 1, len(args.shared_rnn_activations)]
            self.func_names = args.shared_rnn_activations
        elif self.args.network_type == 'cnn':
            self.num_tokens = [len(args.shared_cnn_types), self.args.num_blocks]
            self.func_names = args.shared_cnn_types

        num_total_tokens = sum(self.num_tokens)

        self.encoder = nn.Embedding(num_total_tokens, args.controller_hid)
        self.lstm = nn.LSTMCell(
                args.controller_hid,
                args.controller_hid)

        pivot = 0
        self.decoders = []

        for idx, size in enumerate(self.num_tokens):
            decoder = nn.Linear(args.controller_hid, size)
            self.decoders.append(decoder)

        self._decoders = nn.ModuleList(self.decoders)

        self.reset_parameters()
        self.static_init_hidden = keydefaultdict(self.init_hidden)

        fn = lambda key: get_variable(
                t.zeros(key, self.args.controller_hid), self.args.cuda, requires_grad=False)
        self.static_inputs = keydefaultdict(fn)

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)

    def forward(self, inputs, hidden, block_idx, is_embed):
        if not is_embed:
            embed = self.encoder(inputs)
        else:
            embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self.decoders[block_idx](hx)

        # exploration
        if self.args.mode == 'train':
            logits = self.args.tanh_c * F.tanh(logits)
        elif self.args.mode == 'derive':
            logits = logits / self.args.softmax_temperature

        return logits, (hx, cx)

    def sample(self, batch_size=1, with_details=False, save_dir=None):
        if batch_size < 1:
            raise Exception(f"Wrong batch_size: {batch_size} < 1")

        # [B, L, H]
        inputs = self.static_inputs[batch_size]
        hidden = self.static_init_hidden[batch_size]

        log_probs, entropies = [], []
        activations, prev_nodes = [], []

        for block_idx in range(2*(self.args.num_blocks-1) + 1):
            # 0: function, 1: previous node
            mode = block_idx % 2

            logits, hidden = self.forward(
                    inputs, hidden, block_idx,
                    is_embed=block_idx==0)

            probs = F.softmax(logits)
            log_prob = F.log_softmax(logits)
            entropy = -(log_prob * probs).sum(1, keepdim=True)
            entropies.append(entropy.data[0][0])

            action = probs.multinomial().data
            selected_log_prob = log_prob.gather(1, get_variable(action, requires_grad=False))
            log_probs.append(selected_log_prob)

            inputs = get_variable(action[:,0] + sum(self.num_tokens[:mode]), requires_grad=False)

            if mode == 0:
                activations.append(action[:,0])
            elif mode == 1:
                prev_nodes.append(action[:,0])

        prev_nodes = t.stack(prev_nodes).transpose(0, 1)
        activations = t.stack(activations).transpose(0, 1)

        dags = []
        for nodes, func_ids in zip(prev_nodes, activations):
            dag = defaultdict(list)

            # add first node
            dag[-1] = [Node(0, self.func_names[func_ids[0]])]
            dag[-2] = [Node(0, self.func_names[func_ids[0]])]

            # add following nodes
            for jdx, (idx, func_id) in enumerate(zip(nodes, func_ids[1:])):
                dag[idx].append(Node(jdx+1, self.func_names[func_id]))

            leaf_nodes = set(range(self.args.num_blocks)) - dag.keys()

            # merge with avg
            for idx in leaf_nodes:
                dag[idx] = [Node(self.args.num_blocks, 'avg')]

            # last h[t] node
            dag[self.args.num_blocks] = [Node(self.args.num_blocks + 1, 'h[t]')]
            dags.append(dag)

        if save_dir:
            for idx, dag in enumerate(dags):
                draw_network(dag, os.path.join(save_dir, f"graph{idx}.png"))

        if with_details:
            return dags, log_probs, entropies
        else:
            return dags

    def init_hidden(self, batch_size):
        zeros = t.zeros(batch_size, self.args.controller_hid)
        return (get_variable(zeros, self.args.cuda, requires_grad=False),
                get_variable(zeros.clone(), self.args.cuda, requires_grad=False))

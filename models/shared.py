import numpy as np
from collections import defaultdict, deque

import torch as t
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import get_logger, get_variable, keydefaultdict

logger = get_logger()


def size(p):
    return np.prod(p.size())

def embedded_dropout(embed, words, dropout=0.1, scale=None):
    # code from https://github.com/salesforce/awd-lstm-lm/blob/master/embed_regularize.py
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)) \
                .bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight

    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
            padding_idx = -1
    X = embed._backend.Embedding.apply(words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse
    )
    return X

class LockedDropout(nn.Module):
    # code from https://github.com/salesforce/awd-lstm-lm/blob/master/locked_dropout.py
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class RNN(nn.Module):
    def __init__(self, args, corpus):
        super(RNN, self).__init__()
        self.args = args
        self.corpus = corpus

        self.encoder = nn.Embedding(
                corpus.num_tokens, args.shared_embed)
        self.decoder = nn.Linear(args.shared_hid, corpus.num_tokens)
        self.lockdrop = LockedDropout()

        if self.args.tie_weights:
            self.decoder.weight = self.encoder.weight

        self.w_xh = nn.Linear(args.shared_embed + args.shared_hid, args.shared_hid)
        self.w_xc = nn.Linear(args.shared_embed + args.shared_hid, args.shared_hid)

        self.w_h, self.w_c = defaultdict(dict), defaultdict(dict)

        for idx in range(args.num_blocks):
            for jdx in range(idx+1, args.num_blocks):
                w_h = self.w_h[idx][jdx] = nn.Linear(
                        args.shared_hid, args.shared_hid, bias=False)
                w_c = self.w_c[idx][jdx] = nn.Linear(
                        args.shared_hid, args.shared_hid, bias=False)

        self._w_h = nn.ModuleList(
                [self.w_h[idx][jdx] for idx in self.w_h for jdx in self.w_h[idx]])
        self._w_c = nn.ModuleList(
                [self.w_c[idx][jdx] for idx in self.w_c for jdx in self.w_c[idx]])

        if args.mode == 'train':
            self.batch_norm = nn.BatchNorm1d(args.shared_hid)
        else:
            self.batch_norm = None

        self.reset_parameters()
        self.static_init_hidden = keydefaultdict(self.init_hidden)

        logger.info(f"# of parameters: {format(self.num_parameters, ',d')}")

    def forward(self, inputs, hidden, dag):
        time_steps = inputs.size(0)
        batch_size = inputs.size(1)

        if hidden is None:
            hidden = self.static_init_hidden[batch_size]

        embed = embedded_dropout(
                self.encoder, inputs,
                dropout=self.args.shared_dropoute \
                        if self.args.mode in ['train', 'derive'] else 0)

        if self.args.shared_dropouti > 0:
            embed = self.lockdrop(embed, self.args.shared_dropouti)

        logits = []
        for step in range(time_steps):
            x_t = embed[step]
            logit, hidden = self.cell(x_t, hidden, dag)
            logits.append(logit)
        
        output = t.stack(logits)
        if self.args.shared_dropout > 0:
            output = self.lockdrop(output, self.args.shared_dropout)

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def cell(self, x, h_prev, dag):
        c, h, f, outputs = {}, {}, {}, {}

        f[0] = self.get_f(dag[-1][0].name)
        c[0] = F.sigmoid(self.w_xc(t.cat([x, h_prev], -1)))
        h[0] = c[0] * f[0](self.w_xh(t.cat([x, h_prev], -1))) + (1 - c[0]) * h_prev

        leaf_node_ids = []
        q = deque()
        q.append(0)

        while True:
            if len(q) == 0:
                break

            node_id = q.popleft()
            nodes = dag[node_id]

            for next_node in nodes:
                next_id = next_node.id
                if next_id == self.args.num_blocks:
                    leaf_node_ids.append(node_id)
                    assert len(nodes) == 1, "parent of leaf node should have only one child"
                    continue

                w_h = self.w_h[node_id][next_id]
                w_c = self.w_c[node_id][next_id]

                f[next_id] = self.get_f(next_node.name)
                c[next_id] = F.sigmoid(w_c(h[node_id]))
                h[next_id] = c[next_id] * f[next_id](w_h(h[node_id])) + (1 - c[0]) * h[node_id]

                q.append(next_id)

        # average all the loose ends
        leaf_nodes = [h[node_id] for node_id in leaf_node_ids]
        output = t.mean(t.stack(leaf_nodes, 2), -1)

        # stabilizing the Updates of ω
        if self.batch_norm is not None:
            output = self.batch_norm(output)

        return output, h[self.args.num_blocks-1]

    def init_hidden(self, batch_size):
        zeros = t.zeros(batch_size, self.args.shared_hid)
        return get_variable(zeros, self.args.cuda, requires_grad=False)

    def get_f(self, name):
        name = name.lower()
        if name == 'relu':
            f = F.relu
        elif name == 'tanh':
            f = F.tanh
        elif name == 'identity':
            f = lambda x: x
        elif name == 'sigmoid':
            f = F.sigmoid
        return f

    @property
    def num_parameters(self):
        return sum([size(param) for param in self.parameters()])

    def get_num_cell_parameters(self, dag):
        num = 0

        num += size(self.w_xc)
        num += size(self.w_xh)

        q = deque()
        q.append(0)

        while True:
            if len(q) == 0:
                break

            node_id = q.popleft()
            nodes = dag[node_id]

            for next_node in nodes:
                next_id = next_node.id
                if next_id == self.args.num_blocks:
                    assert len(nodes) == 1, "parent of leaf node should have only one child"
                    continue

                w_h = self.w_h[node_id][next_id]
                w_c = self.w_c[node_id][next_id]

                num += size(w_h)
                num += size(w_c)

                q.append(next_id)

        logger.debug(f"# of cell parameters: {format(self.num_parameters, ',d')}")
        return num

    def reset_parameters(self):
        init_range = 0.025 if self.args.mode == 'train' else 0.04
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.fill_(0)


class CNN(nn.Module):
    def __init__(self, args, images):
        super(CNN, self).__init__()
        self.args = args
        self.images = images

        self.w_c, self.w_h = defaultdict(dict), defaultdict(dict)
        self.reset_parameters()

        raise NotImplemented("In progress...")

    def reset_parameters(self):
        init_range = self.args.shared_init_range

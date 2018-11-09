from __future__ import print_function

from collections import defaultdict
import collections
from datetime import datetime
import os
import json
import logging

import numpy as np
import pygraphviz as pgv

import torch
from torch.autograd import Variable

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 


try:
    import scipy.misc
    imread = scipy.misc.imread
    imresize = scipy.misc.imresize
    imsave = imwrite = scipy.misc.imsave
except:
    import cv2
    imread = cv2.imread
    imresize = cv2.imresize
    imsave = imwrite = cv2.imwrite


##########################
# Network visualization
##########################

def add_node(graph, node_id, label, shape='box', style='filled'):
    if label.startswith('x'):
        color = 'white'
    elif label.startswith('h'):
        color = 'skyblue'
    elif label == 'tanh':
        color = 'yellow'
    elif label == 'ReLU':
        color = 'pink'
    elif label == 'identity':
        color = 'orange'
    elif label == 'sigmoid':
        color = 'greenyellow'
    elif label == 'avg':
        color = 'seagreen3'
    else:
        color = 'white'

    if not any(label.startswith(word) for word in  ['x', 'avg', 'h']):
        label = f"{label}\n({node_id})"

    graph.add_node(
            node_id, label=label, color='black', fillcolor=color,
            shape=shape, style=style,
    )

def draw_network(dag, path):
    makedirs(os.path.dirname(path))
    graph = pgv.AGraph(directed=True, strict=True,
                       fontname='Helvetica', arrowtype='open') # not work?

    checked_ids = [-2, -1, 0]

    if -1 in dag:
        add_node(graph, -1, 'x[t]')
    if -2 in dag:
        add_node(graph, -2, 'h[t-1]')

    add_node(graph, 0, dag[-1][0].name)

    for idx in dag:
        for node in dag[idx]:
            if node.id not in checked_ids:
                add_node(graph, node.id, node.name)
                checked_ids.append(node.id)
            graph.add_edge(idx, node.id)

    graph.layout(prog='dot')
    graph.draw(path)

def make_gif(paths, gif_path, max_frame=50, prefix=""):
    import imageio

    paths.sort()

    skip_frame = len(paths) // max_frame
    paths = paths[::skip_frame]

    images = [imageio.imread(path) for path in paths]
    max_h, max_w, max_c = np.max(
            np.array([image.shape for image in images]), 0)

    for idx, image in enumerate(images):
        h, w, c = image.shape
        blank = np.ones([max_h, max_w, max_c], dtype=np.uint8) * 255

        pivot_h, pivot_w = (max_h-h)//2, (max_w-w)//2
        blank[pivot_h:pivot_h+h,pivot_w:pivot_w+w,:c] = image

        images[idx] = blank

    try:
        images = [Image.fromarray(image) for image in images]
        draws = [ImageDraw.Draw(image) for image in images]
        font = ImageFont.truetype("assets/arial.ttf", 30)

        steps = [int(os.path.basename(path).rsplit('.', 1)[0].split('-')[1]) for path in paths]
        for step, draw in zip(steps, draws):
            draw.text((max_h//20, max_h//20),
                      f"{prefix}step: {format(step, ',d')}", (0, 0, 0), font=font)
    except IndexError:
        pass

    imageio.mimsave(gif_path, [np.array(img) for img in images], duration=0.5)


##########################
# Torch
##########################

def detach(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(detach(v) for v in h)

def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def batchify(data, bsz, use_cuda):
    # code from https://github.com/pytorch/examples/blob/master/word_language_model/main.py 
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    if use_cuda:
        data = data.cuda()
    return data


##########################
# ETC
##########################

Node = collections.namedtuple('Node', ['id', 'name'])


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()


def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger


logger = get_logger()


def prepare_dirs(args):
    """Sets the directories for the model, and creates those directories.

    Args:
        args: Parsed from `argparse` in the `config` module.
    """
    if args.load_path:
        if args.load_path.startswith(args.log_dir):
            args.model_dir = args.load_path
        else:
            if args.load_path.startswith(args.dataset):
                args.model_name = args.load_path
            else:
                args.model_name = "{}_{}".format(args.dataset, args.load_path)
    else:
        args.model_name = "{}_{}".format(args.dataset, get_time())

    if not hasattr(args, 'model_dir'):
        args.model_dir = os.path.join(args.log_dir, args.model_name)
    args.data_path = os.path.join(args.data_dir, args.dataset)

    for path in [args.log_dir, args.data_dir, args.model_dir]:
        if not os.path.exists(path):
            makedirs(path)

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def save_args(args):
    param_path = os.path.join(args.model_dir, "params.json")

    logger.info("[*] MODEL dir: %s" % args.model_dir)
    logger.info("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)

def save_dag(args, dag, name):
    save_path = os.path.join(args.model_dir, name)
    logger.info("[*] Save dag : {}".format(save_path))
    json.dump(dag, open(save_path, 'w'))

def load_dag(args):
    load_path = os.path.join(args.dag_path)
    logger.info("[*] Load dag : {}".format(load_path))
    with open(load_path) as f:
        dag = json.load(f)
    dag = {int(k): [Node(el[0], el[1]) for el in v] for k, v in dag.items()}
    save_dag(args, dag, "dag.json")
    draw_network(dag, os.path.join(args.model_dir, "dag.png"))
    return dag          
  
def makedirs(path):
    if not os.path.exists(path):
        logger.info("[*] Make directories : {}".format(path))
        os.makedirs(path)

def remove_file(path):
    if os.path.exists(path):
        logger.info("[*] Removed: {}".format(path))
        os.remove(path)

def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)

    os.rename(path, new_path)
    logger.info("[*] {} has backup: {}".format(path, new_path))

#!/usr/bin/env python

import torch

import data
import models
import config
from utils import *
from trainer import Trainer
from utils import get_logger

logger = get_logger()


def main(args):
    prepare_dirs(args)

    torch.manual_seed(args.random_seed)

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)

    if args.network_type == 'rnn':
        dataset = data.text.Corpus(os.path.join(args.data_dir, args.dataset))
    elif args.dataset == 'cifar':
        dataset = data.image.Image(os.path.join(args.data_dir, args.dataset))
    else:
        raise NotImplemented(f"{args.dataset} is not supported")

    trainer = Trainer(args, dataset)

    if args.mode == 'train':
        save_args(args)
        trainer.train()
    elif args.mode == 'derive':
        assert args.load_path != "", "`--load_path` should be given in `derive` mode"
        trainer.derive()
    else:
        if not args.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)

"""Entry point."""
import os

import torch

import data
import config
import utils
import trainer

logger = utils.get_logger()


def main(args):  # pylint:disable=redefined-outer-name
    """main: Entry point."""
    utils.prepare_dirs(args)

    torch.manual_seed(args.random_seed)

    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)

    if args.network_type == 'rnn':
        dataset = data.text.Corpus(args.data_path)
    elif args.dataset == 'cifar':
        dataset = data.image.Image(args.data_path)
    else:
        raise NotImplementedError(f"{args.dataset} is not supported")

    trnr = trainer.Trainer(args, dataset)

    if args.mode == 'train':
        utils.save_args(args)
        trnr.train()
    elif args.mode == 'derive':
        assert args.load_path != "", ("`--load_path` should be given in "
                                      "`derive` mode")
        trnr.derive()
    elif args.mode == 'test':
        if not args.load_path:
            raise Exception("[!] You should specify `load_path` to load a "
                            "pretrained model")
        trnr.test()
    elif args.mode == 'single':
        if not args.dag_path:
            raise Exception("[!] You should specify `dag_path` to load a dag")
        utils.save_args(args)
        trnr.train(single=True)
    else:
        raise Exception(f"[!] Mode not found: {args.mode}")

if __name__ == "__main__":
    args, unparsed = config.get_args()
    main(args)

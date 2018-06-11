import argparse

arg_lists = []
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ('true')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')

# Controller
net_arg.add_argument('--num_blocks', type=int, default=5)

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='cifar10')


# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--batch_size', type=int, default=28)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--num_workers', type=int, default=0)


def get_args():
    """Parses all of the arguments above, which mostly correspond to the
    hyperparameters mentioned in the paper.
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    return args, unparsed

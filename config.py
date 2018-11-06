import argparse
from utils import get_logger

logger = get_logger()


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
net_arg.add_argument('--network_type', type=str, choices=['rnn', 'cnn'], default='rnn')

# Controller
net_arg.add_argument('--num_blocks', type=int, default=12)
net_arg.add_argument('--tie_weights', type=str2bool, default=True)
net_arg.add_argument('--controller_hid', type=int, default=100)

# Shared parameters for PTB
# NOTE(brendan): See Merity config for wdrop
# https://github.com/salesforce/awd-lstm-lm.
net_arg.add_argument('--shared_wdrop', type=float, default=0.5)
net_arg.add_argument('--shared_dropout', type=float, default=0.4) # TODO
net_arg.add_argument('--shared_dropoute', type=float, default=0.1) # TODO
net_arg.add_argument('--shared_dropouti', type=float, default=0.65) # TODO
net_arg.add_argument('--shared_embed', type=int, default=1000) # TODO: 200, 500, 1000
net_arg.add_argument('--shared_hid', type=int, default=1000)
net_arg.add_argument('--shared_rnn_max_length', type=int, default=35)
net_arg.add_argument('--shared_rnn_activations', type=eval,
                     default="['tanh', 'ReLU', 'identity', 'sigmoid']")
net_arg.add_argument('--shared_cnn_types', type=eval,
                     default="['3x3', '5x5', 'sep 3x3', 'sep 5x5', 'max 3x3', 'max 5x5']")

# PTB regularizations
net_arg.add_argument('--activation_regularization',
                     type=str2bool,
                     default=False)
net_arg.add_argument('--activation_regularization_amount',
                     type=float,
                     default=2.0)
net_arg.add_argument('--temporal_activation_regularization',
                     type=str2bool,
                     default=False)
net_arg.add_argument('--temporal_activation_regularization_amount',
                     type=float,
                     default=1.0)
net_arg.add_argument('--norm_stabilizer_regularization',
                     type=str2bool,
                     default=False)
net_arg.add_argument('--norm_stabilizer_regularization_amount',
                     type=float,
                     default=1.0)
net_arg.add_argument('--norm_stabilizer_fixed_point', type=float, default=5.0)

# Shared parameters for CIFAR
net_arg.add_argument('--cnn_hid', type=int, default=64)


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='ptb')


# Training / test parameters
learn_arg = add_argument_group('Learning')
learn_arg.add_argument('--mode', type=str, default='train',
                       choices=['train', 'derive', 'test', 'single'],
                       help='train: Training ENAS, derive: Deriving Architectures,\
                       single: training one dag')
learn_arg.add_argument('--batch_size', type=int, default=64)
learn_arg.add_argument('--test_batch_size', type=int, default=1)
learn_arg.add_argument('--max_epoch', type=int, default=150)
learn_arg.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])


# Controller
learn_arg.add_argument('--ppl_square', type=str2bool, default=False)
# NOTE(brendan): (Zoph and Le, 2017) page 8 states that c is a constant,
# usually set at 80.
learn_arg.add_argument('--reward_c', type=int, default=80,
                       help="WE DON'T KNOW WHAT THIS VALUE SHOULD BE") # TODO
# NOTE(brendan): irrelevant for actor critic.
learn_arg.add_argument('--ema_baseline_decay', type=float, default=0.95) # TODO: very important
learn_arg.add_argument('--discount', type=float, default=1.0) # TODO
learn_arg.add_argument('--controller_max_step', type=int, default=2000,
                       help='step for controller parameters')
learn_arg.add_argument('--controller_optim', type=str, default='adam')
learn_arg.add_argument('--controller_lr', type=float, default=3.5e-4,
                       help="will be ignored if --controller_lr_cosine=True")
learn_arg.add_argument('--controller_lr_cosine', type=str2bool, default=False)
learn_arg.add_argument('--controller_lr_max', type=float, default=0.05,
                       help="lr max for cosine schedule")
learn_arg.add_argument('--controller_lr_min', type=float, default=0.001,
                       help="lr min for cosine schedule")
learn_arg.add_argument('--controller_grad_clip', type=float, default=0)
learn_arg.add_argument('--tanh_c', type=float, default=2.5)
learn_arg.add_argument('--softmax_temperature', type=float, default=5.0)
learn_arg.add_argument('--entropy_coeff', type=float, default=1e-4)

# Shared parameters
learn_arg.add_argument('--shared_initial_step', type=int, default=0)
learn_arg.add_argument('--shared_max_step', type=int, default=400,
                       help='step for shared parameters')
# NOTE(brendan): Should be 10 for CNN architectures.
learn_arg.add_argument('--shared_num_sample', type=int, default=1,
                       help='# of Monte Carlo samples')
learn_arg.add_argument('--shared_optim', type=str, default='sgd')
learn_arg.add_argument('--shared_lr', type=float, default=20.0)
learn_arg.add_argument('--shared_decay', type=float, default=0.96)
learn_arg.add_argument('--shared_decay_after', type=float, default=15)
learn_arg.add_argument('--shared_l2_reg', type=float, default=1e-7)
learn_arg.add_argument('--shared_grad_clip', type=float, default=0.25)

# Deriving Architectures
learn_arg.add_argument('--derive_num_sample', type=int, default=100)


# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--save_epoch', type=int, default=4)
misc_arg.add_argument('--max_save_num', type=int, default=4)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--num_gpu', type=int, default=1)
misc_arg.add_argument('--random_seed', type=int, default=12345)
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True)
misc_arg.add_argument('--dag_path', type=str, default='')

def get_args():
    """Parses all of the arguments above, which mostly correspond to the
    hyperparameters mentioned in the paper.
    """
    args, unparsed = parser.parse_known_args()
    if args.num_gpu > 0:
        setattr(args, 'cuda', True)
    else:
        setattr(args, 'cuda', False)
    if len(unparsed) > 1:
        logger.info(f"Unparsed args: {unparsed}")
    return args, unparsed

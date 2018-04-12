# -*- coding: UTF-8 -*-
import argparse


def str2bool(v):
    """
    Convert string to boolean
    :param v: string，such as 'True', '1'
    :return: boolean data
    """
    return v.lower() in ('true', '1')


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    """
    Create and add a group of arguments
    :param name: Argument group name
    :return: Argument group
    """
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--hidden_dim', type=int, default=256, help='')
net_arg.add_argument('--input_dim', type=int, default=3, help='')
net_arg.add_argument('--max_enc_length', type=int, default=None, help='')
net_arg.add_argument('--max_dec_length', type=int, default=None, help='')
net_arg.add_argument('--init_min_val', type=float, default=-0.08, help='for uniform random initializer')
net_arg.add_argument('--init_max_val', type=float, default=+0.08, help='for uniform random initializer')
net_arg.add_argument('--init_first_decoder_input', type=str, default='avg')
net_arg.add_argument('--input_keep_prob', type=float, default=1.0)
net_arg.add_argument('--output_keep_prob', type=float, default=1.0)
net_arg.add_argument('--is_beam_search_used', type=bool, default=False)
net_arg.add_argument('--beam_size', type=int, default=3)


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--task', type=str, default='binpacking')
data_arg.add_argument('--batch_size', type=int, default=128)
data_arg.add_argument('--min_data_length', type=int, default=5)
data_arg.add_argument('--max_data_length', type=int, default=10)
data_arg.add_argument('--train_num', type=int, default=30000)
data_arg.add_argument('--valid_num', type=int, default=1000)
data_arg.add_argument('--test_num', type=int, default=1000)


# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True, help='')
train_arg.add_argument('--optimizer', type=str, default='rmsprop', help='')
train_arg.add_argument('--max_step', type=int, default=100000, help='')
train_arg.add_argument('--lr_start', type=float, default=0.001, help='')
train_arg.add_argument('--lr_decay_step', type=int, default=5000, help='')
train_arg.add_argument('--lr_decay_rate', type=float, default=0.96, help='')
train_arg.add_argument('--max_grad_norm', type=float, default=2.0, help='')
train_arg.add_argument('--checkpoint_secs', type=int, default=300, help='')


# Miscellaneous parameters
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=50, help='')
misc_arg.add_argument('--num_log_samples', type=int, default=5, help='')
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'], help='')
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--output_dir', type=str, default='outputs')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--gpu_memory_fraction', type=float, default=1.0)
misc_arg.add_argument('--random_seed', type=int, default=123, help='')


def get_config():
    """
    Parse the command line parameters
    :return: config: the parameters that have been parsed successfully, unparsed: the parameters that failed to parse
    """
    config, unparsed = parser.parse_known_args()
    return config, unparsed

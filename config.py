###############################################################################
#Copyright (C) 2017  Michael O. Vertolli michaelvertolli@gmail.com
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see http://www.gnu.org/licenses/
###############################################################################

#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--image_size', type=int, default=64, choices=[64, 128])
net_arg.add_argument('--conv_hidden_num', type=int, default=128,
                     choices=[128, 256],help='The number of convolutional filters.')
net_arg.add_argument('--z_num', type=int, default=128, choices=[128])
net_arg.add_argument('--loss_type', type=str, default='began', choices=['began',
                                                                        'began_gmsm',
                                                                        'began_gmsm_chrom',
                                                                        'scaled_began_gmsm',
                                                                        'scaled_began_gmsm_chrom'])

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--data_folder', type=str, default='CelebA')
data_arg.add_argument('--train', type=str2bool, default=True)
data_arg.add_argument('--batch_size', type=int, default=16, choices=[16])

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--gamma', type=float, default=0.7, choices=[0.5, 0.7])

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--model_tag', type=str, default='nvd_cg_ebm_full_reverse')
misc_arg.add_argument('--log_folder', type=str, default=None)
misc_arg.add_argument('--save_subgraphs', type=str2bool, default=True)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

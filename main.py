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

from trainer import Trainer
from trainer_config import config
import tensorflow as tf


def main(model='nvd_cg_ebm_full_reverse', type_='scaled_began_gmsm_b16_z128_sz64_h128_g0.7',
         data_folder='CelebA', log_folder=None, train=True, save_subgraphs=True):
    t = Trainer(model, type_, config(),
                log_folder, data_folder)
    if train:
        step = t.train()
    if save_subgraphs:
        t.c_graph.save_subgraphs(t.log_dir, step, t.sess)
    return t


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

from config import get_config
from trainer import Trainer
from trainer_config import config
import tensorflow as tf


def main(model='nvd_cg_ebm_full_reverse', type_='scaled_began_gmsm_b16_z128_sz64_h128_g0.7',
         data_folder='CelebA', log_folder=None, train=True, save_subgraphs=True):
    """Builds, trains and returns the trainer object.

    Arguments:
    model          := (str) a model name matching the corresponding folder in ./models
    type_          := (str) an aggregate type of model config parameters
    data_folder    := (str) the name of the data folder in ./data 
    log_folder     := (str) name of pre-existing log folder in ./logs to continue training model from
    train          := (boolean) whether to train model or just initialize
    save_subgraphs := (boolean) saves individual SubGraphs within log folder at end of training

    """
    t = Trainer(model, type_, config(),
                log_folder, data_folder)
    if train:
        step = t.train()
    if save_subgraphs:
        t.c_graph.save_subgraphs(t.log_dir, step, t.sess)
    return t


if __name__ == "__main__":
    parsed, unparsed = get_config()
    type_ = '_'.join([parsed.loss_type,
                      'b'+str(parsed.batch_size),
                      'z'+str(parsed.z_num),
                      'sz'+str(parsed.image_size),
                      'h'+str(parsed.conv_hidden_num),
                      'g'+str(parsed.gamma),
                      ])
    t = main(model=parsed.model_tag, type_=type_, data_folder=parsed.data_folder,
             log_folder=parsed.log_folder, train=parsed.train, save_subgraphs=parsed.save_subgraphs)
    

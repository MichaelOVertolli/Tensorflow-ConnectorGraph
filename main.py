from trainer import Trainer
from trainer_config import config
import tensorflow as tf


def main(model='cqs_cg', type_='began_b16_z1024_sz64_g0.5',
         log_folder=None, train=True, save_subgraphs=False):
    t = Trainer(model, type_, config(),
                log_folder, 'imgnet')
    if train:
        step = t.train()
    if save_subgraphs:
        t.c_graph.save_subgraphs(t.log_dir, step, t.sess)
    return t


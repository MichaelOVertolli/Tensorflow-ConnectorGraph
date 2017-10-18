from trainer import Trainer
from trainer_config import config
import tensorflow as tf


def main(train=True, save_subgraphs=False):
    t = Trainer('cqs_cg', '', config(),
                'cqs_cg')
    if train:
        step = t.train()
    if save_subgraphs:
        t.c_graph.save_subgraphs(t.log_dir, step, t.sess)
    return t


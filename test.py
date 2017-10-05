from trainer2 import Trainer
from trainer_config import config
import tensorflow as tf



def test():
    t = Trainer('cqs_cg', '', config(),
                'cqs_cg')
    return t


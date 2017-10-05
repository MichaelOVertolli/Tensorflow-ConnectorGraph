from data_loader import get_loader
from trainer2 import Trainer
from trainer_config import config
import tensorflow as tf



def test():
    c = config()
    ldr = get_loader(c.data_dir,
                     c.batch_size,
                     c.img_size,
                     c.data_format,
                     'train')
    t = Trainer('cqs_cg', '', c,
                'cqs_cg', ldr)
    return t


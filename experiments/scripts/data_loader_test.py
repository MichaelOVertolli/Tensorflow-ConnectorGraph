from data_loader import *


def test():
    d, n, i = setup_rdataset('/home/olias/data/celebc_h128_w128_c3.tfrecords',
                             300,
                             repeat=0,
                             greyscale=False,
                             norm=False,
                             shuffle=False,
                             bool_mask=None,
                             resize=None,
                             data_format=None)
    return d, n, i

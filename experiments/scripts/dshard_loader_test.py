from data_loader import *


def test():
    bools = [[True if (j*50)+i < (j+1)*50 else False for i in range(2000)] for j in range(25)]
    # bools.append([True if i > 1000 else False for i in range(2000-24)])
    d, n, i = setup_sharddata('/home/olias/data/cifar10r',
                             300,
                             repeat=0,
                             greyscale=False,
                             norm=False,
                             shuffle=True,
                             bool_masks=bools,
                             resize=None,
                             data_format=None)
    return d, n, i

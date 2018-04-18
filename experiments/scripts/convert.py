from models.res_cg_ebm_full_reverse.config import config
from models.graphs.res_cg_ebm_base import *
from models.graphs.converter import convert
import tensorflow as tf


def test():
    c = config('began_b16_z128_sz32_h128_g0.7_elu_pconv_wxav_alphas')
    g = build(c)
    conn, i, o, loss, train = convert(g, c, {})
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        conn.connect_graph(i, o, sess)
    return conn, i, o, graph

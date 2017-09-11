import os
from importlib import import_module
import tensorflow as tf


CONFIG_FILE = 'models.{}.config'
GRAPH_FILE = 'models.{}.graph'
GRAPH = 'graph'
META = 'graph.meta'
DIR = './models/'

def builder(model_name):
    path = os.path.join(DIR, model_name)
    files = os.listdir(path)
    config = import_module(CONFIG_FILE.format(model_name))
    graph_ = import_module(GRAPH_FILE.format(model_name))
    if META in files:
        graph = tf.Graph()
        with tf.Session(graph=graph) as sess:
            saver = tf.train.import_meta_graph(os.path.join(path, META))
            saver.restore(sess, os.path.join(path, GRAPH))
    else:
        graph, saver = graph_.build_graph(config.config())
        init = tf.variables_initializer(graph.get_collection('variables'))
        with tf.Session(graph=graph) as sess:
            sess.run(init)
            saver.save(sess, os.path.join(path, GRAPH), write_state=False)
    return graph

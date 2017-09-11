from importlib import import_module
import os
import tensorflow as tf


CONFIG_FILE = 'models.{}.config'
GRAPH_FILE = 'models.{}.graph'
GRAPH = 'graph'
META = 'graph.meta'
DIR = './models/'


class SubGraph(object):
    def __init__(self, model_name):
        self.name = model_name
        self.graph = self.builder(model_name)
        self.inputs = {}
        for tensor in self.graph.get_collection('inputs'):
            self.inputs[tensor.name] = tensor
        self.outputs = {}
        for tensor in self.graph.get_collection('outputs'):
            self.outputs[tensor.name] = tensor
        self.input_names = self.inputs.keys()
        self.output_names = self.outputs.keys()
        self.collections = [model_name+'_'+key for key in self.graph.get_all_collection_keys() if key != 'inputs' and key != 'outputs']

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

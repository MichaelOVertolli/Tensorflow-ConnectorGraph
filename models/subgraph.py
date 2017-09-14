from importlib import import_module
import os
import tensorflow as tf
from errors import SessionGraphError


CONFIG_FILE = 'models.{}.config'
GRAPH_FILE = 'models.{}.graph'
GRAPH = 'graph'
META = 'graph.meta'
DIR = './models/'


class SubGraph(object):
    def __init__(self, model_name, config_type, session):
        """Initializes a SubGraph object from model name and a session object.

        Arguments:
        model_name  := (string) unique folder name of model in models' folder
        config_type := (string) specifies which set of config parameters to use
        session     := (tf.Session) a reference to a session object with an empty Graph

        """
        if session.graph.version != 0:
            raise SessionGraphError('The session graph must be empty.')
        self.name = model_name
        self.graph = self.builder(model_name, config_type, session)
        self.inputs = {}
        for tensor in self.graph.get_collection('inputs'):
            self.inputs[tensor.name] = tensor
        self.outputs = {}
        for tensor in self.graph.get_collection('outputs'):
            self.outputs[tensor.name] = tensor
        self.input_names = self.inputs.keys()
        self.output_names = self.outputs.keys()
        self.collections = [key for key in self.graph.get_all_collection_keys()
                            if key != 'inputs' and key != 'outputs']


    def __eq__(self, other):
        #Not using duck typing for my own sanity.
        if type(other) is SubGraph:
            return self.name == other.name
        else:
            raise TypeError('Comparison with invalid type {}.'.format(type(other)))


    def builder(self, model_name, config_type, sess):
        module_name = '_'.join(model_name.split('_')[:-1]) #strips the index (_#) off of the model name
        path = os.path.join(DIR, module_name)
        files = os.listdir(path)
        config = import_module(CONFIG_FILE.format(module_name))
        graph_ = import_module(GRAPH_FILE.format(module_name))
        if META in files:
            saver = tf.train.import_meta_graph(os.path.join(path, META))
            saver.restore(sess, os.path.join(path, GRAPH))
        else:
            saver = graph_.build_graph(model_name, config.config(config_type))
            init = tf.variables_initializer(tf.get_collection('variables'))
            sess.run(init)
            saver.save(sess, os.path.join(path, GRAPH), write_state=False)
        return sess.graph

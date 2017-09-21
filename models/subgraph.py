from importlib import import_module
import os
import tensorflow as tf
from errors import SessionGraphError, FirstInitialization


SUBGRAPH_STR = "Model:\t{}\nInputs:\t{}\nOutputs\t{}\n"


class SubGraph(object):
    def __init__(self, name, config_type, graph):
        self.name = name
        self.config_type = config_type
        self.graph = graph
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
        try:
            return self.name == other.name
        except AttributeError:
            raise TypeError('{} is an invalid type.'.format(type(other)))


    def __str__(self):
        return SUBGRAPH_STR.format(self.name+self.config_type,
                                   ', '.join(self.input_names),
                                   ', '.join(self.output_names))

CONFIG_FILE = 'models.{}.config'
GRAPH_FILE = 'models.{}.graph'
GRAPH = 'graph_{}'
META = 'graph_{}.meta'
DIR = './models/'


class BuiltSubGraph(SubGraph):
    def __init__(self, model_name, config_type, session):
        """Initializes a SubGraph object from model name and a session object.

        Arguments:
        model_name  := (string) unique folder name of model in models' folder
        config_type := (string) specifies which set of config parameters to use
        session     := (tf.Session) a reference to a session object with an empty Graph

        """
        if session.graph.version != 0:
            raise SessionGraphError('The session graph must be empty.')
        if self.graph_is_built(model_name, config_type):
            self.restore(model_name, config_type, session)
            super(BuiltSubGraph, self).__init__(model_name, config_type, session.graph)
        else:
            self.build(model_name, config_type, session)
            raise FirstInitialization('SubGraph object {} must be re-initialized.'.format(model_name))


    def build(self, model_name, config_type, sess):
        module_name = self.get_module_name(model_name)
        config = import_module(CONFIG_FILE.format(module_name))
        graph_ = import_module(GRAPH_FILE.format(module_name))
        self.saver = graph_.build_graph(model_name, config.config(config_type))
        init = tf.variables_initializer(tf.get_collection('variables'))
        sess.run(init)
        self.saver.save(sess, os.path.join(self.path, GRAPH.format(config_type)), write_state=False)

    def graph_is_built(self, model_name, config_type):
        module_name = self.get_module_name(model_name)
        self.path = os.path.join(DIR, module_name)
        files = os.listdir(self.path)
        return META.format(config_type) in files


    def get_module_name(self, model_name):
        return '_'.join(model_name.split('_')[:-1]) #strips the index (_#) off of the model name

    
    def restore(self, model_name, config_type, sess, input_map=None):
        self.saver = tf.train.import_meta_graph(os.path.join(self.path,
                                                             META.format(config_type)),
                                                import_scope=model_name,
                                                input_map=input_map)
        self.saver.restore(sess, os.path.join(self.path, GRAPH.format(config_type)))




###############################################################################
#Copyright (C) 2017  Michael O. Vertolli michaelvertolli@gmail.com
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see http://www.gnu.org/licenses/
###############################################################################

from importlib import import_module
import os
import tensorflow as tf
from errors import SessionGraphError, FirstInitialization, InvalidSubGraphError


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
        self.collections = [key for key in self.graph.get_all_collection_keys()]

    def __eq__(self, other):
        try:
            return self.name == other.name
        except AttributeError:
            raise TypeError('{} is an invalid type.'.format(type(other)))


    def __str__(self):
        return SUBGRAPH_STR.format(self.name+self.config_type,
                                   ', '.join(self.input_names),
                                   ', '.join(self.output_names))


    def restore(self, model_name, config_type, sess, input_map=None):
        if self.collections:
            raise InvalidSubGraphError('Base SubGraph cannot have collections. ' +
                                       'Consider using BuiltSubGraph.')
        else:
            tf.import_graph_def(self.frozen_graph_def,
                                input_map=input_map,
                                name=model_name)


CONFIG_FILE = 'models.{}.config'
GRAPH_FILE = 'models.{}.graph'
GRAPH = 'graph_{}'
META = 'graph_{}.meta'
DIR = './models/'
CHKPNT = 'checkpoint'


class BuiltSubGraph(SubGraph):
    def __init__(self, model_name, config_type, session, log_dir=None):
        """Initializes a SubGraph object from model name and a session object.

        Arguments:
        model_name  := (string) unique folder name of model in models' folder
        config_type := (string) specifies which set of config parameters to use
        session     := (tf.Session) a reference to a session object with an empty Graph

        """
        if session.graph.version != 0:
            raise SessionGraphError('The session graph must be empty.')
        self.log_dir = log_dir
        if self.log_dir is not None:
            with open(os.path.join(self.log_dir, CHKPNT)) as f:
                txt = f.readline()
            # gets checkpoint file name from log_dir
            self.log_dir = os.path.join(self.log_dir, txt.split(': ')[1][1:-2])
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
                                                clear_devices=True,
                                                import_scope=model_name,
                                                input_map=input_map)
        if self.log_dir is None:
            self.saver.restore(sess, os.path.join(self.path, GRAPH.format(config_type)))
        else:
            self.saver = tf.train.Saver(sess.graph.get_collection('variables'))
            self.saver.restore(sess, self.log_dir)


    def freeze(self, sess):
        #designed to be called during initialization
        #sess must include the graph of this subgraph
        #self.restore(self.name, self.config_type, sess)
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(),
            self.strip_names(self.output_names))
        #assume this is only ever used to initialize a FrozenSubGraph
        #so we can set the name parameter to 0
        name = 'frozen_{}_0'.format(self.get_module_name(self.name))
        frozen = FrozenSubGraph(name, self.config_type, frozen_graph_def,
                                ['/'.join([name, tensor_name])
                                 for tensor_name in self.input_names],
                                ['/'.join([name, tensor_name])
                                 for tensor_name in self.output_names])
        return frozen


    def strip_names(self, names):
        return [name.split(':')[0] for name in names]


class FrozenSubGraph(SubGraph):
    def __init__(self, model_name, config_type, frozen_graph_def, inpts, outpts):
        super(FrozenSubGraph, self).__init__(model_name, config_type, tf.Graph())
        self.frozen_graph_def = frozen_graph_def
        self.input_names = inpts
        self.output_names = outpts


    def restore(self, model_name, config_type, sess, input_map=None):
        tf.import_graph_def(self.frozen_graph_def,
                            input_map=input_map,
                            name=model_name)

    def copy(self, new_name):
        return FrozenSubGraph(new_name, self.config_type,
                              self.frozen_graph_def,
                              [self.rename(new_name, tensor_name)
                               for tensor_name in self.input_names],
                              [self.rename(new_name, tensor_name)
                               for tensor_name in self.output_names])


    def rename(self, new_name, tensor_name):
        return '/'.join([new_name]+tensor_name.split('/')[1:])

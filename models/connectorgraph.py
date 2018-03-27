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

from errors import MissingSubgraphError, MissingTensorError, MissingConnectionError
from errors import ConnectionConflictError, ExhaustedGraphStepsError
from errors import SessionGraphError, GraphNotConnectedError
from errors import NoVariableExistsError, MultipleVariablesExistError
from errors import NoSaversError
import os
from subgraph import FrozenSubGraph
import tensorflow as tf
from types import MethodType


VARS = '/variables'
MODEL_VARS = '/model_variables'
TRAIN_VARS = '/trainable_variables'


class ConnectorGraph(object):

    def __init__(self, config):
        self._MAX_GRAPH_STEPS = 10000 #sanity constant for maximum depth of the graph
                                      #helpful for cycle check
        self.subgraphs = {}
        #connections are fromgraph_fromtensor_tograph_totensor: Connection()
        self.connections = {}
        self.config = config


    def connect_graph(self, inputs, outputs, sess):
        if sess.graph.version != 0:
            raise SessionGraphError('The session graph must be empty.')
        self.input_maps = dict([(subgraph_name, None) for subgraph_name in self.subgraphs.keys()])
        processed = dict([(subgraph_name, False) for subgraph_name in self.subgraphs.keys()])
        pending = self.get_start_subgraphs()
        for count in range(self._MAX_GRAPH_STEPS):
            try:
                subgraph_name = pending.pop(0)
            except IndexError:
                break
            subgraph = self.subgraphs[subgraph_name]
            
            subgraph.restore(subgraph_name, subgraph.config_type, sess,
                             self.input_maps[subgraph_name])
            self.rename_collections(subgraph, sess)
            processed[subgraph_name] = True
            forward_conn = self.get_forward_connections(subgraph_name)
            for conn in forward_conn:
                if not processed[conn.to_graph]:
                    #clearer and handles initialization
                    self.input_maps[conn.to_graph] = self.build_input_map(sess.graph, [conn], self.input_maps[conn.to_graph])
                    completed_back_conns = [processed[c.from_graph]
                                            for c in self.get_back_connections(conn.to_graph)]
                    if all(completed_back_conns) and conn.to_graph not in pending:
                        #guarantees that only subgraphs that have all their
                        #back connections already processed get added to pending list
                        pending.append(conn.to_graph) 
        if count == self._MAX_GRAPH_STEPS - 1:
            raise ExhaustedGraphStepsError('Probable cycle in connections. Otherwise, increase MAX_GRAPH_STEPS size.')
        for inpt in inputs:
            tf.add_to_collection('inputs', sess.graph.get_tensor_by_name(inpt))
        for output in outputs:
            tf.add_to_collection('outputs', sess.graph.get_tensor_by_name(output))
        self.aggregate_var_collections(sess)
        #TODO: if going to preserve graph, then should have error checking
        #      when collections are modified, which invalidates graph state 
        self.graph = sess.graph


    def build_input_map(self, graph, connections, input_map):
        if input_map is None:
            input_map = {}
        for conn in connections:
            to_tensor = '/'.join(conn.to_tensor.split('/')[1:])
            from_tensor = conn.from_tensor
            input_map[to_tensor] = graph.get_tensor_by_name(from_tensor)
        return input_map


    def rename_collections(self, subgraph, sess):
        for c in subgraph.collections:
            name = '/'.join([subgraph.name, c])
            for x in tf.get_collection(c):
                tf.add_to_collection(name, x)
            sess.graph.clear_collection(c)
            # tensors = [sess.graph.get_tensor_by_name(tensor.name)
            #            for tensor in subgraph.graph.get_collection(collection)
            #            if 'null' not in tensor.name]
            # for tensor in tensors:
            #     tf.add_to_collection(name, tensor)


    def aggregate_var_collections(self, sess):
        for c in sess.graph.get_all_collection_keys():
            if VARS in c:
                for v in sess.graph.get_collection(c):
                    sess.graph.add_to_collection(VARS[1:], v)
            elif MODEL_VARS in c:
                for v in sess.graph.get_collection(c):
                    sess.graph.add_to_collection(MODEL_VARS[1:], v)
            elif TRAIN_VARS in c:
                for v in sess.graph.get_collection(c):
                    sess.graph.add_to_collection(TRAIN_VARS[1:], v)
            

    def add_subgraph(self, subgraph):
        self.subgraphs[subgraph.name] = subgraph


    def remove_subgraph(self, subgraph):
        try:
            temp = self.subgraphs[subgraph.name]
        except KeyError:
            raise MissingSubgraphError('{} has not been added as a subgraph.'.format(subgraph))
        else:
            del self.subgraphs[subgraph.name]


    def quick_connect(self, connection_lists):
        for conn_list in connection_lists:
            self.add_connection(*conn_list)


    #TODO: add the tensor compatibility check in here
    def add_connection(self, from_graph, to_graph, from_tensor, to_tensor):
        """Adds a connection between subgraphs.

        Arguments:
        from_graph  := (string) the name of the subgraph the connection starts from
        to_graph    := (string) the name of the subgraph the connection terminates in
        from_tensor := (string) the name of the tensor that initiates the connection
        to_tensor   := (string) the name of the tensor that terminates the connection

        """
        identifier = '_'.join([from_graph, from_tensor, to_graph, to_tensor]) 
        try:
            fg = self.subgraphs[from_graph]
        except KeyError:
            raise MissingSubgraphError('{} has not been added as a subgraph.'.format(from_graph))
        try:
            tg = self.subgraphs[to_graph]
        except KeyError:
            raise MissingSubgraphError('{} has not been added as a subgraph.'.format(to_graph))
        if not from_tensor in fg.output_names:
            raise MissingTensorError('{} is not a valid output tensor in the {} subgraph.'.format(from_tensor, from_graph))
        if not to_tensor in tg.input_names:
            raise MissingTensorError('{} is not a valid input tensor in the {} subgraph.'.format(to_tensor, to_graph))
        #check if to_tensor already mapped
        to_ident = '_'.join([to_graph, to_tensor])
        already_mapped = [conn_key if (conn_key != identifier and to_ident in conn_key) else False
                          for conn_key in self.connections.keys()]
        if any(already_mapped):
            #There can only ever be one conflicting connection as the error occurs the moment there is one.
            conflict_conn = self.connections[[_ for _ in already_mapped if _ != False][0]]
            conflict_graph = conflict_conn.from_graph
            conflict_tensor = conflict_conn.from_tensor
            error_text = 'Tensor {} of subgraph {} already has an input connection from tensor {} of subgraph {}.'.format(
                to_tensor, to_graph, conflict_tensor, conflict_graph)
            raise ConnectionConflictError(error_text)
        self.connections[identifier] = Connection(from_graph, to_graph, from_tensor, to_tensor)

    def remove_connection(self, from_graph, to_graph, from_tensor, to_tensor):
        """Removes a connection between subgraphs.

        Arguments:
        from_graph  := (string) the name of the subgraph the connection starts from
        to_graph    := (string) the name of the subgraph the connection terminates in
        from_tensor := (string) the name of the tensor that initiates the connection
        to_tensor   := (string) the name of the tensor that terminates the connection

        """
        identifier = '_'.join([from_graph, from_tensor, to_graph, to_tensor]) 
        try:
            fg = self.subgraphs[from_graph]
        except KeyError:
            raise MissingSubgraphError('{} has not been added as a subgraph.'.format(from_graph))
        try:
            tg = self.subgraphs[to_graph]
        except KeyError:
            raise MissingSubgraphError('{} has not been added as a subgraph.'.format(to_graph))
        if not from_tensor in fg.output_names:
            raise MissingTensorError('{} is not a valid output tensor in the {} subgraph.'.format(from_tensor, from_graph))
        if not to_tensor in tg.input_names:
            raise MissingTensorError('{} is not a valid input tensor in the {} subgraph.'.format(to_tensor, to_graph))
        try:
            _ = self.connections[identifier]
        except KeyError:
            raise MissingConnectionError('The connection {} does not exist.'.format(identifier))
        else:
            del self.connections[identifier]


    def get_back_connections(self, subgraph_name):
        return self._get_subgraph_connections(subgraph_name, 'to_graph')


    def get_forward_connections(self, subgraph_name):
        return self._get_subgraph_connections(subgraph_name, 'from_graph')


    def _get_subgraph_connections(self, subgraph_name, type_attribute):
        try:
            subgraph = self.subgraphs[subgraph_name]
        except KeyError:
            raise MissingSubgraphError('{} has not been added as a subgraph.'.format(subgraph_name))
        connections = []
        for identifier, conn in self.connections.items():
            if getattr(conn, type_attribute) == subgraph_name:
                connections.append(conn)
        return connections


    def get_start_subgraphs(self):
        subgraphs = set(self.get_from_subgraphs())
        subgraphs -= set(self.get_to_subgraphs())
        return list(subgraphs)


    def get_end_subgraphs(self):
        subgraphs = set(self.get_to_subgraphs())
        subgraphs -= set(self.get_from_subgraphs())
        return list(subgraphs)


    def get_from_subgraphs(self):
        return self._get_conn_part('from_graph')


    def get_to_subgraphs(self):
        return self._get_conn_part('to_graph')


    def get_from_tensors(self, connections):
        return self._get_conn_part('from_tensor', connections=None)


    def get_to_tensors(self, connections):
        return self._get_conn_part('to_tensor', connections=None)


    def _get_conn_part(self, type_attribute, connections=None):
        if connections is None:
            connections = self.connections.values()
        subgraphs = []
        for conn in connections:
            subgraphs.append(getattr(conn, type_attribute))
        return subgraphs


    def tensors_to_names(self, tensors):
        return [t.name for t in tensors]


    def get_variable(self, variable_name):
        results = [var for var in self.graph.get_collection('variables') if variable_name == var.name]
        if len(results) == 0:
            raise NoVariableExistsError('{} does not exist.'.format(variable_name))
        elif len(results) > 1:
            raise MultipleVariablesExistError('{} matches multiple variables.'.format(variable_name))
        else:
            return results[0]


    def get_all_variables(self, variable_names):
        variables = []
        for n in variable_names:
            variables.append(self.get_variable(n))
        return variables


    def get_feed_dict(self, trainer):
        raise NotImplementedError('get_feed_dict must be overridden in ConnectorGraph construction.')


    def send_outputs(self, trainer, step):
        raise NotImplementedError('send_outputs must be overridden in ConnectorGraph construction.')


    def set_block_index(self, block_index):
        self.block_index = block_index


    def set_init_params(self, saver, log_dir):
        self.init_params = [saver, log_dir]


    def initialize(self, sess):
        saver, log_dir = self.init_params
        saver.restore(sess, log_dir)


    def attach_func(self, func):
        if callable(func):
            if func.__name__ == 'get_feed_dict':
                self.get_feed_dict = MethodType(func, self)
            elif func.__name__ == 'send_outputs':
                self.send_outputs = MethodType(func, self)
            else:
                raise AttributeError('{} does not exist or cannot be attached.'.format(func))
        else:
            raise TypeError('{} cannot be attached. Must be a callable type.'.format(func))


    def print_subgraphs(self):
        for subgraph in self.subgraphs.values():
            print subgraph


    def save_graph(self, folder):
        if self.graph is None:
            raise GraphNotConnectedError('There is no graph to save.')
        else:
            f = tf.summary.FileWriter(folder, self.graph)


    def add_subgraph_savers(self, savers):
        self.savers = savers


    def save_subgraphs(self, logdir, step, sess):
        if self.savers is None:
            raise NoSaversError('There are no savers in this Connectorgraph')
        else:
            for subgraph, saver_def in self.savers.items():
                path = os.path.join(logdir, subgraph)
                if not os.path.isdir(path):
                    os.makedirs(path)
                saver = tf.train.Saver(saver_def=saver_def)
                saver.save(sess, os.path.join(path, subgraph), None)


    def close(self):
        del self._MAX_GRAPH_STEPS
        for sub in self.subgraphs.values():
            sub.close()
        self.subgraphs.clear()
        del self.subgraphs
        self.input_maps.clear()
        del self.input_maps
        for con in self.connections.values():
            con.close()
        self.connections.clear()
        del self.connections
        self.savers.clear()
        del self.savers
        del self.graph
        del self.config
        try:
            del self.get_feed_dict
        except AttributeError:
            pass
        try:
            del self.send_outputs
        except AttributeError:
            pass
        del self.init_params[:]
        del self.init_params
        del self.block_index

    
class Connection(object):
    def __init__(self, from_graph, to_graph, from_tensor, to_tensor):
        self.from_graph = from_graph
        self.to_graph = to_graph
        self.from_tensor = from_tensor
        self.to_tensor = to_tensor


    def __eq__(self, other):
        return self.from_graph == other.from_graph and \
            self.to_graph == other.to_graph and \
            self.from_tensor == other.from_tensor and \
            self.to_tensor == other.to_tensor


    def __str__(self):
        return ', '.join([self.from_graph, self.to_graph, self.from_tensor, self.to_tensor])


    def close(self):
        del self.from_graph
        del self.to_graph
        del self.from_tensor
        del self.to_tensor

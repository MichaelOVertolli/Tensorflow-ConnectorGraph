from errors import MissingSubgraphError, MissingTensorError, MissingConnectionError
from errors import ConnectionConflictError, ExhaustedGraphStepsError
from errors import SessionGraphError
from subgraph import SubGraph
import tensorflow as tf


class ConnectorGraph(object):

    def __init__(self):
        self._MAX_GRAPH_STEPS = 10000 #sanity constant for maximum depth of the graph
                                      #helpful for cycle check
        self.subgraphs = {}
        #connections are fromgraph_fromtensor_tograph_totensor: Connection()
        self.connections = {}


    def connect_graph(self, inputs, outputs, sess):
        if sess.graph.version != 0:
            raise SessionGraphError('The session graph must be empty.')
        input_maps = dict([(subgraph_name, None) for subgraph_name in self.subgraphs.keys()])
        processed = dict([(subgraph_name, False) for subgraph_name in self.subgraphs.keys()])
        pending = self.get_start_subgraphs()
        for count in range(self._MAX_GRAPH_STEPS):
            try:
                subgraph_name = pending.pop(0)
            except IndexError:
                break
            subgraph = self.subgraphs[subgraph_name]
            completed_back_conns = [processed[conn.from_graph]
                                    for conn in self.get_back_connections(subgraph_name)]
            if not all(completed_back_conns):
                #Push the graph to the back of the queue
                #If there is a cycle, this will repeat indefinitely
                pending.append(subgraph_name)
            else:
                _ = tf.import_graph_def(subgraph.graph.as_graph_def(),
                                        input_map=input_maps[subgraph_name],
                                        name='')
            self.add_collections(subgraph, sess)
            processed[subgraph_name] = True
            forward_conn = self.get_forward_connections(subgraph_name)
            for conn in forward_conn:
                if not processed[conn.to_graph]:
                    #clearer and handles initialization
                    input_maps[conn.to_graph] = self.build_input_map(sess.graph, [conn], input_maps[conn.to_graph])
                    pending.append(conn.to_graph)
        if count == self._MAX_GRAPH_STEPS - 1:
            raise ExhaustedGraphStepsError('Probable cycle in connections. Otherwise, increase MAX_GRAPH_STEPS size.')
        for inpt in inputs:
            tf.add_to_collection('inputs', sess.graph.get_tensor_by_name(inpt))
        for output in outputs:
            tf.add_to_collection('outputs', sess.graph.get_tensor_by_name(output))
        return sess.graph


    def build_input_map(self, graph, connections, input_map):
        if input_map is None:
            input_map = {}
        for conn in connections:
            input_map[conn.to_tensor] = graph.get_tensor_by_name(conn.from_tensor)
        return input_map


    def add_collections(self, subgraph, sess):
        for collection in subgraph.collections:
            name = '/'.join([subgraph.name, collection])
            tensors = [sess.graph.get_tensor_by_name(tensor.name)
                       for tensor in subgraph.graph.get_collection(collection)
                       if 'null' not in tensor.name]
            for tensor in tensors:
                tf.add_to_collection(name, tensor)


    def add_subgraph(self, subgraph):
        self.subgraphs[subgraph.name] = subgraph


    def remove_subgraph(self, subgraph):
        try:
            temp = self.subgraphs[subgraph.name]
        except KeyError:
            raise MissingSubgraphError('{} has not been added as a subgraph.'.format(subgraph))
        else:
            del self.subgraphs[subgraph.name]


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

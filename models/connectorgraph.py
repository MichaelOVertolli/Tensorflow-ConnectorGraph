from errors import MissingSubgraphError, MissingTensorError, MissingConnectionError
import tensorflow as tf


class ConnectorGraph(object):

    def __init__(self):
        self.subgraphs = {}
        #connections are from_to: Connection()
        self.connections = {}


    def connect_graph(self):
        graph = tf.Graph()
        start_subgraphs = self.get_start_subgraphs()
        end_subgraphs = self.get_end_subgraphs()
        with tf.Session(graph=graph) as sess:
            for subgraph in start_subgraphs:
                connections = self.get_subgraph_connections(subgraph.name)
                for conn in connections:
                    


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
        identifier = from_graph+'_'+to_graph
        try:
            fg = self.subgraphs[from_graph]
        except KeyError:
            raise MissingSubgraphError('{} has not been added as a subgraph.'.format(from_graph))
        try:
            tg = self.subgraphs[to_graph]
        except KeyError:
            raise MissingSubgraphError('{} has not been added as a subgraph.'.format(to_graph))
        if not from_tensor in fg.input_names:
            raise MissingTensor('{} is not a valid output tensor in the {} subgraph.'.format(from_tensor, from_graph))
        if not to_tensor in tg.input_names:
            raise MissingTensor('{} is not a valid input tensor in the {} subgraph.'.format(to_tensor, to_graph))
        self.connections[identifier] = Connection(from_graph, to_graph, from_tensor, to_tensor)

    def remove_connection(self, from_graph, to_graph, from_tensor, to_tensor):
        """Removes a connection between subgraphs.

        Arguments:
        from_graph  := (string) the name of the subgraph the connection starts from
        to_graph    := (string) the name of the subgraph the connection terminates in
        from_tensor := (string) the name of the tensor that initiates the connection
        to_tensor   := (string) the name of the tensor that terminates the connection

        """
        identifier = from_graph+'_'+to_graph
        try:
            fg = self.subgraphs[from_graph]
        except KeyError:
            raise MissingSubgraphError('{} has not been added as a subgraph.'.format(from_graph))
        try:
            tg = self.subgraphs[to_graph]
        except KeyError:
            raise MissingSubgraphError('{} has not been added as a subgraph.'.format(to_graph))
        if not from_tensor in fg.input_names:
            raise MissingTensor('{} is not a valid output tensor in the {} subgraph.'.format(from_tensor, from_graph))
        if not to_tensor in tg.input_names:
            raise MissingTensor('{} is not a valid input tensor in the {} subgraph.'.format(to_tensor, to_graph))
        try:
            _ = self.connections[identifier]
        except KeyError:
            raise MissingConnectionError('The connection {} does not exist.'.format(identifier))
        else:
            del self.connections[identifier]


    def get_subgraph_connections(self, subgraph_name):
        try:
            subgraph = self.subgraphs[subgraph_name]
        except KeyError:
            raise MissingSubgraphError('{} has not been added as a subgraph.'.format(subgraph_name))
        connections = []
        for identifier, conn in self.connections.values():
            if conn.from_graph == subgraph_name:
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


    def get_from_tensors(self):
        return self._get_conn_part('from_tensor')


    def get_to_tensors(self):
        return self._get_conn_part('to_tensor')


    def _get_conn_part(self, type_attribute):
        subgraphs = []
        for identifier, conn in self.connections.values():
            subgraphs.append(getattr(conn, type_attribute))
        return subgraphs

    
class Connection(object):
    def __init__(from_graph, to_graph, from_tensor, to_tensor):
        self.from_graph = from_graph
        self.to_graph = to_graph
        self.from_tensor = from_tensor
        self.to_tensor = to_tensor

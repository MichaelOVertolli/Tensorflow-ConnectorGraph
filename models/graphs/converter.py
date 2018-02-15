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

from ..connectorgraph import ConnectorGraph
import numpy as np
import tensorflow as tf
from ..model_utils import *


OUTP = '/output:0'
VARS = '/trainable_variables'


def get_connections(graph):
    connections = []
    for edge in graph.edges:
        from_, to = edge
        attrs = graph.edges[edge]
        out = attrs['out']
        in_ = attrs['in']
        if type(out) is list:
            for i in range(len(out)):
                connections.append([from_, to, from_+out[i], to+in_[i]])
        else:
            connections.append([from_, to, from_+out, to+in_])
    return connections


def get_nodes(graph, type_):
    out = []
    for node in graph.nodes:
        try:
            tensors = graph.nodes[node][type_]
        except KeyError:
            pass
        else:
            for tensor in tensors:
                out.append(node+tensor)
    return out


def get_sets(graph, type_):
    out = {}
    for node in graph.nodes:
        try:
            sets = graph.nodes[node][type_]
        except KeyError:
            pass
        else:
            for s in sets:
                try:
                    o = out[s]
                except KeyError:
                    out[s] = []
                    o = out[s]
                o.append(node)
    return out


def node_has_attr(graph, attr):
    out = []
    for node in graph.nodes:
        try:
            _ = graph.nodes[node][attr]
        except KeyError:
            pass
        else:
            out.append(node)
    return out


def convert(graph, config, load_map={}):
    conngraph = ConnectorGraph(config)

    for node in graph:
        if 'loss' in node:
            try:
                config_mod = graph.nodes[node]['config']
            except KeyError:
                config_type = config.lss_type
            else:
                config_type = '_'.join([config.lss_type]+config_mod)
        else:
            try:
                config_mod = graph.nodes[node]['config']
            except KeyError:
                config_type = config.mdl_type
            else:
                config_type = '_'.join([config.mdl_type]+config_mod)
        print node, config_type
        try:
            log_dir, convert_from, log_dir_from = load_map[node]
        except KeyError:
            log_dir, convert_from, log_dir_from = None, None, None
        subgraph = init_subgraph(node, config_type, log_dir, convert_from, log_dir_from)
        conngraph.add_subgraph(subgraph)

    conngraph.print_subgraphs()

    conngraph.quick_connect(get_connections(graph))

    inputs = get_nodes(graph, 'inputs')
    outputs = get_nodes(graph, 'outputs')
    conngraph.block_index = graph.graph['block_index']
    # img_pairs = get_nodes(graph, 'img')
    # img_pairs = [(tensor.split('/')[0], tensor) for tensor in img_pairs]
    # loss_sets = get_sets(graph, 'loss')
    # for loss in loss_sets:
    #     loss_sets[loss] = loss_sets[loss][0]+OUTP
    # train_sets = get_sets(graph, 'train')
    # savers = node_has_attr(graph, 'train')
    # savers = [(node, node+VARS) for node in savers]

    return conngraph, inputs, outputs


def build_variables(conngraph, sess, train_sets):
    variables = {}
    for net in train_sets:
        try:
            var_set = variables[net]
        except KeyError:
            variables[net] = []
            var_set = variables[net]
        for subgraph in train_sets[net]:
            if type(subgraph) is list:
                s_set = []
                for s in subgraph:
                    s_set.extend(tf.get_collection(s+VARS))
                var_set.append(s_set)
            else:
                var_set.extend(tf.get_collection(subgraph+VARS))
    return variables


def branching_build_variables(conngraph, sess, train_sets): # test this
    variables = {}
    for net in train_sets:
        if type(train_sets[net]) is dict:
            variables[net] = {}
            for gloss in train_sets[net]:
                loss_vars = []
                for sub in train_sets[net][gloss]:
                    loss_vars.extend(tf.get_collection(sub+VARS))
                variables[net][gloss] = loss_vars
        else:
            variables[net] = []
            for sub in train_sets[net]:
                variables[net].extend(tf.get_collection(sub+VARS))
    return variables

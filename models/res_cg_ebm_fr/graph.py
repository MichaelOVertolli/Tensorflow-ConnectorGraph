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

from ..graphs.converter import convert
from ..graphs.res_cg_ebm_fr import build
from ..train_funcs.res_cg_ebm_fr import build_train_ops, build_feed_func, build_send_func



def build_graph(config):
    G = build(config)

    conngraph, inputs, outputs = convert(G, config)
    
    conngraph = build_train_ops(conngraph, inputs, outputs, **G.graph)

    get_feed_dict = build_feed_func(**G.graph)
    conngraph.attach_func(get_feed_dict)
    send_outputs = build_send_func(**G.graph)
    conngraph.attach_func(send_outputs)
        
    return conngraph

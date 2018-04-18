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

from models.graphs.converter import *
from models.graphs.res_cg_ebm_mad import *
from models.configs.res_cg_ebm_mad import *
from models.model_utils import strip_index, init_subgraph
from netgen import *
import networkx as nx
from networkx.algorithms.dag import ancestors, descendants

def test(sess):
    name = 'res_gen_pair_08'
    config = 'z128_sz64_h128_g0.7_elu_pconv_wxav_block3'
    log_dir = './logs/NETGEN_celebc_res_cg_ebm_fr_full64_scaled_began_gmsm_b16_z128_sz64_h128_g0.7_elu_pconv_wxav_0314_172152/base_block/res_gen_pair_08/'
    s = BuiltSubGraph(name, config, sess, log_dir)
    
    

    return s

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

from evaluations import write_data
import glob
from models.errors import NANError, ModalCollapseError
from models.graphs.converter import *
from models.graphs.res_cg_ebm_mad import *
from models.configs.res_cg_ebm_mad import *
from models.model_utils import strip_index
from netgen import *
import networkx as nx
from networkx.algorithms.dag import ancestors, descendants
import numpy as np
import os
from random import randint


def test():
    first_write=True
    gamma = 5
    glr = 8e-6 
    dlr = 8e-7
    rlr = 8e-6
    n = NetGen('res_cg_ebm_fr_full32_bw_reg',
               'b16_z128_sz32_h128_gms0.0_chrom0.0_g0.{}_elu_pconv_wxav'.format(gamma),
               'mnist/fr_full32_bw_reg',
               [{'dir': 'base_block', 'data': 'mnist_trainr', 'glr': glr, 'dlr': dlr, 'rlr': rlr, 'fetch_size': 5000, 'resize': None},],
               branching=False)
    n.run()
    event_dir = sorted(glob.glob(os.path.join(n.log_dir, 'base_block', '*')))[0]
    write_data('./logs/mnist/fr_full32_bw_reg', event_dir,
               [('g_lr', glr), ('d_lr', dlr), ('r_lr', rlr), ('gamma', gamma)], first_write=first_write)
    n.close()
    del n

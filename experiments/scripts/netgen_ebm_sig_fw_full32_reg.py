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
    for i in range(100):
        d, e = randint(1, 9), randint(4, 7)
        glr = float('{}e-{}'.format(d, e))
        d1, e1 = randint(1, 9), randint(4, 7)
        while e1 < e or (e1 == e and d1 > d):
            d1, e1 = randint(1, 9), randint(4, 7)
        dlr = float('{}e-{}'.format(d1, e1))
        n = NetGen('res_cg_ebm_sig_fw_full32_reg',
                   'b16_z128_sz32_h128_gms0.0_chrom0.0_g0.5_elu_pconv_wxav',
                   'cifar10/noise/param_search_lr',
                   [{'dir': 'base_block', 'data': 'cifar10r', 'glr': glr, 'dlr': dlr, 'fetch_size': 5000, 'resize': [32, 32]},],
                   branching=False)
        try:
            n.run()
        except (NANError, ModalCollapseError):
            pass
        event_dir = sorted(glob.glob(os.path.join(n.log_dir, 'base_block', '*')))[0]
        print event_dir
        write_data('./logs/cifar10/noise/param_search_lr', event_dir,
                   [('g_lr', glr), ('d_lr', dlr)], first_write=first_write)
        first_write=False
        n.close()
        del n

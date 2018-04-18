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

from models.res_cg_ebm_fr.config import config
from models.graphs.res_cg_ebm_fr import *
from models.train_funcs.res_cg_ebm_fr import *
from models.graphs.converter import convert
import tensorflow as tf


def test():
    c = config('began_b16_z128_sz32_h128_g0.7_elu_pconv_wxav_alphas')
    G = build(c)
    #conn, i, o, loss_sets, train_sets, img_pairs, saver_pairs = convert(g, c)
    conn, i, o = convert(G, c)
    
    #conn = build_train_ops(conn, i, o, 'res_train', loss_sets, train_sets, img_pairs, saver_pairs)
    conn = build_train_ops(conn, i, o, **G.graph)

    get_feed_dict = build_feed_func(**G.graph)#BRGF+OUTP, GENF+INPT, [CNCT+O_IN, LSSD+O_IN], ALIN+ALPH)
    conn.attach_func(get_feed_dict)
    send_outputs = build_send_func(**G.graph)
    conn.attach_func(send_outputs)
    
    return conn #, i, o, loss_sets, train_sets, img_pairs, saver_pairs

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

from evaluations import *
from models.graphs.converter import *
from models.graphs.res_cg_ebm_mad import *
from models.configs.res_cg_ebm_mad import *
from models.model_utils import *
from netgenrunner import *
import networkx as nx
from networkx.algorithms.dag import ancestors, descendants
import tensorflow as tf
geval = tf.contrib.gan.eval


def test():
    model_dir = '/home/olias/Code/BEGAN-tensorflow/logs/mnist/fr_full32_bw_reg/NETGEN_res_cg_ebm_fr_full32_bw_reg_b16_z128_sz32_h128_gms0.0_chrom0.0_g0.9_elu_pconv_wxav_0413_095207'
    nr = NetGenRunner('res_cg_ebm_fr_full32_bw_reg',
                      'b16_z128_sz32_h128_gms0.0_chrom0.0_g0.9_elu_pconv_wxav',
                      '/home/olias/Code/BEGAN-tensorflow/logs/mnist/fr_full32_bw_reg/NETGEN_res_cg_ebm_fr_full32_bw_reg_b16_z128_sz32_h128_gms0.0_chrom0.0_g0.9_elu_pconv_wxav_0413_095207/base_block',
                      [
                         '/home/olias/Code/BEGAN-tensorflow/logs/mnist/fr_full32_bw_reg/NETGEN_res_cg_ebm_fr_full32_bw_reg_b16_z128_sz32_h128_gms0.0_chrom0.0_g0.9_elu_pconv_wxav_0413_095207/final_graph.gpickle',
                      ],
                      branching=False)

    G = nr.Gs.items()[0][1]
    try:
        rev_output = G.graph['rev_output']
    except KeyError:
        rev_output = 'res_rev_00/output:0'
    nr.prep_sess_named(G, './data/mnist_allr/mnist_all.tfrecords', 5000, None, greyscale=True)
    feed_dict = nr.make_feed_dict(**G.graph)
    latents, images, names = nr.reverse_data(feed_dict, G.graph['gen_outputs']['R'], rev_output,
                                             G.graph['rev_input'], G.graph['data_inputs'][0])
    np.save(os.path.join(model_dir, 'images.npy'), images)
    np.save(os.path.join(model_dir, 'latents.npy'), latents)
    with open(os.path.join(model_dir, 'names.pkl'), 'w') as f:
        pickle.dump(names, f)

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

import cv2
from evaluations import *
from models.graphs.converter import *
from models.graphs.res_cg_ebm_mad import *
from models.configs.res_cg_ebm_mad import *
from models.model_utils import *
from netgenrunner import *
import networkx as nx
from networkx.algorithms.dag import ancestors, descendants
import pickle
import tensorflow as tf
geval = tf.contrib.gan.eval




def test():
    model_dir = '/home/olias/Code/BEGAN-tensorflow/logs/mnist/fr_full32_bw_reg/NETGEN_res_cg_ebm_fr_full32_bw_reg_b16_z128_sz32_h128_gms0.0_chrom0.0_g0.9_elu_pconv_wxav_0413_095207'
    images = np.squeeze(np.load(os.path.join(model_dir, 'images.npy')))
    with open(os.path.join(model_dir, 'names.pkl'), 'r') as f:
        names = pickle.load(f)
    log_dir = '/home/olias/data/mnist_avgdists'
    with open(os.path.join(model_dir, 'alldists_netgen_latents.pkl'), 'r') as f:
        alldists = pickle.load(f)
    centers = np.load(os.path.join(log_dir, 'tsne_centers.npy'))
    new_img_sz = [16000, 16000]
    size = 32
    cnt = 500
    sideln = 100
    centers = np.int64(centers * (cnt - (2*sideln))) + sideln
    pts = get_points(cnt, cnt)
    clusters, next_imdists = get_cluster_data(centers, alldists, pts)
    canvas = np.ones(new_img_sz, np.uint8)*255
    for k in clusters:
        data = clusters[k]
        canvas = add_img_bw(canvas, data['c_img'], data['c_pt']*size)
    for i in xrange(int(1e10)):
        try:
            _, name, im, k = next_imdists.pop(0)
        except IndexError:
            break # no more images to do
        data = clusters[k]
        try:
            pt, _ = data['ptdists'].popitem(False)
        except IndexError:
            break # no more points on the canvas left
        # draw
        # canvas = add_img_bw(canvas, images[names[name]], np.int32(pt)*size)
        canvas = add_img_bw(canvas, im, np.int32(pt)*size)
        # add next
        try:
            imd = nxt_imdist(k, data['imdists'])
        except IndexError:
            pass # there will be data remaining from other clusters or we're done
        else:
            next_imdists.append(imd)
            next_imdists = sorted(next_imdists, key=lambda v: v[0])
        # cleanup
        for k2 in clusters:
            if k2 == k:
                continue
            del clusters[k2]['ptdists'][pt]
    fname = 'entourage_all_{}.jpg'.format(sideln)
    im = Image.fromarray(np.uint8(canvas))
    im.save(os.path.join(model_dir, fname))
    # return centers, pts

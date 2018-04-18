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



def test():
    model_dir = '/home/olias/Code/BEGAN-tensorflow/logs/mnist/fr_full32_bw_reg/NETGEN_res_cg_ebm_fr_full32_bw_reg_b16_z128_sz32_h128_gms0.0_chrom0.0_g0.9_elu_pconv_wxav_0413_095207'
    images = np.squeeze(np.load(os.path.join(model_dir, 'images.npy')))
    with open(os.path.join(model_dir, 'names.pkl'), 'r') as f:
        names = pickle.load(f)
    with open(os.path.join(model_dir, 'alldists_netgen_latents.pkl'), 'r') as f:
        prev_dists = pickle.load(f)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=4, nOctaveLayers=1, edgeThreshold = 25, sigma=0.5)
    log_dir = '/home/olias/data/mnist_all'
    size = None
    grey = True
    alpha = None
    pdist = None
    splits = split_imgs(log_dir)
    sift_sq = sift_dist_sq(8)
    sift_sq.next()
    all_dists = {}
    for k in splits.keys():
        print 'Starting label {}.'.format(k)
        files = splits[k]
        c_i, _ = prev_dists[k]
        c_i = np.array(c_i, np.uint8)
        imgs = np.zeros([len(files)]+list(images.shape[1:]), np.uint8)
        for i, f in enumerate(files):
            imgs[i] = images[names[f]]
        dists = emd_to_one(c_i, None, files, imgs, None, sift_sq, pdist, alpha, sift)
        all_dists[k] = (c_i, dists)
    return splits, images, all_dists

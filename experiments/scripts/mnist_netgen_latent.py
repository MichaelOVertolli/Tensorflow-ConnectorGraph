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
    center_i = [1, 25, 65, 3, 12, 35, 0, 10, 130, 4]
    model_dir = '/home/olias/Code/BEGAN-tensorflow/logs/mnist/fr_full32_bw_reg/NETGEN_res_cg_ebm_fr_full32_bw_reg_b16_z128_sz32_h128_gms0.0_chrom0.0_g0.9_elu_pconv_wxav_0413_095207'
    images = np.squeeze(np.load(os.path.join(model_dir, 'images.npy')))
    latents = np.load(os.path.join(model_dir, 'latents.npy'))
    with open(os.path.join(model_dir, 'names.pkl'), 'r') as f:
        names = pickle.load(f)
    centers = [images[i] for i in center_i]
    clatents = [latents[i] for i in center_i]
    log_dir = '/home/olias/data/mnist_all'
    size = None
    grey = True
    alpha = None
    pdist = None
    splits = split_imgs(log_dir)
    all_dists = {}
    for k in splits.keys():
        print 'Starting label {}.'.format(k)
        files = splits[k]
        c_i = centers[int(k)]
        lt_i = clatents[int(k)]
        imgs = np.zeros([len(files)]+list(images.shape[1:]), np.uint8)
        lts = np.zeros([len(files), latents.shape[1]])
        for i, f in enumerate(files):
            imgs[i] = images[names[f]]
            lts[i] = latents[names[f]]
        dists = latent_to_one(c_i, lt_i, files, imgs, lts)
        all_dists[k] = (c_i, dists)
    return splits, images, all_dists

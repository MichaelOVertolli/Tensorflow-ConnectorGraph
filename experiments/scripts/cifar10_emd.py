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
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=4, nOctaveLayers=1, edgeThreshold = 50, sigma=0.5)
    log_dir = '/home/olias/data/cifar10_all'
    size = None
    alpha = 0.1
    pdist = pos_dist_mat(32)
    pdist.next()
    splits = split_imgs(log_dir)
    sift_sq = sift_dist_sq(8)
    sift_sq.next()
    all_dists = {}
    with open('/home/olias/data/cifar10_dists/all_cdists.pkl', 'r') as f:
        c_imgs = dict(pickle.load(f))
    for k in splits.keys():
        print 'Starting label {}.'.format(k)
        files = splits[k]
        c_i = files.index(c_imgs[str(k)])
        imgs = load_imgs(log_dir, files, False, size)
        gimgs = load_imgs(log_dir, files, True, size)
        dists = emd_to_one(imgs[c_i], gimgs[c_i], files, imgs, gimgs, sift_sq, pdist, alpha, sift)
        all_dists[k] = (imgs[c_i], dists)
    #     
    # im1 = Image.open(f1)
    # im2 = Image.open(f2)
    # size = [32, 32]
    # im1 = im1.resize(size, Image.BICUBIC)
    # im2 = im2.resize(size, Image.BICUBIC)
    # im1g = np.array(im1.convert('L'))
    # im2g = np.array(im2.convert('L'))
    # im1 = np.array(im1)
    # im2 = np.array(im2)
    # im1 = rgb2lab(im1 / 255.)
    # im2 = rgb2lab(im2 / 255.)
    # sift_sq = sift_dist_sq(8)
    # sift_sq.next()
    # pdist = pos_dist_mat(32)
    # pdist.next()
    # k, d = compute_sift(im1g)
    # k2, d2 = compute_sift(im2g)
    # pdist = binpos_dist_mat(d, d2)
    # odist = binorient_dist_mat(d, d2, 8)
    # pdist2, odist2 = sift_dist_sq(8)
    # siftemd = compute_emd(im1g, im2g, None, sift_sq)
    # print 'sift done'
    # coloremd = compute_emd(im1, im2, None, pdist=pdist)
    return splits, imgs, all_dists

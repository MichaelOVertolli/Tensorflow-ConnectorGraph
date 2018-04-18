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


def test():
    f1 = '/home/olias/data/000001.jpg'
    f2 = '/home/olias/data/000002.jpg'
    im1 = Image.open(f1)
    im2 = Image.open(f2)
    size = [32, 32]
    im1 = im1.resize(size, Image.BICUBIC)
    im2 = im2.resize(size, Image.BICUBIC)
    im1g = np.array(im1.convert('L'))
    im2g = np.array(im2.convert('L'))
    im1 = np.array(im1)
    im2 = np.array(im2)
    im1 = rgb2lab(im1 / 255.)
    im2 = rgb2lab(im2 / 255.)
    sift_sq = sift_dist_sq(8)
    sift_sq.next()
    pdist = pos_dist_mat(32)
    pdist.next()
    # k, d = compute_sift(im1g)
    # k2, d2 = compute_sift(im2g)
    # pdist = binpos_dist_mat(d, d2)
    # odist = binorient_dist_mat(d, d2, 8)
    # pdist2, odist2 = sift_dist_sq(8)
    siftemd = compute_emd(im1g, im2g, None, sift_sq)
    print 'sift done'
    coloremd = compute_emd(im1, im2, None, pdist=pdist)
    return im1, im2, im1g, im2g, siftemd, coloremd

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
import pickle



def test():
    log_dir = '/home/olias/data/mnist_avgdists'
    with open(os.path.join(log_dir, 'alldists.pkl'), 'r') as f:
        alldists = pickle.load(f)
    centers = np.load(os.path.join(log_dir, 'tsne_centers.npy'))
    new_img_sz = [16000, 16000]
    size = 32
    cnt = 500
    sideln = 75
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
    im.save(os.path.join(log_dir, fname))
    # return centers, pts

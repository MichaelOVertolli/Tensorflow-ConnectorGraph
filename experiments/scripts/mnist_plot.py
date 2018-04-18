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
    fname = 'alldists.pkl'
    with open(os.path.join(log_dir, fname), 'r') as f:
        alldists = pickle.load(f)
    new_img_sz = [5000, 5000]
    size = 32
    cnt = 150
    fname = 'entourage_{}.jpg'
    for k in alldists.keys():
        print 'Starting {} label.'.format(k)
        cim, imdists = alldists[k]
        imdists = sorted(imdists, key=lambda v: v[0])
        cpt, ptdists = get_center_dists(get_points(cnt, cnt))
        print cpt
        canvas = np.ones(new_img_sz, np.uint8)*255
        canvas = add_img_bw(canvas, cim, cpt*size)
        while True:
            try:
                dist, name, im = imdists.pop(0)
            except IndexError:
                break
            _, pt = ptdists.pop(0)
            canvas = add_img_bw(canvas, im, pt*size)
        im = Image.fromarray(np.uint8(canvas))
        im.save(os.path.join(log_dir, fname.format(k)))

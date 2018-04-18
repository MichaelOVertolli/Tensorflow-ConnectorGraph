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
    im1 = img_as_ubyte(np.array(im1.resize(size, Image.BICUBIC)))
    im2 = img_as_ubyte(np.array(im2.resize(size, Image.BICUBIC)))
    im1 = flatten2color(rgb2lab(im1 / 255.))
    im2 = flatten2color(rgb2lab(im2 / 255.))

    gdist = color_gdist_mat(im1, im2)
    return im1, im2, gdist

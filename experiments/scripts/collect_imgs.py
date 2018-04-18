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
    model_log_dir = '/home/olias/data/params/mnist/param_search_lr/NETGEN_res_cg_ebm_fw_full32_bw_reg_b16_z128_sz32_h128_gms0.0_chrom0.0_g0.5_elu_pconv_wxav_0320_211607/base_block/0320_211607'
    log_dir = '/home/olias/data/params/mnist/grow/param_search_lr_grid_step2'
    block = 'block1'
    gamma = 0.9
    data, basepath = get_img_data(log_dir, block, gamma_filter=[gamma])
    new_size = [8000, 8000, 3]
    size = 16
    pxtrim = 100
    index = index_imgs(new_size, pxtrim, data)
    # join = join_imgs([p[1] for p in index.values()[0]], 2, size)
    grid = grid_spacing().flatten()[8:]
    img = draw_grid(np.ones(new_size)*255, grid, pxtrim, dp=1)
    img = plot_imgs(new_size, index, 4, size, np.flip(img, 0))
    
    im = Image.fromarray(np.uint8(img))
    im.save(os.path.join(log_dir, 'params_vis{}.jpg'.format(gamma)))
    # aggr_imgs(log_dir, block)
    # return index, grid

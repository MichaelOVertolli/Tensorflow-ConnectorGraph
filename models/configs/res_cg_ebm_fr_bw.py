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

from .. import config_base as cb
from ..errors import ConfigError
import numpy as np

#batch_size
B_16 = 'b16'
#gamma
G_5 = 'g0.5'
G_7 = 'g0.7'
#lss_type
BEGAN = 'began'
BEGAN_GMSM = 'began_gmsm'
BEGAN_GMSM_CHROM = 'began_gmsm_chrom'
SCALED_BEGAN_GMSM = 'scaled_began_gmsm'
SCALED_BEGAN_GMSM_CHROM = 'scaled_began_gmsm_chrom'
SCALED_BEGAN_GMSM_HALFCHROM = 'scaled_began_gmsm_halfchrom'
#mdl_type
Z128_SZ32 = 'z128_sz32'
Z256_SZ32 = 'z256_sz32'
Z128_SZ64 = 'z128_sz64'
Z256_SZ64 = 'z256_sz64'
Z512_SZ64 = 'z512_sz64'
Z1024_SZ64 = 'z1024_sz64'
Z2048_SZ64 = 'z2048_sz64'
Z256_SZ128 = 'z256_sz128'
Z512_SZ64 = 'z512_sz64'
Z512_SZ256 = 'z512_sz256'
Z512_SZ512 = 'z512_sz512'
Z512_SZ1024 = 'z512_sz1024'
H64 = 'h64'
H128 = 'h128'
H256 = 'h256'
H512 = 'h512'
PROJ = 'project'
ELU_ = 'elu'
RELU = 'relu'
LRLU = 'leaky_relu'
PCNV = 'pconv'
PAVG = 'pavg'
WXAV = 'wxav'
WHE_ = 'whe'
NPIX = 'npixel'
NBAT = 'nbatch'
MINI = 'minibatch'
ALPH = 'alphas'


def config(type_):
    config = cb.Config()


    if B_16 in type_:
        config.batch_size = 16
    else:
        raise ConfigError('Invalid config type {} for batch_size.'.format(type_))
    if G_5 in type_:
        config.gamma = 0.5
    elif G_7 in type_:
        config.gamma = 0.7
    else:
        raise ConfigError('Invalid config type {} for gamma.'.format(type_))
    if SCALED_BEGAN_GMSM_HALFCHROM in type_:
        config.lss_type = SCALED_BEGAN_GMSM_HALFCHROM
    elif SCALED_BEGAN_GMSM_CHROM in type_:
        config.lss_type = SCALED_BEGAN_GMSM_CHROM
    elif SCALED_BEGAN_GMSM in type_:
        config.lss_type = SCALED_BEGAN_GMSM
    elif BEGAN_GMSM_CHROM in type_:
        config.lss_type = BEGAN_GMSM_CHROM
    elif BEGAN_GMSM in type_:
        config.lss_type = BEGAN_GMSM
    elif BEGAN in type_:
        config.lss_type = BEGAN
    else:
        raise ConfigError('Invalid config type {} for lss_type.'.format(type_))
    if Z128_SZ32 in type_:
        config.mdl_type = Z128_SZ32
        config.z_num = 128
        config.img_size = 32
    elif Z256_SZ32 in type_:
        config.mdl_type = Z256_SZ32
        config.z_num = 256
        config.img_size = 32
    elif Z128_SZ64 in type_:
        config.mdl_type = Z128_SZ64
        config.z_num = 128
        config.img_size = 64
    elif Z256_SZ64 in type_:
        config.mdl_type = Z256_SZ64
        config.z_num = 256
        config.img_size = 64
    elif Z512_SZ64 in type_:
        config.mdl_type = Z512_SZ64
        config.z_num = 512
        config.img_size = 64
    elif Z1024_SZ64 in type_:
        config.mdl_type = Z1024_SZ64
        config.z_num = 1024
        config.img_size = 64
    elif Z2048_SZ64 in type_:
        config.mdl_type = Z2048_SZ64
        config.z_num = 2048
        config.img_size = 64
    elif Z256_SZ128 in type_:
        config.mdl_type = Z256_SZ128
        config.z_num = 256
        config.img_size = 128
    elif Z512_SZ64 in type_:
        config.mdl_type = Z512_SZ64
        config.z_num = 512
        config.img_size = 64
    elif Z512_SZ256 in type_:
        config.mdl_type = Z512_SZ256
        config.z_num = 512
        config.img_size = 256
    elif Z512_SZ512 in type_:
        config.mdl_type = Z512_SZ512
        config.z_num = 512
        config.img_size = 512
    elif Z512_SZ1024 in type_:
        config.mdl_type = Z512_SZ1024
        config.z_num = 512
        config.img_size = 1024
    else:
        raise ConfigError('Invalid znum or img_size for type: {}.'.format(type_))
    if H64 in type_:
        config.mdl_type = '_'.join([config.mdl_type, H64])
    elif H128 in type_:
        config.mdl_type = '_'.join([config.mdl_type, H128])
    elif H256 in type_:
        config.mdl_type = '_'.join([config.mdl_type, H256])
    elif H512 in type_:
        config.mdl_type = '_'.join([config.mdl_type, H512])
    else:
        raise ConfigError('Invalid hidden_num for type: {}.'.format(type_))
    if LRLU in type_:
        config.mdl_type = '_'.join([config.mdl_type, LRLU])
    elif RELU in type_:
        config.mdl_type = '_'.join([config.mdl_type, RELU])
    elif ELU_ in type_:
        config.mdl_type = '_'.join([config.mdl_type, ELU_])
    else:
        raise ConfigError('Invalid activation_fn in type: {}.'.format(type_))
    if PCNV in type_:
        config.mdl_type = '_'.join([config.mdl_type, PCNV])
    elif PAVG in type_:
        config.mdl_type = '_'.join([config.mdl_type, PAVG])
    else:
        raise ConfigError('Invalid pool_type in type: {}.'.format(type_))
    if WXAV in type_:
        config.mdl_type = '_'.join([config.mdl_type, WXAV])
    elif WHE_ in type_:
        config.mdl_type = '_'.join([config.mdl_type, WHE_])
    else:
        raise ConfigError('Invalid weights_init in type: {}.'.format(type_))
    if NPIX in type_:
        config.mdl_type = '_'.join([config.mdl_type, NPIX])
    elif NBAT in type_:
        config.mdl_type = '_'.join([config.mdl_type, NBAT])
    if ALPH in type_:
        config.alphas = True
        config.mdl_type = '_'.join([config.mdl_type, ALPH])
    else:
        config.alphas = False

    config.repeat_num = int(np.log2(config.img_size)) - 1
    config.base_size = 4
    config.g_lr = 2e-5
    config.d_lr = 4e-6
    config.r_lr = 8e-5
    config.lr_lower_boundary = 1e-6
    config.lambda_k = 0.001
    config.alpha_update_steps = 20000

    config.data_format = 'NCHW'

    config.name = type_

    return config

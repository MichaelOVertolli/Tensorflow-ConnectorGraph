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


#Types:
IMPR = 'improved'
NVDG = 'nvd_grad'

def config(type_):
    config = cb.Config()

    if IMPR in type_:
        config.improved = True
    else:
        config.improved = False
    if NVDG in type_:
        config.nvd_grad = True
    else:
        config.nvd_grad = False

    config.iwass_lambda = 0.1
    config.iwass_epsilon = 0.001
    config.iwass_target = 1.0
    config.z_num = 128
    
    config.name = type_

    return config

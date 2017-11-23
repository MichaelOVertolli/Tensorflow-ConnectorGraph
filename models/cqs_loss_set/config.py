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
BEGAN = 'began'
SCALED_BEGAN_GMSM = 'scaled_began_gmsm'


def config(type_):
    config = cb.Config()

    if type_ == BEGAN:
        config.l1weight = 1.0
        config.gmsweight = 0.0
        config.chromweight = 0.0
    elif type_ == SCALED_BEGAN_GMSM:
        config.l1weight = 2.0
        config.gmsweight = 1.0
        config.chromweight = 0.0
    else:
        raise ConfigError('Invalid config type: {}.'.format(type_))

    config.totalweight = config.l1weight + \
                         config.gmsweight + \
                         config.chromweight

    config.name = type_

    return config

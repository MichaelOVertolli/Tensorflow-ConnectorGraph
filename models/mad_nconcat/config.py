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
import re

CNT_ = 'count'

def config(type_):
    config = cb.Config()

    config.name = ''

    m = re.search('(?<={})\d+'.format(CNT_), type_)
    if m is not None:
        config.count = int(m.group(0))
    else:
        raise ConfigError('Invalid count in type: {}.'.format(type_))

    return config

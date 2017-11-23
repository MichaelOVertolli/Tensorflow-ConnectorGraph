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

from models.config_base import Config
from models.errors import ConfigError


def config():
    config = Config()

    config.use_gpu = True

    config.data_format = 'NCHW'

    config.max_step = 280000
    config.start_step = 0
    config.save_step = 5000
    config.log_step = 50
    config.lr_update_step = 100000

    return config


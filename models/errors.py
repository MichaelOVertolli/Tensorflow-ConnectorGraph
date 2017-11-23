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

class Error(Exception):
    pass


class ConfigError(Exception):
    pass


class ConnectionError(Error):
    pass


class MissingSubgraphError(ConnectionError):
    pass


class MissingTensorError(ConnectionError):
    pass


class MissingConnectionError(ConnectionError):
    pass


class ConnectionConflictError(ConnectionError):
    pass


class ExhaustedGraphStepsError(ConnectionError):
    pass


class GraphNotConnectedError(ConnectionError):
    pass


class NoVariableExistsError(ConnectionError):
    pass


class MultipleVariablesExistError(ConnectionError):
    pass


class NoSaversError(ConnectionError):
    pass


class SessionGraphError(Error):
    pass


class SubGraphError(Error):
    pass


class FirstInitialization(SubGraphError):
    pass


class InvalidSubGraphError(SubGraphError):
    pass

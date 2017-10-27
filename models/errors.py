

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

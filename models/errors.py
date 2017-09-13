

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


class SessionGraphError(Error):
    pass

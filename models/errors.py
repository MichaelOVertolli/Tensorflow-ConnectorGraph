

class Error(Exception):
    pass


class ConnectionError(Error):
    pass


class MissingSubgraphError(ConnectionError):
    pass


class MissingTensorError(ConnectionError):
    pass


class MissingConnectionError(ConnectionError):
    pass

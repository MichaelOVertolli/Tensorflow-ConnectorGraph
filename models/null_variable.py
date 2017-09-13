import tensorflow as tf


def make_null_variable():
    """Returns a Variable that does nothing.

    This variable is used as a hack to build subgraphs
    that have no trainable variables in them. For example,
    evaluation and optimization subgraphs.

    """
    null = tf.Variable(0, False, None, False, name='null')

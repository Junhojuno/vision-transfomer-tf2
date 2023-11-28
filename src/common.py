import tensorflow as tf
from tensorflow.keras import layers


class StochasticDepth(layers.Layer):
    """Creates a stochastic depth layer."""

    def __init__(self, stochastic_depth_drop_rate, **kwargs):
        """Initializes a stochastic depth layer.

        Args:
            stochastic_depth_drop_rate: A `float` of drop rate.
            **kwargs: Additional keyword arguments to be passed.

        Returns:
            A output `tf.Tensor` of which should have the same shape as input.
        """
        super(StochasticDepth, self).__init__(**kwargs)
        self._drop_rate = stochastic_depth_drop_rate

    def get_config(self):
        config = {'stochastic_depth_drop_rate': self._drop_rate}
        base_config = super(StochasticDepth, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        if not training or self._drop_rate is None or self._drop_rate == 0:
            return inputs

        keep_prob = 1.0 - self._drop_rate
        batch_size = tf.shape(inputs)[0]
        random_tensor = keep_prob
        random_tensor += tf.random.uniform(
            [batch_size] + [1] * (inputs.shape.rank - 1), dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.math.divide(inputs, keep_prob) * binary_tensor
        return output

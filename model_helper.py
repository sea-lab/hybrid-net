import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops


class MaskStealingLayer(tf.keras.layers.Masking):
    def compute_mask(self, inputs, mask=None):
        return super(tf.keras.layers.Masking, self).compute_mask(self.inputs[1], mask)
    
    def call(self, inputs):
        signals, mask = inputs
        boolean_mask = tf.keras.backend.any(
            math_ops.not_equal(inputs[1], self.mask_value), axis=-1, keepdims=True)
        # outputs = inputs[0] * math_ops.cast(boolean_mask, inputs[0].dtype)
        outputs = signals * mask
        # Compute the mask and outputs simultaneously.
        outputs._keras_mask = array_ops.squeeze(boolean_mask, axis=-1)  # pylint: disable=protected-access
        return outputs

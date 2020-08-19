import tensorflow as tf
from tensorflow.python.ops import math_ops, array_ops
import numpy


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


def get_model_input_output_layers(inputs, outputs, input_length, n_states, convs, grus, skip_denses=0):
    n_in = len(inputs)
    n_out = len(outputs)
    n_features = n_in + n_out

    bias_initializer = tf.keras.initializers.Constant(numpy.log(0.01))

    signals = tf.keras.Input(shape=[input_length, n_features], name='signals')
    mask = tf.keras.Input(shape=[input_length, 1], name='mask')

    x = signals
    x = MaskStealingLayer(0)((x, mask))

    for filters, conv_size in convs:
        x = tf.keras.layers.Conv1D(filters=filters, kernel_size=conv_size, padding="same", name=f'conv_{conv_size}')(x)

    for gru_size in grus:
        x = tf.keras.layers.GRU(gru_size, return_sequences=True)(x)
        # x = tf.keras.layers.LeakyReLU()(x)

    if skip_denses >= 3:
        raise ValueError('There are only 2 dense layers in the end!')

    if not skip_denses >= 2:
        x = tf.keras.layers.Dense(grus[-1])(x)
        x = tf.keras.layers.LeakyReLU()(x)

    if not skip_denses >= 1:
        x = tf.keras.layers.Dense(n_states, bias_initializer=bias_initializer, activation='softmax')(x)

    # x = tf.keras.layers.UpSampling1D(2 ** 1)(x)

    return [signals, mask], x


def make_model(*args, **kwargs):
    input_layer, output_layer = get_model_input_output_layers(*args, **kwargs)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

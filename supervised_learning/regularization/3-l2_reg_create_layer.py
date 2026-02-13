#!/usr/bin/env python3
"""
3-l2_reg_create_layer module

Contains the function l2_reg_create_layer that creates
a TensorFlow layer with L2 regularization.
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a fully connected layer with L2 regularization.

    Parameters:
    - prev: tf.Tensor, output of the previous layer
    - n: int, number of nodes in the new layer
    - activation: activation function (e.g., tf.nn.tanh)
    - lambtha: float, L2 regularization parameter

    Returns:
    - tf.Tensor, output of the new layer
    """
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.L2(l2=lambtha)
    )(prev)
    return layer

#!/usr/bin/env python3
"""
3-l2_reg_create_layer.py

Contains:
- l2_reg_create_layer: creates a Dense layer with L2 regularization
- l2_reg_cost: computes total cost per layer including L2
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
    )
    return layer(prev)


def l2_reg_cost(cost, model):
    """
    Computes total cost including L2 penalties for each layer.

    Parameters:
    - cost: tf.Tensor, base loss without regularization
    - model: Keras model with L2 regularization applied to layers

    Returns:
    - tf.Tensor with total cost per layer (excluding input layer)
    """
    l2_costs = []

    for layer in model.layers:
        # Only layers with kernel_regularizer count
        if hasattr(layer, "kernel_regularizer") and layer.kernel_regularizer:
            l2_penalty = layer.kernel_regularizer(layer.kernel)
            l2_costs.append(cost + l2_penalty)

    # Stack all costs into a single tensor
    return tf.stack(l2_costs)

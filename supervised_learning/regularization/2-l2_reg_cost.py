#!/usr/bin/env python3
"""
2-l2_reg_cost module

Contains the function l2_reg_cost that calculates
the cost of a Keras neural network with L2 regularization.
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Computes the total cost of a neural network with L2 regularization.

    Parameters:
    - cost: tf.Tensor containing the cost without regularization
    - model: Keras model with layers that include L2 regularization

    Returns:
    - tf.Tensor containing the total cost for each layer with L2
    """
    l2_costs = []

    # Only consider layers that have weights (skip input layer)
    for layer in model.layers:
        if hasattr(layer, "kernel_regularizer") and layer.kernel_regularizer:
            # L2 penalty for this layer
            l2_penalty = layer.kernel_regularizer(layer.kernel)
            # Add to base cost
            l2_costs.append(cost + l2_penalty)

    # Return as a tensor
    return tf.stack(l2_costs)

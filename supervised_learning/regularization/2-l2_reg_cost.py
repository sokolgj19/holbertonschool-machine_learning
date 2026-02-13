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
    - tf.Tensor containing the total cost for each layer, including L2
    """
    l2_costs = []

    for layer in model.layers:
        # Check if layer has kernel_regularizer (weights)
        if hasattr(layer, "kernel_regularizer") and layer.kernel_regularizer:
            # Add L2 penalty for this layer
            l2_penalty = layer.kernel_regularizer(layer.kernel)
            l2_costs.append(l2_penalty)
        else:
            # No regularization for this layer
            l2_costs.append(tf.constant(0.0, dtype=cost.dtype))

    # Convert list to tensor and add base cost
    total_cost = tf.stack([cost + l2 for l2 in l2_costs])
    return total_cost

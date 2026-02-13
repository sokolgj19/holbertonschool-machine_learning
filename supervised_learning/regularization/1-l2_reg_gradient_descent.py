#!/usr/bin/env python3
"""
1-l2_reg_gradient_descent module

Contains the function l2_reg_gradient_descent that updates
the weights and biases of a neural network using gradient
descent with L2 regularization.
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network
    using gradient descent with L2 regularization.

    Parameters:
    - Y: one-hot numpy.ndarray of shape (classes, m)
    - weights: dictionary of weights and biases
    - cache: dictionary of layer outputs
    - alpha: learning rate
    - lambtha: L2 regularization parameter
    - L: number of layers in the network

    Updates weights in place.
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y  # Last layer gradient

    for layer in range(L, 0, -1):
        A_prev = cache["A" + str(layer - 1)]
        W = weights["W" + str(layer)]

        # Gradient with respect to weights and bias
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update weights and bias in place
        weights["W" + str(layer)] = W - alpha * dW
        weights["b" + str(layer)] = weights["b" + str(layer)] - alpha * db

        if layer > 1:
            dZ = np.matmul(weights["W" + str(layer)].T, dZ) * (1 - A_prev ** 2)

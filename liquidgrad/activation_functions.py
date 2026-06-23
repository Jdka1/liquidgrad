"""
TODO: Add docs.
"""
import numpy as np


def tanh(z):
    """Squashes any number into (-1, 1). Smooth, centered at zero."""
    return np.tanh(z)


def sigmoid(z):
    """Squashes any number into (0, 1). Useful when you want a 0..1 output."""
    return 1.0 / (1.0 + np.exp(-z))
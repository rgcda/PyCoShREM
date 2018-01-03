def mse(a, b):
    """Return the elementwise mean squared error of two numpy arrays."""
    return ((a - b) ** 2).mean()

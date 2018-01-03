import numpy as np
from itertools import product


def curvature(orientations):
    """
    Measure curvature on an array of element-wise orientation measurement.

    Example:
    Set up a shearlet system for edge detection on the Lena picture
    >>> i = Image.open("../tests/resources/img/lena.jpg").convert("L")
    >>> img = np.asarray(i)
    >>> edge_sys = EdgeSystem(*img.shape)
    >>> edges, edge_orientations = edge_sys.detect(img, min_contrast=70, pivoting_scales='lowest')

    Apply thinning on the edge measurements:
    >>> thinned_edges = mask(edges, thin_mask(edges))
    >>> thinned_edge_orientations = mask(edge_orientations, thin_mask(edges))

    Calculate curvature:
    >>> edge_curv = curvature(thinned_edge_orientations)

    Args:
        orientations: Thinned orientation measurement as returned by EdgeSystem.detect or RidgeSystem.detect

    Returns:
        array with local curvature measurements

    """

    curv = np.zeros_like(orientations)
    offsets = [(a, b) for a, b in product(list(range(-1, 2)), list(range(-1, 2))) if not (a == b) & (b == 0)]
    for row in range(1, orientations.shape[0] - 1):
        for col in range(1, orientations.shape[1] - 1):
            if orientations[row, col] >= 0:
                count = 0
                left = 0
                right = 0
                for dr, dc in offsets:
                    if orientations[row + dr, col + dc] >= 0:
                        if count > 0:
                            left = orientations[row + dr, col + dc]
                        else:
                            right = orientations[row + dr, col + dc]
                        count += 1

                d_or_left = _scale_angle(orientations[row, col] - left)
                d_or_right = _scale_angle(right - orientations[row, col])

                curv[row, col] = abs(d_or_left + d_or_right)/2 if (count == 2) else -1
            else:
                curv[row,col] = -1
    return curv


def _scale_angle(angle):
    """Scales a single angle to a value between 0 and 180"""
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180
    return angle


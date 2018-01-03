import numpy as np


def cone_orientation(level):
    """
    Compute shear-indexes for the vertical and horizontal cones of a shearlet system for a specific shear-level.
    Args:
        level: Shear Level
    """
    ks = np.zeros(2 ** level + 2, dtype=np.int_)

    new_oris = np.arange(2 ** max(level - 2, 0) + 1)
    ks[new_oris] = new_oris
    conev = new_oris + 1

    new_oris = np.arange(conev[-1], conev[-1] + 2 ** (level - 1) + 1, dtype=np.int_)
    ks[new_oris] = (-1 * new_oris) + new_oris[len(new_oris) // 2]
    coneh = new_oris + 1

    new_oris = np.arange(coneh[-1], (2 ** level + 2))

    if new_oris.size:
        ks[new_oris] = new_oris - (new_oris[-1] + 1)
        conev = np.concatenate((conev, new_oris + 1))

    return conev, coneh, ks

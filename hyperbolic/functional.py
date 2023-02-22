import numpy as np
import torch



def mobius_transform(p, k):
    """
        Returns an hyperbolic isometry that sends p to the origin
    """
    assert(k < 0)
    p = p if isinstance(p, complex) else complex(*p)
    
    r = 1 / np.sqrt(-k)
    transform = lambda z : (z - p) / (1 - np.conj(p/r) * z/r)
    return transform


def mobius_inverse(p, k):
    """
        Inverse hyperbolic isometry of the Mobius transform
        Maps the origin back to p
    """
    assert(k < 0)
    p = p if isinstance(q, complex) else complex(*p)

    r = 1 / np.sqrt(-k)
    transform = lambda w : (w + p) / (1 + np.conj(p/r) * w/r)
    return transform


import numpy as np
import functools
import itertools


def get_poly(p, deg=2):
    if isinstance(deg, int):
        if deg == 0:
            return 1
        elif deg == 1:
            return p
        else:
            return np.array([
                functools.reduce(lambda a, b: a * b, tup)
                for tup in itertools.combinations_with_replacement(p, deg)
            ])
    elif isinstance(deg, list):
        return np.hstack([get_poly(p, deg=d) for d in deg])
    else:
        raise ValueError("deg should be an integer or a list of integers.")

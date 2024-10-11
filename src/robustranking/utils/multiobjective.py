import copy

import numpy as np

# HV

# NDS


def dominates(x, y):
    r"""
    Note: assumes minimising.

    Args:
        x: list of objective vector
        y: list of objective vector

    Returns: Bool which says if x dominates y (x \prec y)

    """
    return np.count_nonzero(x > y) == 0 and np.count_nonzero(x < y) > 0


def incomparable(x, y):
    """When x and y neither dominate each other."""
    return not (dominates(x, y) or dominates(y, x))


def fast_non_dominated_sorting(points):
    """
    Taken from NSGA-II paper.

    TODO cite paper and algorithm number

    Args:
        points: 2d array. shape=(points, objectives)

    Returns:
        front: list of list of the ranks
        dominates_points: list with for each point the points it dominates
        dominated_count: list containting the number of points that dominate a point
        ranks:  list containing the rank per point
    """
    points = np.array(points)
    n_points = points.shape[0]

    n = np.zeros(n_points, dtype=int)

    dominates_points = {p: set() for p in range(n_points)}  # Dominates
    front = {1: set()}  # Fronts
    ranks = np.zeros(n_points, dtype=int)  # Point ranks
    for p in range(n_points):
        # TODO optimize by using np.where
        for q in range(n_points):
            if p == q:
                continue
            if dominates(points[p, :], points[q, :]):
                dominates_points[p].add(q)
            elif dominates(points[q, :], points[p, :]):
                n[p] += 1
        if n[p] == 0:
            front[1].add(p)
    dominated_count = copy.copy(n)
    i = 1
    while len(front[i]) != 0:
        Q = set()
        for p in front[i]:
            for q in dominates_points[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    ranks[q] = i
                    Q.add(q)
        i += 1
        front[i] = Q

    del front[i]

    front = [np.array(list(f)) for _, f in front.items()]
    dominates_points = [np.array(list(s)) for _, s in dominates_points.items()]

    return front, dominates_points, dominated_count, ranks

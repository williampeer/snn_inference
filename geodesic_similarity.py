import numpy as np

from nmf_results import *


def geodesic_similarity(m1, m2):
    return 1. - 2./ np.pi * np.arccos(np.inner(m1, m2))


def brute_force_max_geodesic_similarity(m1, m2):
    assert m1.shape[0] == 3 and m1.shape[1] == 12, "shape expected to be (3,12)"
    assert m2.shape[0] == 3 and m2.shape[1] == 12, "shape expected to be (3,12)"
    s = geodesic_similarity(m1, m2)
    assert s.shape[0] == 3 and s.shape[1] == 3, "similarity combos should then be (3,3). similarities shape: {}".format(s)

    # 3! = 6
    similarities = []
    similarities.append(np.mean([s[0][0], s[1][1], s[2][2]]))
    similarities.append(np.mean([s[0][0], s[1][2], s[2][1]]))
    similarities.append(np.mean([s[0][1], s[1][0], s[2][2]]))
    similarities.append(np.mean([s[0][1], s[1][2], s[2][0]]))
    similarities.append(np.mean([s[0][2], s[1][0], s[2][1]]))
    similarities.append(np.mean([s[0][2], s[1][1], s[2][0]]))
    return np.max(similarities)

import numpy as np

from nmf_results import *


def geodesic_similarity(m1, m2):
    u_len_norm = np.max([np.sqrt(np.power(m1, 2).sum()), np.sqrt(np.power(m2, 2).sum())])
    m1 = np.abs(m1) / u_len_norm
    m2 = np.abs(m2) / u_len_norm

    return 1. - np.arccos(np.inner(m1, m2)) * 2./ np.pi


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


def mean_geodesic_similarity(ps1, ps2):
    similarities = []
    for key in ps1:
        cp1 = ps1[key]
        if str(cp1.__class__).__contains__('torch.Tensor'):
            cp1 = cp1.clone().detach().numpy()
        cp2 = ps2[key]
        if str(cp2.__class__).__contains__('torch.Tensor'):
            cp2 = cp2.clone().detach().numpy()
        # cp1 = np.abs(cp1)
        # cp2 = np.abs(cp2)

        similarities.append(geodesic_similarity(cp1, cp2))
    return np.mean(similarities)

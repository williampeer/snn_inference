import numpy as np


def euclidean_similarity(mp, tp):
    euclid_dist = np.sqrt(np.pow(tp-mp), 2)
    return 1. / (1. + euclid_dist)


def mean_euclidean_similarity(ps1, ps2):
    similarities = []
    for key in ps1:
        similarities.append(euclidean_similarity(ps1[key], ps2[key]))
    return np.mean(similarities)

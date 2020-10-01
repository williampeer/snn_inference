import numpy as np


def euclidean_similarity(mp, tp):
    if str(mp.__class__).__contains__('torch.Tensor'):
        mp = mp.clone().detach().numpy()
    if str(tp.__class__).__contains__('torch.Tensor'):
        tp = tp.clone().detach().numpy()
    # mp = mp/mp.shape[0]
    # tp = tp/tp.shape[0]
    # normalise
    abs_max = np.max([np.max(np.abs(mp)), np.max(np.abs(tp))])
    mp = mp/abs_max
    tp = tp/abs_max
    # div by pop size?
    euclid_dist = np.sqrt(np.power(tp-mp, 2).sum())
    return 1 / (1. + euclid_dist)


def mean_euclidean_similarity(ps1, ps2):
    similarities = []
    for key in ps1:
        similarities.append(euclidean_similarity(ps1[key], ps2[key]))
    return np.mean(similarities)

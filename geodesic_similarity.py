import numpy as np


def geodesic_similarity(m1, m2):
    # inner_prod = np.inner(m1, m2)
    # inner_prod[inner_prod > 1.] = 1.
    return 1. - 2./ np.pi * np.arccos(np.inner(m1, m2))

t = np.asarray([
    [0.1927, 0.2310, 0.1808, 0.3006, 0.3047, 0.4500, 0.3168, 0.3364, 0.2481, 0.2636, 0.3467, 0.1623],
    [0.3824, 0.3177, 0.3572, 0.3120, 0.3054, 0.1134, 0.0623, 0.1886, 0.3720, 0.3337, 0.1455, 0.3340],
    [0.2363, 0.2713, 0.3050, 0.0234, 0.1060, 0.0388, 0.6208, 0.2046, 0.0644, 0.1690, 0.3175, 0.4511]])

f1 = np.asarray([
    [0.1521,0.0999,0.2366,0.2007,0.4868,0.2363,0.2161,0.4752,0.2881,0.1376,0.3763,0.2486],
    [0.2910,0.1760,0.1322,0.4627,0.2489,0.5924,0.0551,0.1825,0.0833,0.1244,0.0983,0.4141],
    [0.3676,0.1634,0.2063,0.3491,0.3247,0.3966,0.3371,0.1408,0.2509,0.1387,0.3344,0.2892]
])

o1 = geodesic_similarity(t[0, :], f1[0, :])
o2 = geodesic_similarity(t[0, :], f1[1, :])
o3 = geodesic_similarity(t[0, :], f1[2, :])
print(o1, o2, o3)

o1 = geodesic_similarity(t[1, :], f1[0, :])
o2 = geodesic_similarity(t[1, :], f1[1, :])
o3 = geodesic_similarity(t[1, :], f1[2, :])
print(o1, o2, o3)

o1 = geodesic_similarity(t[2, :], f1[0, :])
o2 = geodesic_similarity(t[2, :], f1[1, :])
o3 = geodesic_similarity(t[2, :], f1[2, :])
print(o1, o2, o3)

print(geodesic_similarity(t, f1))

import numpy as np

from nmf_results import *


def geodesic_similarity(m1, m2):
    # inner_prod = np.inner(m1, m2)
    # inner_prod[inner_prod > 1.] = 1.
    return 1. - 2./ np.pi * np.arccos(np.inner(m1, m2))


o_gd_vrdfrd1 = geodesic_similarity(glif1, gd_vrdfrd1)
print('o_gd_vrdfrd1:', o_gd_vrdfrd1)

from geodesic_similarity import *
from nmf_results import *

print('glif2, r: 0.4 & model 1 - max geodesic similarity:', brute_force_max_geodesic_similarity(glif2_04.T, f1.T))
print('glif2, r: 0.4 & model 2 - max geodesic similarity:', brute_force_max_geodesic_similarity(glif2_04.T, f2.T))

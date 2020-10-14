import numpy as np

from plot_similarities import bar_plot_similarities

# 'fitted_spike_train_glif_ensembles_seed_1_DE_poisson_nll_exp_num_0'
# [NaN]

# 'fitted_spike_train_glif_ensembles_seed_1_CMA_vrdfrd_exp_num_0'
# [0.4986]


# 'fitted_spike_train_glif_ensembles_seed_1_CMA_firing_rate_distance_exp_num_0'
# [0.1287, 0.3709, NaN]

# 'fitted_spike_train_glif_ensembles_seed_1_CMA_poisson_nll_exp_num_0'
# [NaN]

# 'fitted_spike_train_glif_ensembles_seed_1_Adam_vrdfrd_1_lr_0_0005_exp_num_0'
# 'fitted_spike_train_glif_ensembles_seed_1_Adam_vrdfrd_1_lr_0_001_exp_num_0'
# 'fitted_spike_train_glif_ensembles_seed_1_Adam_firing_rate_distance_lr_0_001_exp_num_0'
# 'fitted_spike_train_glif_ensembles_seed_1_Adam_firing_rate_distance_2_lr_0_001_exp_num_0'

CMA_vrdfrd = [0.4986]
CMA_frd = [0.1287, 0.3709]

NGO_frd = [0.0868, 0.1336]

Adam_seed_1_vrdfrd_lr_0_0005 = [0.4092, 0.3855, 0.3636, 0.4779, 0.1750, 0.1236, 0.4362, 0.3651, 0.4779]
Adam_seed_1_vrdfrd_lr_0_001 = [0.3933, 0.4708, 0.4882, 0.4877, 0.1868, 0.4822, 0.1538]

Adam_frd = [0.4753, 0.4807, 0.0912, 0.1030, 0.3112, 0.4128, 0.0947, 0.2139]
Adam_frd_2 = [0.3500, 0.3112, 0.4803, 0.2125, 0.1164, 0.1151, 0.4905, 0.1688, 0.1185, 0.1376, 0.4074, 0.4912, 0.1222]


CMA_vrdfrd_distances = 1. - np.asarray(CMA_vrdfrd)
CMA_frd_distances = 1. - np.asarray(CMA_frd)

NGO_frd_distances = 1. - np.asarray(NGO_frd)

Adam_seed_1_vrdfrd_lr_0_0005_distances = 1. - np.asarray(Adam_seed_1_vrdfrd_lr_0_0005)
Adam_seed_1_vrdfrd_lr_0_001_distances = 1. - np.asarray(Adam_seed_1_vrdfrd_lr_0_001)

Adam_frd_distances = 1. - np.asarray(Adam_frd)
Adam_frd_2_distances = 1. - np.asarray(Adam_frd_2)


CMA = [CMA_vrdfrd_distances, CMA_frd_distances]
NGO = [[0.], NGO_frd_distances]
GDs_0_0005 = [Adam_seed_1_vrdfrd_lr_0_0005_distances, Adam_frd_distances]
GDs_0_001 = [Adam_seed_1_vrdfrd_lr_0_001_distances, Adam_frd_2_distances]
all_exp_res = [CMA, NGO, GDs_0_0005, GDs_0_001]

bar_plot_similarities(all_exp_res, xticks=['CMA', 'NGO', 'Adam, $\\alpha=0.05\ \%$', 'Adam, $\\alpha=0.1\ \%$'],
                      legends=['vRD + $f_r$-penalty', '$f_d + f_r$-penalty'])

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
from plot_similarities import bar_plot_similarities

Adam_seed_1_vrdfrd_lr_0_0005 = [[0.4092, 0.3855, 0.3636, 0.4779, 0.1750, 0.1236, 0.4362, 0.3651, 0.4779]]
Adam_seed_1_vrdfrd_lr_0_001 = [[0.3933, 0.4708, 0.4882, 0.4877, 0.1868, 0.4822, 0.1538]]

Adam_frd = [[0.4753, 0.4807, 0.0912, 0.1030, 0.3112, 0.4128, 0.0947, 0.2139]]
Adam_frd_2 = [[0.3500, 0.3112, 0.4803, 0.2125, 0.1164, 0.1151, 0.4905, 0.1688, 0.1185, 0.1376, 0.4074, 0.4912, 0.1222]]



CMA_vrdfrd_converged = [0.4986]
CMA_frd_converged = [0.3709]

Adam_seed_1_vrdfrd_lr_0_0005_converged = [[0.4092, 0.3855, 0.3636, 0.4779, 0.4362, 0.3651, 0.4779]]
Adam_seed_1_vrdfrd_lr_0_001_converged = [[0.3933, 0.4708, 0.4882, 0.4877, 0.4822]]

Adam_frd_converged = [[0.4753, 0.4807, 0.3112, 0.4128]]
Adam_frd_2_converged = [[0.3500, 0.3112, 0.4803, 0.4905, 0.4074, 0.4912]]

EAs = [CMA_vrdfrd_converged, CMA_frd_converged]
GDs = [Adam_seed_1_vrdfrd_lr_0_0005_converged, Adam_seed_1_vrdfrd_lr_0_001_converged, Adam_frd_converged, Adam_frd_2_converged]
all_exp_res = [EAs, GDs]

bar_plot_similarities(all_exp_res, xticks=['EAs', 'Adam'], legends=['vRD + $f_r$-penalty, $\\alpha=0.05\ \%$', 'vRD + $f_r$-penalty, $\\alpha=0.1\ \%$',
                                                                    '$f_d + f_r$-penalty, $\\alpha=0.1\ \%$', '$f_d + f_r$-penalty, $\\alpha=0.1\ \%$'])

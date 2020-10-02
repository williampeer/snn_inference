import numpy as np

from plot_similarities import bar_plot_similarities

gd_vrdfrd_glif1_0_4 = np.array([0.3577, 0.3547, 0.3913, 0.3369, 0.3685, 0.3368, 0.4145, 0.2707, 0.3188])

generated_spike_train_glif1_0_4_fit_kl_div = np.array( [0.2444, 0.3638, 0.3387])
generated_spike_train_glif2_fit_kl_div = np.array([0.3142, 0.3310, 0.2964, 0.2812])
generated_spike_train_async_0_4_fit_kl_div = np.array([0.4735, 0.3115, 0.4653, 0.3159])
generated_spike_train_glif_slower_more_sync_fit_kl_div = np.array([0.3297, 0.3348, 0.2856])
kl_div_exp = [generated_spike_train_glif1_0_4_fit_kl_div, generated_spike_train_glif2_fit_kl_div, generated_spike_train_async_0_4_fit_kl_div, generated_spike_train_glif_slower_more_sync_fit_kl_div]

generated_spike_train_glif1_0_4_fit_poisson_nll = np.array([0.3137, 0.3616, 0.2975, 0.2933, 0.3232])
generated_spike_train_glif2_0_4_fit_poisson_nll = np.array([0.3657, 0.4536, 0.4152, 0.3529, 0.2951])
generated_spike_train_async_0_4_fit_poisson_nll = np.array([0.4464, 0.4759, 0.4582, 0.3269, 0.4176])
generated_spike_train_glif_slower_more_sync_fit_poisson_nll = np.array([0.3596, 0.3948, 0.3729, 0.3660, 0.3409])
pnll_exp = [generated_spike_train_glif1_0_4_fit_poisson_nll, generated_spike_train_glif2_0_4_fit_poisson_nll, generated_spike_train_async_0_4_fit_poisson_nll, generated_spike_train_glif_slower_more_sync_fit_poisson_nll]

generated_spike_train_glif1_0_4_fit_vrd = np.array([0.3128, 0.2898, 0.3032])
generated_spike_train_glif2_0_4_fit_vrd = np.array( [0.3447, 0.3947, 0.3591, 0.3673])
generated_spike_train_async_0_4_fit_vrd = np.array( [0.3674, 0.3567])
generated_spike_train_glif_slower_more_sync_fit_vrd = np.array([0.3517, 0.3514, 0.2526, 0.2919])
vrd_exp = [generated_spike_train_glif1_0_4_fit_vrd, generated_spike_train_glif2_0_4_fit_vrd, generated_spike_train_async_0_4_fit_vrd, generated_spike_train_glif_slower_more_sync_fit_vrd]

generated_spike_train_glif1_0_4_2_and_3_fit_vrdfrd = np.array([0.1920, 0.2334, 0.2366, 0.2536, 0.2800, 0.2675, 0.2138])
generated_spike_train_glif2_fit_vrdfrd = np.array( [0.2947, 0.3738, 0.2734])
generated_spike_train_async_0_4_fit_vrdfrd = np.array([0.3352, 0.3028, 0.3725])
generated_spike_train_glif_slower_more_sync_fit_vrdfrd = np.array( [0.3927, 0.2812, 0.2520, 0.2456])
vrdfrd_exp = [generated_spike_train_glif1_0_4_2_and_3_fit_vrdfrd, generated_spike_train_glif2_fit_vrdfrd, generated_spike_train_async_0_4_fit_vrdfrd, generated_spike_train_glif_slower_more_sync_fit_vrdfrd]

# ------------------------------------------ EAs -----------------------------------------------------
spikes_brian_params_by_optim_optim_DE_loss_fn_gamma_factor_budget_2000 = np.array([0.2717, 0.2765, 0.2660, 0.3289, 0.2627, 0.2777, 0.3119, 0.3009, 0.2987, 0.2305])
spikes_brian_params_by_optim_optim_DE_loss_fn_poisson_nll_budget_2000 = np.array([0.4075, 0.3411, 0.3181, 0.3752, 0.3748, 0.3328, 0.3925, 0.4132, 0.3365, 0.3338])
spikes_brian_params_by_optim_optim_DE_loss_fn_van_rossum_dist_budget_2000 = np.array([0.3691, 0.3795, 0.3585, 0.3384, 0.4027, 0.3735, 0.3449, 0.3974])
spikes_brian_params_by_optim_optim_DE_loss_fn_vrdfrd_budget_2000 = np.array([0.3158, 0.2820, 0.2647, 0.3235, 0.2904, 0.3068, 0.2964, 0.2886, 0.3016])
DE_exps = [spikes_brian_params_by_optim_optim_DE_loss_fn_gamma_factor_budget_2000, spikes_brian_params_by_optim_optim_DE_loss_fn_poisson_nll_budget_2000, spikes_brian_params_by_optim_optim_DE_loss_fn_van_rossum_dist_budget_2000, spikes_brian_params_by_optim_optim_DE_loss_fn_vrdfrd_budget_2000]

spikes_brian_params_by_optim_optim_CMA_loss_fn_poisson_nll_budget_2000 = np.array([0.3524, 0.3692, 0.3811, 0.3777, 0.2901, 0.3434, 0.3977, 0.3307, 0.3747])
spikes_brian_params_by_optim_optim_CMA_loss_fn_van_rossum_dist_budget_2000 = np.array([0.3125, 0.3379, 0.3464, 0.3221, 0.3300, 0.4072, 0.2922, 0.3491])
spikes_brian_params_by_optim_optim_CMA_loss_fn_vrdfrd_budget_2000 = np.array([0.2853, 0.3143, 0.3251, 0.3051, 0.2785, 0.2853, 0.2559, 0.3391, 0.3457])
CMA_exps = [spikes_brian_params_by_optim_optim_CMA_loss_fn_poisson_nll_budget_2000, spikes_brian_params_by_optim_optim_CMA_loss_fn_van_rossum_dist_budget_2000, spikes_brian_params_by_optim_optim_CMA_loss_fn_vrdfrd_budget_2000]

# bar_plot_similarities([kl_div_exp, pnll_exp, vrd_exp, vrdfrd_exp], xticks=['GLIF 1', 'GLIF 2', 'GLIF async', 'GLIF slower sync'], legends=['KL divergence', 'Poisson NLL', 'van Rossum dist (vRD)', 'vRD + $f_r$-penalty'])

glif1_exp = [generated_spike_train_glif1_0_4_fit_kl_div, generated_spike_train_glif1_0_4_fit_poisson_nll, generated_spike_train_glif1_0_4_fit_vrd, generated_spike_train_glif1_0_4_2_and_3_fit_vrdfrd]
glif2_exp = [generated_spike_train_glif2_fit_kl_div, generated_spike_train_glif2_0_4_fit_poisson_nll, generated_spike_train_glif2_0_4_fit_vrd, generated_spike_train_glif2_fit_vrdfrd]
glifasync_exp = [generated_spike_train_async_0_4_fit_kl_div, generated_spike_train_async_0_4_fit_poisson_nll, generated_spike_train_async_0_4_fit_vrd, generated_spike_train_async_0_4_fit_vrdfrd]
glifsync_exp = [generated_spike_train_glif_slower_more_sync_fit_kl_div, generated_spike_train_glif_slower_more_sync_fit_poisson_nll, generated_spike_train_glif_slower_more_sync_fit_vrd, generated_spike_train_glif_slower_more_sync_fit_vrdfrd]
bar_plot_similarities([glif1_exp, glif2_exp, glifasync_exp, glifsync_exp], xticks=['GLIF 1', 'GLIF 2', 'GLIF async', 'GLIF slower sync'], legends=['KL divergence', 'Poisson NLL', 'van Rossum dist (vRD)', 'vRD + $f_r$-penalty'])

bar_plot_similarities([DE_exps], xticks=['GLIF 1'], legends=['$\Gamma$-factor', 'Poisson NLL', 'van Rossum dist (vRD)', 'vRD + $f_r$-penalty'], title='Geodesic similarity exp. 1 DE')
bar_plot_similarities([CMA_exps], xticks=['GLIF 1'], legends=['Poisson NLL', 'van Rossum dist (vRD)', 'vRD + $f_r$-penalty'], title='Geodesic similarity exp. 1 CMA')


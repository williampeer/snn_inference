from Dev.brian2_custom_network_opt import *
import nevergrad as ng
from Log import Logger
from experiments import zip_dicts

from plot import plot_all_param_pairs_with_variance

# TODO: PSO, NGO, CMA(?)
num_exps = 5; budget = 4
# for optim in [ng.optimizers.DE, ng.optimizers.CMA, ng.optimizers.PSO, ng.optimizers.NGO]:
params_by_optim = {}
for optim in [ng.optimizers.DE]:
    current_plottable_params_for_optim = {}
    other_params_for_optim = {}
    for exp_i in range(num_exps):
        N = 12
        w_mean = 0.3; w_var = 0.5; rand_ws = (w_mean - w_var) + 2 * w_var * np.random.random((N ** 2))
        instrum = ng.p.Instrumentation(rate=ng.p.Scalar(init=0.6).set_bounds(0.01, 1.),
                                       w=ng.p.Array(init=rand_ws).set_bounds(-1., 1.),
                                       f_I=ng.p.Array(init=0.4 * np.ones((N,))).set_bounds(0.01, 0.99),
                                       C_m=ng.p.Array(init=1.5 * np.ones((N,))).set_bounds(1., 3.),
                                       G=ng.p.Array(init=0.8 * np.ones((N,))).set_bounds(0.01, 0.99),
                                       R_I=ng.p.Array(init=18. * np.ones((N,))).set_bounds(10., 30.),
                                       f_v=ng.p.Array(init=0.14 * np.ones((N,))).set_bounds(0.01, 0.99),
                                       E_L=ng.p.Array(init=-65. * np.ones((N,))).set_bounds(-90., -30.),

                                       b_s=ng.p.Array(init=0.3 * np.ones((N,))).set_bounds(0.01, 0.9),
                                       b_v=ng.p.Array(init=0.5 * np.ones((N,))).set_bounds(0.01, 0.9),
                                       a_v=ng.p.Array(init=0.5 * np.ones((N,))).set_bounds(0.01, 0.9),
                                       delta_theta_s=ng.p.Array(init=30. * np.ones((N,))).set_bounds(1., 40.),
                                       delta_V=ng.p.Array(init=12. * np.ones((N,))).set_bounds(0.01, 35.),
                                       theta_inf=ng.p.Array(init=-20. * np.ones((N,))).set_bounds(-25., 0.),
                                       I_A=ng.p.Array(init=8. * np.ones((N,))).set_bounds(0.5, 10.),

                                       loss_fn=ng.p.Choice(['van_rossum_dist', 'poisson_nll', 'gamma_factor']))

        optimizer = optim(parametrization=instrum, budget=budget)

        logger = Logger(log_fname='brian2_network_nevergrad_optimization_budget_{}'.format(budget))
        logger.log('setup experiment with the optimizer {}'.format(optimizer.__str__()))

        recommendation = optimizer.minimize(run_simulation_for, verbosity=2)

        logger.log('recommendation.value: {}'.format(recommendation.value))
        fitted_params = recommendation.value[1]
        for p_i, key in enumerate(fitted_params):
            if key not in ['loss_fn', 'rate', 'w']:
                if exp_i == 0:
                    current_plottable_params_for_optim[p_i] = [np.copy(fitted_params[key])]
                else:
                    current_plottable_params_for_optim[p_i].append(np.copy(fitted_params[key]))
            else:
                if exp_i == 0:
                    other_params_for_optim[key] = [np.copy(fitted_params[key])]
                else:
                    other_params_for_optim[key].append(np.copy(fitted_params[key]))

    params_by_optim[optim.name] = zip_dicts(current_plottable_params_for_optim, other_params_for_optim)

    plot_all_param_pairs_with_variance(current_plottable_params_for_optim,
                                       exp_type='nevergrad',
                                       uuid='single_objective_optim',
                                       target_params=target_parameters,
                                       param_names=list(fitted_params.keys())[:3],
                                       custom_title="KDEs for values across experiments ({})".format(optim.name),
                                       logger=logger, fname='single_objective_KDE_optim_{}'.format(optim.name))

torch.save(params_by_optim, './saved/single_objective_fitted_params_optim_{}_budget_{}.pt'.format(optim, budget))
